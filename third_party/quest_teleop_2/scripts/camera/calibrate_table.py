#!/usr/bin/env python
"""Map top-down camera pixels to table coordinates, so the arm can be sent to a
seen cube.

    python scripts/camera/calibrate_table.py
    python scripts/camera/calibrate_table.py --object cube --out config/table.yaml

The cube's pixel is the centre of the SAM 3 mask of --object, the same pixel the reach
places later. --no-sam takes the background detector's box centre instead.

Never commands motion. It opens the robot receive-only and reads where you have
put the arm, so position it by freedrive or teleoperation.

For each point, twice over, at four or more well-spread spots:

  1. Put the cube down and move the arm OUT of the camera's view.
  2. Press `c` -- the cube's pixel is captured.
  3. Move the arm so the TCP is centred directly over that cube.
  4. Press `t` -- the TCP's x, y is captured.

Two steps because the hand hides the cube from a top-down camera; you cannot
see it and touch it at the same time.

Press `s` to solve and save. Spread the points across the whole area you intend
to grasp in -- including where evaluation will go, not just where demos are
recorded. A homography fitted to one corner extrapolates badly to the rest.

Calibrate with the object you will actually grasp. The detected pixel is the
object's TOP face, whose centre sits directly above its base centre, so
pixel -> (x, y) is exactly a homography for that height and the parallax is
absorbed into the fit. Change the object's height and this must be redone.

Press `p` once, with the hand held in the attitude you want to approach from,
to store the template pose. That is where the orientation and height of every
scripted approach come from. Skip it to keep the template already on file,
which trained policies depend on.

Keys
    c         capture the cube's pixel (arm clear of the view)
    t         capture the TCP position (arm over the cube)
    p         store the current full TCP pose as the approach template
    u         undo the last complete point
    s         solve, report the error, and save
    q         quit without saving
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quest_teleop2.robot.rtde_io import RTDEInterface  # noqa: E402
from scripts._common import add_common_args, build_camera_rig, load_configs, setup_logging  # noqa: E402
from quest_teleop2.vision.table import detect_object  # noqa: E402

GREEN, YELLOW, RED, BOLD, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[1m", "\033[0m"

#: Below this many points the homography is exactly determined and its
#: reprojection error is meaningless -- it fits its own inputs perfectly and
#: tells you nothing about the rest of the table.
MIN_POINTS = 4

#: Reprojection error above which the fit is not worth trusting, metres. The
#: grasp policy is trained to absorb approach error, so this need not be
#: surgical; it needs to be small against the noise deliberately injected at
#: demo time (+-3 cm).
MAX_RMS_M = 0.010


def solve(pixels: np.ndarray, table: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit pixel -> table and report how well it reprojects.

    A homography, not a full camera calibration, because the cube is confined
    to a plane. The projective map between two planes has 8 degrees of freedom
    and needs no intrinsics, no distortion model and no depth -- four point
    pairs determine it. Any out-of-plane structure would break that, which is
    why this is only valid for objects sitting on the table.
    """
    import cv2  # noqa: PLC0415

    H, _ = cv2.findHomography(pixels.reshape(-1, 1, 2), table.reshape(-1, 1, 2),
                              cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("findHomography failed -- are the points collinear?")
    reproj = cv2.perspectiveTransform(pixels.reshape(-1, 1, 2).astype(np.float64), H)
    err = np.linalg.norm(reproj.reshape(-1, 2) - table, axis=1)
    return H, float(np.sqrt((err ** 2).mean())), err


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", default="static",
                        help="which camera looks down at the table")
    parser.add_argument("--out", type=Path, default=Path("config/table.yaml"),
                        help="where to write the homography")
    parser.add_argument("--scale", type=float, default=1.5, help="display magnification")
    parser.add_argument("--object", default="cube",
                        help="what SAM 3 looks for: a config/objects.yaml name or a short "
                             "phrase")
    parser.add_argument("--sam", action=argparse.BooleanOptionalAction, default=True,
                        help="find the object with SAM 3 (the default), as the reach does; "
                             "--no-sam uses the background detector")
    parser.add_argument("--template-only", action="store_true",
                        help="re-capture just the approach attitude, leaving the "
                             "homography and its points untouched. No camera, no "
                             "geometry -- the template is only a TCP pose")
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging(args.verbose)
    robot_cfg, _, _ = load_configs(args)

    if args.template_only:
        return capture_template(robot_cfg, args.out)

    import cv2  # noqa: PLC0415

    rig, names = build_camera_rig(args.camera)
    if rig is None or not names:
        print(f"{RED}No camera named {args.camera!r} in config/cameras.yaml{RESET}")
        return 1
    # lock=True pins exposure and white balance, as every other tool does. The
    # background model compares Lab distance against a locked capture, so an
    # auto-exposed frame differs from it everywhere at once and the detector
    # sees one frame-sized blob instead of the object.
    rig.start(lock=True)

    sam = spec = flash = None
    if args.sam:
        from quest_teleop2.vision.sam import SamClient, decide_pixel, draw, object_spec  # noqa: PLC0415

        try:
            spec, _ = object_spec(args.object, need_height=False)
            print(f"  starting SAM 3 for '{spec.prompt}'...", end="", flush=True)
            sam = SamClient().start()
            print(" ok")
        except (ValueError, OSError, RuntimeError, TimeoutError) as exc:
            print(f"\n{RED}{exc}{RESET}")
            rig.stop()
            return 1
    locator = f"sam3:{spec.name}" if sam is not None else "background"

    pixels: list[tuple[float, float]] = []
    table: list[tuple[float, float]] = []
    poses: list[list[float]] = []
    pending_px: tuple[float, float] | None = None
    template: list[float] | None = None

    print(f"\n{BOLD}Table calibration{RESET}  camera={args.camera}")
    print("  c capture cube pixel (arm clear) | t capture TCP (arm over cube)")
    print("  p store approach template | u undo | s solve and save | q quit\n")

    rtde = RTDEInterface(robot_cfg)
    try:
        print("  connecting to the robot...", end="", flush=True)
        rtde.connect()
        print(" ok")
    except ConnectionError as exc:
        # The camera is already open, and leaving it that way blocks the next
        # attempt: a V4L2 device opens once.
        print()
        print(f"{RED}{exc}{RESET}")
        rig.stop()
        return 1
    win = "calibrate table"
    try:
        waited = 0.0
        while True:
            frame, _age, _seq = rig.cameras[0].snapshot()
            if frame is None:
                # Spinning here with no sleep and no window is indistinguishable
                # from a hang: no picture, no keys, one core pegged. Say so
                # instead, since a camera that has not delivered in five seconds
                # is not about to.
                time.sleep(0.02)
                waited += 0.02
                if waited > 5.0:
                    print(f"\n{RED}No frames from camera {args.camera!r} after 5 s.{RESET}")
                    print("  Another process may be holding it -- a V4L2 device opens once.")
                    return 1
                continue
            waited = 0.0
            box, score = None, 0.0
            if sam is None:
                box, score, _floor = detect_object(frame)
            if flash is not None and time.monotonic() < flash[1]:
                shown = flash[0].copy()
            else:
                shown = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
            if pending_px is not None:
                cv2.drawMarker(shown, (int(pending_px[0]), int(pending_px[1])), (0, 255, 255),
                               cv2.MARKER_CROSS, 16, 2)
            if box is not None:
                x, y, w, h = box
                cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(shown, (int(x + w / 2), int(y + h / 2)), 4, (0, 255, 0), -1)
            for i, (u, v) in enumerate(pixels):
                cv2.circle(shown, (int(u), int(v)), 6, (255, 128, 0), 2)
                cv2.putText(shown, str(i), (int(u) + 8, int(v)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
            stage = "now press t with the TCP over the cube" if pending_px else \
                    ("press c with the arm CLEAR" if box is not None or sam is not None
                     else "no cube seen")
            cv2.putText(shown, f"{len(pixels)} points | {stage}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            tmpl = "template SET" if template is not None else "no template (press p)"
            cv2.putText(shown, tmpl, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if template is not None else (0, 165, 255), 2)
            if args.scale != 1.0:
                shown = cv2.resize(shown, None, fx=args.scale, fy=args.scale)
            cv2.imshow(win, shown)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                print("Quit without saving.")
                return 1
            if key == ord("c"):
                if sam is not None:
                    try:
                        decision = decide_pixel(sam.segment(frame, [spec.prompt]), spec)
                    except (OSError, RuntimeError) as exc:
                        print(f"  {RED}SAM 3 failed: {exc}{RESET}")
                        continue
                    flash = (cv2.cvtColor(draw(frame, decision), cv2.COLOR_RGB2BGR),
                             time.monotonic() + 1.5)
                    if decision.target is None:
                        print(f"  {YELLOW}{decision.reason} -- is the arm out of the "
                              f"view?{RESET}")
                        continue
                    pending_px = (float(decision.target.px[0]), float(decision.target.px[1]))
                    score = decision.target.score
                elif box is None:
                    print(f"  {YELLOW}no cube detected -- is the arm blocking the view?{RESET}")
                    continue
                else:
                    x, y, w, h = box
                    pending_px = (x + w / 2.0, y + h / 2.0)
                print(f"  cube at pixel ({pending_px[0]:.1f}, {pending_px[1]:.1f}), "
                      f"score {score:.2f}. Now move the TCP over it and press t.")
            elif key == ord("t"):
                if pending_px is None:
                    print(f"  {YELLOW}capture the cube pixel first, with c{RESET}")
                    continue
                state = rtde.receive()
                if state is None:
                    print(f"  {RED}no RTDE data{RESET}")
                    continue
                tx, ty = float(state.actual_tcp_pose[0]), float(state.actual_tcp_pose[1])
                pixels.append(pending_px)
                table.append((tx, ty))
                # The full pose is free to keep here and impossible to recover
                # later without repeating the whole physical procedure. It is
                # the fallback template, and it lets the fit be re-solved with
                # height taken into account if that ever becomes necessary.
                poses.append([round(float(v), 5) for v in state.actual_tcp_pose])
                print(f"  {GREEN}point {len(pixels) - 1}{RESET}: "
                      f"pixel ({pending_px[0]:.1f}, {pending_px[1]:.1f}) -> "
                      f"table ({tx:+.4f}, {ty:+.4f})")
                pending_px = None
            elif key == ord("p"):
                state = rtde.receive()
                if state is None:
                    print(f"  {RED}no RTDE data{RESET}")
                    continue
                template = [round(float(v), 5) for v in state.actual_tcp_pose]
                print(f"  {GREEN}template pose{RESET}: {template}")
                print("   every scripted approach inherits this orientation and height.")
            elif key == ord("u") and pixels:
                pixels.pop()
                table.pop()
                poses.pop()
                print(f"  undone, {len(pixels)} left")
            elif key == ord("s"):
                if len(pixels) < MIN_POINTS:
                    print(f"  {YELLOW}need at least {MIN_POINTS} points, have "
                          f"{len(pixels)}{RESET}")
                    continue
                H, rms, err = solve(np.array(pixels, dtype=np.float64),
                                    np.array(table, dtype=np.float64))
                print(f"\n  reprojection error per point (mm): "
                      f"{np.round(err * 1000, 1).tolist()}")
                print(f"  RMS {rms * 1000:.1f} mm over {len(pixels)} points")
                # Let the fit be rejected. The per-point errors only exist
                # after solving, so without this the choice is write a bad fit
                # or lose every point and start over -- and the fix is usually
                # to drop one outlier and add two more, which costs a minute.
                if rms > MAX_RMS_M and len(pixels) > MIN_POINTS:
                    print(f"  {YELLOW}Keep capturing to improve it, or write it "
                          f"anyway?{RESET}")
                    print("   The worst point is "
                          f"#{int(np.argmax(err))} at {err.max() * 1000:.1f} mm -- "
                          "u undoes the last one.")
                    try:
                        if input("   Write this fit? [y/N] ").strip().lower() \
                                not in ("y", "yes"):
                            print("   Not written. Add points and press s again.")
                            continue
                    except (EOFError, KeyboardInterrupt):
                        print("\n   Not written.")
                        continue
                if len(pixels) == MIN_POINTS:
                    print(f"  {YELLOW}With exactly {MIN_POINTS} points the fit is exact "
                          f"and this error is meaningless.{RESET}")
                    print("  Add a fifth before trusting it.")
                if rms > MAX_RMS_M:
                    print(f"  {YELLOW}Above the {MAX_RMS_M * 1000:.0f} mm target. "
                          f"Check the TCP really was centred over the cube.{RESET}")
                if template is None:
                    template = _existing_template(args.out)
                    if template is not None:
                        print(f"  keeping the existing approach template {template}")
                if template is None:
                    # The template sets the orientation AND height of every
                    # approach in every episode from here on. Falling back to an
                    # incidental capture is a silent way to get all of them
                    # slightly wrong, so make it a decision rather than a note.
                    print(f"\n  {RED}No approach template captured.{RESET} p stores "
                          f"the attitude every")
                    print("   approach inherits; without it the last TCP capture is "
                          "used, which")
                    print("   was wherever your hand happened to be for the final "
                          "point.")
                    try:
                        if input("   Use the last capture anyway? [y/N] ").strip() \
                                .lower() not in ("y", "yes"):
                            print("   Not written. Put the hand in your grasp "
                                  "attitude, press p, then s.")
                            continue
                    except (EOFError, KeyboardInterrupt):
                        print("\n   Not written.")
                        continue
                    template = poses[-1]
                _write(args.out, H, pixels, table, poses, template, rms, args.camera, locator)
                print(f"\n  {GREEN}written to {args.out}{RESET}")
                return 0
    finally:
        cv2.destroyAllWindows()
        rig.stop()
        rtde.disconnect()
        if sam is not None:
            sam.close()


def _existing_template(path: Path) -> list[float] | None:
    """The template already on disk: policies were trained on its attitude and height."""
    import yaml  # noqa: PLC0415

    if not path.exists():
        return None
    t = (yaml.safe_load(path.read_text()) or {}).get("template_pose")
    return [float(v) for v in t] if t is not None and len(t) == 6 else None


def capture_template(robot_cfg, path: Path) -> int:
    """Replace only `template_pose` in an existing calibration.

    The template is a TCP pose and nothing else -- the attitude and height every
    approach inherits. It has no bearing on the homography, so re-capturing it
    does not mean redoing the point fit, which is the expensive half.

    Needs no camera: there is nothing to see. Position the hand by freedrive and
    the pose is read straight off RTDE.
    """
    import yaml  # noqa: PLC0415

    if not path.exists():
        print(f"{RED}No calibration at {path} to update.{RESET}")
        print("  Run the full procedure first, without --template-only.")
        return 1
    data = yaml.safe_load(path.read_text()) or {}
    previous = data.get("template_pose")

    rtde = RTDEInterface(robot_cfg)
    try:
        print("  connecting to the robot...", end="", flush=True)
        rtde.connect()
        print(" ok")
    except ConnectionError as exc:
        print()
        print(f"{RED}{exc}{RESET}")
        return 1

    try:
        print(f"\n{BOLD}Approach template{RESET}  -> {path}")
        if previous is not None:
            print(f"  current: {[round(v, 4) for v in previous]}")
        print("  Put the hand where you want every approach to START from: the")
        print("  attitude you grasp in, at the height you grasp from. Freedrive")
        print("  it there -- nothing is commanded.")
        try:
            input("\n  Press ENTER to capture, Ctrl-C to leave it unchanged... ")
        except (EOFError, KeyboardInterrupt):
            print("\n  Unchanged.")
            return 1

        state = rtde.receive()
        if state is None:
            print(f"  {RED}no RTDE data{RESET}")
            return 1
        pose = [round(float(v), 5) for v in state.actual_tcp_pose]
        print(f"  captured: {pose}")
        if previous is not None:
            moved = np.linalg.norm(np.asarray(pose[:3]) - np.asarray(previous[:3]))
            print(f"  {moved * 1000:.1f} mm from the previous template "
                  f"(z {previous[2]:.4f} -> {pose[2]:.4f})")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"{path.stem}.{stamp}.bak{path.suffix}")
        backup.write_bytes(path.read_bytes())
        data["template_pose"] = [float(v) for v in pose]
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        print(f"\n  {GREEN}written{RESET} -- previous kept as {backup.name}")
        print(f"  The homography and its {len(data.get('points', []))} points are "
              f"untouched.")
        return 0
    finally:
        rtde.disconnect()


def _write(path: Path, H: np.ndarray, pixels, table, poses, template,
           rms: float, camera: str, locator: str = "background") -> None:
    """Save the homography, the approach template, and the points behind both.

    The points are kept so the fit can be re-solved or extended later without
    redoing the physical procedure -- which needs the robot, the cube and an
    operator, and is the expensive part.

    `approach_offset` is carried over from whatever is already on disk. It
    describes where the WRIST camera sits relative to the TCP, which this
    procedure never measures, and TableMap.load defaults it to zeros when
    absent -- so writing a fresh file without it silently moved every approach
    by the offset's magnitude, with nothing to indicate why.
    """
    import yaml  # noqa: PLC0415

    carried = {}
    if path.exists():
        # A calibration costs ten careful minutes with the robot, the object
        # and an operator; the new one is not known to be better until it has
        # been used. Keep the old one rather than discovering that afterwards.
        from datetime import datetime  # noqa: PLC0415

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"{path.stem}.{stamp}.bak{path.suffix}")
        backup.write_bytes(path.read_bytes())
        previous = yaml.safe_load(path.read_text()) or {}
        print(f"   previous calibration kept as {backup.name} "
              f"(RMS {float(previous.get('rms_error_m', float('nan'))) * 1000:.1f} mm)")
        if "approach_offset" in previous:
            carried["approach_offset"] = previous["approach_offset"]
            print(f"   carrying over approach_offset {previous['approach_offset']} "
                  "-- re-tune it if the hand or wrist camera moved")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({
        "camera": camera,
        "locator": locator,
        "homography": [[float(v) for v in row] for row in H],
        "rms_error_m": float(rms),
        "template_pose": [float(v) for v in template],
        "points": [{"pixel": [float(u), float(v)], "table": [float(x), float(y)],
                    "pose": [float(v) for v in pose]}
                   for (u, v), (x, y), pose in zip(pixels, table, poses)],
        **carried,
    }, sort_keys=False))


if __name__ == "__main__":
    raise SystemExit(main())

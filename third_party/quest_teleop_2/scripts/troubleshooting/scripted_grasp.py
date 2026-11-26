#!/usr/bin/env python
"""The classical baseline: locate the cube, move to it, close, lift, score.

    python scripts/troubleshooting/scripted_grasp.py --reach-only --trials 10
    python scripts/troubleshooting/scripted_grasp.py --reach-only --wait --offset-x 0
    python scripts/troubleshooting/scripted_grasp.py --reach-only --noise-xy 0.03
    python scripts/troubleshooting/scripted_grasp.py --trials 50 --close-from data/probe.hdf5

Two jobs, and the first one is what you run right after calibrating.

--reach-only exercises the whole classical half with the hand de-energised:
locate the cube from the top-down camera, map the pixel through the homography,
move there, report, retreat. Run it with the default zero noise first -- the arm
should end up centred over the cube, and if it does not the calibration is
wrong and nothing downstream can work. Then add --noise-xy to see the spread a
policy will have to absorb.

Without --reach-only it becomes the go/no-go gate for the whole project: add a
FIXED closing pose and a lift, and measure how often a script alone succeeds.

    > 70%    the task is scriptable. Do not record demos against it -- raise
             difficulty first (randomise cube orientation, or a harder object).
    20-50%   the window. There is room for a policy to be worth training.
    < 10%    the approach or the calibration is broken, not the task.

The closing pose is not invented here. --close-from reads it out of recorded
demonstrations, because a pose that a human actually grasped this cube with is
the only one known to be safe and effective on this hand. Record a handful of
throwaway demos first; three is enough.

Every trial homes the arm first, since the top-down camera cannot see the cube
through it.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quest_teleop2.data.schema import N_JOINTS  # noqa: E402
from quest_teleop2.robot.grasp import HOLD_THRESHOLD, LIFT_M, grasp_residual  # noqa: E402
from quest_teleop2.robot.safety import SafetyAbort  # noqa: E402
from quest_teleop2.robot.ur5e_arm import UR5eArm  # noqa: E402
from quest_teleop2.teleop.hand_retarget import urdf_limits  # noqa: E402
from quest_teleop2.vision.table import TableMap, locate_cube, pre_grasp_pose  # noqa: E402
from scripts._common import (  # noqa: E402
    add_common_args,
    add_hand_args,
    add_sam_args,
    sam_object,
    start_sam,
    stop_sam,
    build_camera_rig,
    build_hand,
    confirm,
    hand_side,
    view_loop,
    start_video,
    stop_video,
    hand_urdf,
    load_configs,
    setup_logging,
)

_log = logging.getLogger(__name__)

GREEN, YELLOW, RED, BOLD, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[1m", "\033[0m"

#: Seconds to hold the closing command before lifting. The hand's own low-pass
#: runs at 5 Hz, so a lift commanded immediately after the close would start
#: while the fingers are still moving.
#: Ramp the fingers rather than stepping them: the hand has no e-stop.
CLOSE_RAMP_S = 1.0
OPEN_S = 0.8
CLOSE_SETTLE_S = 1.5

#: Seconds to wait at the top of the lift before reading the fingers. Same
#: reason, and it also lets a marginal grasp finish failing.
LIFT_SETTLE_S = 1.0


def _save_offset(path: Path | None, offset: np.ndarray) -> None:
    """Merge the approach offset into the calibration, leaving the rest alone.

    Rewritten in place rather than appended: the homography and the points that
    produced it cost a robot, a cube and an operator to obtain, and must
    survive a tuning session that only meant to nudge an aim point.
    """
    import yaml  # noqa: PLC0415

    p = Path(path) if path is not None else Path("config/table.yaml")
    d = yaml.safe_load(p.read_text())
    d["approach_offset"] = [float(offset[0]), float(offset[1])]
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    print(f"  {GREEN}approach_offset written to {p}{RESET}: "
          f"({offset[0] * 1000:+.0f}, {offset[1] * 1000:+.0f}) mm")


def _floored(pose: np.ndarray, floor_z: float) -> np.ndarray:
    """Never command the TCP below the taught grasp height.

    Belt and braces over the one-sided z noise: this holds regardless of how
    the target was arrived at, so a future flag or an arithmetic slip cannot
    drive the hand through the table. The taught height is the floor because it
    is the one z known to have been safe -- an operator held the hand there.
    """
    pose = np.asarray(pose, dtype=np.float64).reshape(6)
    if pose[2] < floor_z:
        pose = pose.copy()
        pose[2] = floor_z
    return pose


def closing_pose_from(dataset: Path, side: str) -> np.ndarray:
    """The commanded hand pose at peak finger stall, over successful demos.

    Peak stall is the moment the fingers were pressing hardest on the object,
    so the command issued there is a grasp that demonstrably held on this
    hardware. Taking the median across demos rather than one demo's value keeps
    a single odd take from defining the baseline.

    `side` must match the hand that produced the dataset. The two hands are
    mirror images and the four fingers' abduction is negated between them, so a
    right-hand pose commanded to a left hand splays the fingers the wrong way
    while closing -- into each other. That is refused rather than warned about.
    """
    import json  # noqa: PLC0415

    import h5py  # noqa: PLC0415

    poses = []
    with h5py.File(dataset, "r") as f:
        recorded = json.loads(f["data"].attrs.get("env_args", "{}")).get(
            "env_kwargs", {}).get("hand_side")
        if recorded is not None and recorded != side:
            raise ValueError(
                f"{dataset} was recorded with the {recorded} hand, but this run "
                f"is configured for the {side} hand.\n"
                "  The hands are mirror images; that pose would close the fingers "
                "into each other.\n"
                f"  Record a few demos with the {side} hand and use those."
            )
        if recorded is None:
            _log.warning(
                "%s does not record which hand produced it -- it predates that "
                "field. Verify it was the %s hand before trusting this pose.",
                dataset, side,
            )
        for name, demo in f["data"].items():
            if not bool(demo.attrs.get("success", True)):
                continue
            actions = np.asarray(demo["actions"])
            measured = np.asarray(demo["obs"]["gripper"])
            commanded = actions[:, N_JOINTS:]
            if commanded.shape != measured.shape:
                raise ValueError(
                    f"{dataset}:{name} has {commanded.shape} commanded hand joints "
                    f"against {measured.shape} measured. Wrong dataset?"
                )
            poses.append(commanded[int(np.argmax((commanded - measured).mean(axis=1)))])
    if not poses:
        raise ValueError(f"{dataset} contains no successful demonstrations.")
    return np.median(np.stack(poses), axis=0)


def _enter(prompt: str) -> bool:
    """Hold until ENTER. False when the operator answers q or the input is closed."""
    try:
        return input(prompt).strip().lower() != "q"
    except EOFError:
        return False


def run_trials(arm, gripper, cam, table, template, closing, args, rng,
               results: list[dict], stop: threading.Event) -> None:
    """The trial loop. Runs on a worker thread when --view is on.

    Split out so that OpenCV can own the main thread: its GUI calls are only
    reliable there, and the robot's moves are blocking, so one of the two has
    to give up the main thread for the view to stay live during motion.
    """
    grasp_z = float(template[2])
    waiting = bool(args.reach_only and getattr(args, "wait", False))
    for trial in range(args.trials):
        if stop.is_set():
            print(f"  stopped after {trial} trials")
            return
        arm.move_home()
        if gripper is not None:
            gripper.ramp_to(gripper.open_pose, seconds=OPEN_S)
        if waiting and not _enter("  arm at home: place the cube, then ENTER to reach "
                                  "(q then ENTER stops) "):
            print(f"  stopped after {trial} trials")
            return

        frame, _age, _seq = cam.snapshot()
        if frame is None:
            print(f"  {trial:3d}  {YELLOW}no camera frame{RESET}")
            continue
        found = locate_cube(frame, table)
        if found is None:
            print(f"  {trial:3d}  {YELLOW}cube not found{RESET}  "
                  f"(is it in view, and is the arm clear?)")
            results.append({"found": False})
            continue
        cube_xy, score = found

        # z error is one-sided UPWARD. Symmetric noise would put some
        # approaches below the height the template was taught at, which is
        # the one direction with a table in it. Error the policy has to
        # absorb is worth having; error that drives the hand into the table
        # is not, and the two are not the same thing.
        noise = np.array([rng.uniform(-args.noise_xy, args.noise_xy),
                          rng.uniform(-args.noise_xy, args.noise_xy),
                          rng.uniform(0.0, args.noise_z)])
        # Aim at the cube plus the camera offset. The offset is part of the
        # intended target and the noise is deliberate error; they are added
        # separately so the report can subtract the noise alone and still say
        # whether the calibration is good.
        aim = table.aim(cube_xy)
        height = grasp_z + args.standoff + getattr(args, "approach_raise_m", 0.0)
        target = _floored(pre_grasp_pose(aim, template, height, noise), grasp_z)
        state = (arm.approach(target, vertical=True)
                 if getattr(args, "approach_from_above", False) else arm.move_to_pose(target))
        offset = state.actual_tcp_pose[:2] - aim

        row = {"found": True, "cube": cube_xy, "aim": aim, "score": score,
               "noise": noise, "offset": offset}

        if args.reach_only:
            print(f"  {trial:3d}  cube ({cube_xy[0]:+.3f}, {cube_xy[1]:+.3f}) "
                  f"score {score:5.2f} | noise ({noise[0] * 1000:+4.0f}, "
                  f"{noise[1] * 1000:+4.0f}) mm | TCP-cube "
                  f"({offset[0] * 1000:+5.1f}, {offset[1] * 1000:+5.1f}) mm")
            if waiting:
                _enter("       holding over the cube: ENTER to go home ")
        else:
            # Descend to the template height, close, lift, and read the
            # fingers. Blocking moves throughout: this is deliberately the
            # dumbest possible controller, because that is what it measures.
            arm.move_to_pose(
                _floored(pre_grasp_pose(aim, template, grasp_z, noise), grasp_z))
            gripper.ramp_to(closing, seconds=CLOSE_RAMP_S)
            time.sleep(CLOSE_SETTLE_S)
            lifted = state.actual_tcp_pose.copy()
            lifted[2] = grasp_z + LIFT_M
            arm.move_to_pose(_floored(lifted, grasp_z))
            time.sleep(LIFT_SETTLE_S)
            residual, per_joint = grasp_residual(closing, gripper.state())
            held = residual > HOLD_THRESHOLD
            row.update(held=held, residual=residual)
            mark = f"{GREEN}HELD{RESET}" if held else f"{RED}empty{RESET}"
            print(f"  {trial:3d}  {mark}  residual {residual:+.4f}  "
                  f"cube ({cube_xy[0]:+.3f}, {cube_xy[1]:+.3f})")
            gripper.ramp_to(gripper.open_pose, seconds=OPEN_S)

        results.append(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reach-only", action="store_true",
                        help="approach and report, never touching the hand")
    parser.add_argument("--trials", type=int, default=10, help="how many attempts")
    parser.add_argument("--wait", action="store_true",
                        help="with --reach-only: wait at home for ENTER before each reach, "
                             "so the cube can be moved, and hold over the cube until "
                             "ENTER; q then ENTER stops")
    parser.add_argument("--close-from", type=Path, default=None,
                        help="dataset to take the fixed closing pose from")
    parser.add_argument("--standoff", type=float, default=0.10,
                        help="metres above the template pose to approach at")
    parser.add_argument("--offset-x", type=float, default=None,
                        help="metres to shift every approach along base x, where "
                             "+x is AWAY from the robot. Corrects for the wrist "
                             "camera sitting off the TCP. Default: whatever is in "
                             "the calibration")
    parser.add_argument("--offset-y", type=float, default=None,
                        help="the same along base y, which is lateral")
    parser.add_argument("--save-offset", action="store_true",
                        help="write the offset into config/table.yaml so recording "
                             "and evaluation aim the same way")
    parser.add_argument("--noise-xy", type=float, default=0.0,
                        help="uniform +-metres of approach error in x and y "
                             "(start at 0 to check the calibration)")
    parser.add_argument("--noise-z", type=float, default=0.0,
                        help="metres of approach error in z, UPWARD ONLY -- the "
                             "approach is never lower than the taught height")
    parser.add_argument("--seed", type=int, default=0, help="noise seed")
    parser.add_argument("--table", type=Path, default=None,
                        help="override config/table.yaml")
    parser.add_argument("--serial", default="", help="hand serial, if more than one")
    parser.add_argument("--video", nargs="?", const="auto", default=None, metavar="DIR",
                        help="film both cameras, with the detector drawn on the "
                             "top-down one. DIR is optional; the default is a "
                             "timestamped folder under recordings/")
    parser.add_argument("--view", action="store_true",
                        help="live camera window during the run, with the cube "
                             "boxed and the wrist view alongside; q stops after "
                             "the current trial")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation")
    parser.add_argument("--object", default="cube", help="config/objects.yaml entry, with --sam")
    add_sam_args(parser)
    add_hand_args(parser)
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging(args.verbose)
    robot_cfg, _, _ = load_configs(args)

    try:
        table = TableMap.load(args.table)
        template = table.require_template()
        sam_spec = sam_object(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"{RED}{exc}{RESET}")
        return 1

    closing: np.ndarray | None = None
    if not args.reach_only:
        if args.close_from is None:
            print(f"{RED}A scripted grasp needs a closing pose.{RESET}")
            print("  Pass --close-from <dataset.hdf5>, or use --reach-only to test")
            print("  the approach alone. Three teleoperated demos are enough to")
            print("  supply one -- guessing a 20-joint pose risks the hand.")
            return 1
        try:
            closing = closing_pose_from(args.close_from, hand_side(args))
        except (OSError, KeyError, ValueError) as exc:
            print(f"{RED}Could not read a closing pose from {args.close_from}: {exc}{RESET}")
            return 1

    # With --view the whole rig is opened so the wrist can be shown alongside;
    # the wrist view is how you tell whether the cube is even in frame at the
    # standoff you chose, which is the thing that decides how far out a policy
    # can be handed control.
    if args.view and not args.reach_only:
        # --view runs the trials on a worker so OpenCV can keep the main
        # thread, but the Wuji SDK refuses calls from any thread but the one
        # that constructed it. Reach-only never touches the hand, so it is
        # fine; a full scripted grasp is not.
        print(f"{RED}--view works with --reach-only only.{RESET} A scripted grasp "
              f"commands the hand,")
        print("  and the hand's SDK refuses calls from the viewer's worker thread.")
        return 1

    rig, names = build_camera_rig("" if (args.view or args.video) else table.camera)
    if rig is None or table.camera not in names:
        print(f"{RED}No camera named {table.camera!r} in config/cameras.yaml{RESET}")
        return 1
    detect = names.index(table.camera)

    if args.offset_x is not None:
        table.approach_offset[0] = args.offset_x
    if args.offset_y is not None:
        table.approach_offset[1] = args.offset_y
    if args.save_offset:
        _save_offset(args.table, table.approach_offset)

    grasp_z = float(template[2])
    print(f"\n{BOLD}Scripted {'reach' if args.reach_only else 'grasp'}{RESET}  "
          f"{args.trials} trials, camera={table.camera}")
    print(f"  calibration RMS   {table.rms_error_m * 1000:.1f} mm")
    print(f"  approach at       z={grasp_z + args.standoff:.3f} "
          f"({args.standoff * 1000:.0f} mm above the template)")
    print(f"  injected noise    +-{args.noise_xy * 1000:.0f} mm xy, "
          f"+{args.noise_z * 1000:.0f} mm z (upward only)")
    print(f"  hard floor        z >= {grasp_z:.3f}, the taught grasp height")
    print(f"  approach offset   ({table.approach_offset[0] * 1000:+.0f}, "
          f"{table.approach_offset[1] * 1000:+.0f}) mm  "
          f"(+x away from the robot, y lateral)")
    if closing is not None:
        print(f"  closing pose      from {args.close_from} "
              f"(mean {closing.mean():.3f} normalised)")
    print("\n  The arm homes before each trial so the camera can see the cube.")
    if not args.yes and not confirm("The arm will move. Proceed?"):
        print("Aborted.")
        return 1

    rng = np.random.default_rng(args.seed)
    arm = UR5eArm(robot_cfg)
    gripper = None
    results: list[dict] = []
    stop = threading.Event()

    video = sam = None
    try:
        rig.start(lock=True)
        if args.video is not None:
            video = start_video(args.video, rig, table, "scripted_grasp")
        sam = start_sam(sam_spec, arm, args.sam_pick)
        info = arm.connect()
        print(f"Connected. PolyScope {info['polyscope']}, {info['safety_status']}.")
        if closing is not None:
            gripper = build_hand(args, serial_number=args.serial)
            gripper.connect()
            print(f"Hand connected. {gripper}")

        run = (arm, gripper, rig.cameras[detect], table, template, closing,
               args, rng, results, stop)
        if not args.view:
            run_trials(*run)
        else:
            # The robot goes to a worker and the display keeps the main thread,
            # so the window stays live through blocking moves AND through
            # homing. Exceptions are carried back rather than lost in the
            # thread, or a safety abort would surface as a silent early exit.
            raised: list[BaseException] = []

            def _work() -> None:
                try:
                    run_trials(*run)
                except BaseException as exc:  # noqa: BLE001
                    raised.append(exc)

            worker = threading.Thread(target=_work, daemon=True)
            worker.start()
            try:
                view_loop(rig, table, worker, stop, title="scripted grasp")
            finally:
                stop.set()
                worker.join(timeout=60.0)
            if raised:
                raise raised[0]

        return _report(results, args)

    except SafetyAbort as exc:
        print(f"\n{RED}SAFETY ABORT:{RESET} {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return _report(results, args)
    finally:
        # Hand first: it has no e-stop, and arm shutdown can block.
        if gripper is not None:
            try:
                gripper.ramp_to(gripper.open_pose, seconds=OPEN_S)
                time.sleep(0.5)
                gripper.disconnect()
            except Exception as exc:  # noqa: BLE001
                print(f"{RED}hand disconnect failed: {exc}{RESET}")
        if video is not None:
            print(stop_video(video))
        arm.close()
        stop_sam(sam)
        rig.stop()
        print("Disconnected.")


def _report(results: list[dict], args: argparse.Namespace) -> int:
    """Print the summary, and for the gate say what the number means."""
    done = [r for r in results if r.get("found")]
    if not done:
        print(f"\n{RED}No trial located the cube.{RESET} Check the camera view "
              f"and that the arm clears it at home.")
        return 1
    lost = len(results) - len(done)
    print(f"\n{BOLD}{len(done)} of {len(results)} trials located the cube"
          f"{RESET}" + (f"  ({lost} lost)" if lost else ""))

    offsets = np.stack([r["offset"] for r in done])
    noises = np.stack([r["noise"][:2] for r in done])
    # Subtract the noise we injected: what is left is the error the pipeline
    # contributed, which is the number that says whether the calibration works.
    residual_xy = offsets - noises
    print(f"  TCP minus cube, noise removed: "
          f"mean ({residual_xy[:, 0].mean() * 1000:+.1f}, "
          f"{residual_xy[:, 1].mean() * 1000:+.1f}) mm, "
          f"max |{np.abs(residual_xy).max() * 1000:.1f}| mm")
    if np.abs(residual_xy).max() > 0.02:
        print(f"  {YELLOW}Over 20 mm of unexplained offset. A consistent bias is a "
              f"TCP or template problem;{RESET}")
        print(f"  {YELLOW}scatter is the homography. Re-run the calibration with "
              f"more, wider-spread points.{RESET}")

    if args.reach_only:
        print("\n  Watch the arm, not just these numbers: the question is whether "
              "the palm\n  lands centred over the cube. Once it does, re-run with "
              "--noise-xy 0.03.")
        return 0

    graded = [r for r in done if "held" in r]
    if not graded:
        return 0
    rate = sum(r["held"] for r in graded) / len(graded)
    print(f"\n{BOLD}Scripted success: {rate:.0%}{RESET} ({sum(r['held'] for r in graded)}"
          f"/{len(graded)})")
    if rate > 0.70:
        print(f"  {RED}The task is scriptable.{RESET} A policy has nothing to add "
              f"here, so do not\n  record demonstrations against it. Raise "
              f"difficulty first: randomise the cube's\n  orientation, or print a "
              f"harder object (scripts/make_objects.py).")
    elif rate < 0.10:
        print(f"  {YELLOW}Almost nothing succeeds.{RESET} Suspect the approach or "
              f"the closing pose\n  before the task -- check --reach-only lands "
              f"centred, and that the closing\n  pose came from demos with THIS "
              f"hand.")
    else:
        print(f"  {GREEN}In the window.{RESET} There is room for a learned policy "
              f"to be worth training.\n  This is the baseline every later number "
              f"is quoted against.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

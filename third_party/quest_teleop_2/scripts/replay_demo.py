#!/usr/bin/env python
"""Replay a recorded episode on the robot.

    python scripts/replay_demo.py --dataset data/demos.hdf5 --demo 0 --dry-run
    python scripts/replay_demo.py --dataset data/demos.hdf5 --demo 0
    python scripts/replay_demo.py --dataset data/demos.hdf5 --demo 0 --images --with-live

Run this before training. If replaying the recorded joint targets reproduces
the motion, the action space is right and a policy predicting those targets can
drive the arm.

The arm is moved to the episode's first configuration with moveJ, because
servoJ rejects large single-tick jumps.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import json

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quest_teleop2.data.schema import split_action  # noqa: E402
from quest_teleop2.teleop.hand_retarget import urdf_limits  # noqa: E402
from quest_teleop2.robot.safety import SafetyAbort  # noqa: E402
from quest_teleop2.robot.ur5e_arm import UR5eArm  # noqa: E402
from scripts._common import add_common_args, add_hand_args, build_hand, confirm, hand_side, hand_urdf, load_configs, setup_logging  # noqa: E402



#: Target width of the image row. Panels are scaled to hit this, then the window
#: is opened at that size -- WINDOW_NORMAL alone spawns whatever OpenCV feels
#: like, which is much smaller than the image on most setups.
WINDOW_W = 1900
#: Ceiling on magnification. Past ~2x a 640x480 frame is being invented, not shown.
MAX_SCALE = 2.5


def _open_window(name: str, n_panels: int, scale: float) -> str:
    """Create a resizable window already sized to the panels it will hold."""
    import cv2  # noqa: PLC0415

    from quest_teleop2.vision.display import GAP  # noqa: PLC0415

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    width = int(n_panels * 640 * scale) + GAP * max(0, n_panels - 1)
    cv2.resizeWindow(name, width, int(480 * scale))
    return name


def load_demo(path: Path, index: int) -> tuple[np.ndarray, dict]:
    with h5py.File(path, "r") as f:
        name = f"demo_{index}"
        if name not in f["data"]:
            available = sorted(int(k.split("_")[1]) for k in f["data"])
            raise SystemExit(f"{name} not in {path}. Available: {available}")
        demo = f["data"][name]
        actions = np.asarray(demo["actions"], dtype=np.float64)
        meta = {k: demo.attrs[k] for k in demo.attrs}
        meta["joint_pos"] = np.asarray(demo["obs"]["joint_pos"], dtype=np.float64)
        env = json.loads(f["data"].attrs.get("env_args", "{}")).get("env_kwargs", {})
        meta["action_hz"] = env.get("action_hz")
        meta["camera_names"] = list(env.get("camera_names") or [])
        if "timestamp" in demo["extra"]:
            meta["timestamp"] = np.asarray(demo["extra"]["timestamp"],
                                           dtype=np.float64).ravel()
        if "image_age" in demo["extra"]:
            meta["image_age"] = np.asarray(demo["extra"]["image_age"],
                                           dtype=np.float64).ravel()
        # Read the JPEG blobs out now: the file is closed before the arm moves,
        # and holding an open HDF5 handle across a robot motion is a good way to
        # leave a half-flushed file behind if the motion aborts.
        images: dict[str, list[bytes]] = {}
        if "images" in demo["obs"]:
            for cam in demo["obs"]["images"]:
                ds = demo["obs"]["images"][cam]
                images[cam] = [np.asarray(ds[i]).tobytes() for i in range(len(ds))]
        meta["images"] = images
    return actions, meta


def decode_panels(images: dict[str, list[bytes]], i: int, scale: float,
                  live: dict[str, np.ndarray] | None = None) -> "np.ndarray | None":
    """One tiled BGR image for frame `i`: recorded views, then live ones."""
    import cv2  # noqa: PLC0415

    from quest_teleop2.vision.display import overlay, tile  # noqa: PLC0415

    panels = []
    for cam, blobs in images.items():
        if i >= len(blobs):
            panels.append(None)
            continue
        # imdecode gives BGR; `overlay` takes RGB, so convert rather than let
        # the two cameras' colours silently disagree with the live feed.
        frame = cv2.cvtColor(cv2.imdecode(np.frombuffer(blobs[i], np.uint8),
                                          cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        panels.append(overlay(frame, [(f"{cam}  recorded  frame {i}", (255, 255, 255))],
                              scale=scale))
    for cam, frame in (live or {}).items():
        panels.append(overlay(frame, [(f"{cam}  LIVE NOW", (80, 200, 255))], scale=scale))
    return tile(panels)


def _play_video(images: dict[str, list[bytes]], horizon: int, dt: float,
                args: argparse.Namespace, meta: dict, n_panels: int) -> int:
    """Play the recorded views alone, at the rate they were recorded at."""
    import cv2  # noqa: PLC0415

    win = _open_window(
        f"demo {args.demo} — {meta.get('start_name', '?')} -> {meta.get('goal_name', '?')}",
        n_panels, args.image_scale)
    print(f"\nPlaying {horizon} frames at {1 / dt:.0f} Hz, "
          f"{args.image_scale:.2f}x ({int(n_panels * 640 * args.image_scale)} px wide). "
          "SPACE pauses, ',' / '.' step, q quits.")
    i, paused = 0, False
    try:
        while 0 <= i < horizon:
            panel = decode_panels(images, i, args.image_scale)
            if panel is not None:
                cv2.imshow(win, panel)
            key = cv2.waitKey(max(1, int(dt * 1000))) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            elif key == ord(","):
                paused, i = True, max(0, i - 1)
                continue
            elif key == ord("."):
                paused, i = True, min(horizon - 1, i + 1)
                continue
            if not paused:
                i += 1
    finally:
        cv2.destroyAllWindows()
    return 0


#: Seconds to ramp the fingers to a new pose. The hand has no e-stop.
HAND_PARK_S = 2.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/demos.hdf5"),
                        help="HDF5 dataset to replay from")
    parser.add_argument("--demo", type=int, default=0,
                        help="which episode in the dataset")
    parser.add_argument("--speed", type=float, default=1.0, help="replay rate multiplier (<=1)")
    parser.add_argument("--dry-run", action="store_true", help="analyse only, do not connect")
    parser.add_argument("--no-hand", action="store_true",
                        help="replay the arm only, leaving the hand de-energised")
    add_hand_args(parser)
    parser.add_argument("--serial", default="", help="hand serial, if more than one")
    parser.add_argument("--plot", type=Path, default=None, help="write a comparison plot here")
    parser.add_argument("--images", action="store_true",
                        help="show the recorded camera views in lockstep with the replay. "
                             "With --dry-run this plays the video alone, without moving "
                             "the arm.")
    parser.add_argument("--with-live", action="store_true",
                        help="also show what the cameras see now, beside the recorded "
                             "frames -- catches a static camera that has moved.")
    parser.add_argument("--image-scale", type=float, default=None,
                        help=f"display magnification for the image panels. Default sizes "
                             f"the row to about {WINDOW_W} px wide, whether that means "
                             f"scaling two panels up or four down. Raise it for a bigger "
                             f"window on a large monitor.")
    parser.add_argument("--yes", action="store_true",
                        help="skip the confirmation prompt")
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging(args.verbose)
    robot_cfg, _, _ = load_configs(args)

    actions, meta = load_demo(args.dataset, args.demo)
    joint_targets, gripper_traj = split_action(actions)
    gripper_width = gripper_traj.shape[1]
    horizon = len(actions)
    dt = robot_cfg.control.dt

    # Pace from the RECORDED timestamps, not from control.frequency_hz. The
    # teleop loop does not always keep up with 125 Hz -- reading the hand over
    # USB once starved it to 43 Hz -- and replaying a 43 Hz trajectory at 125 Hz
    # runs the arm ~3x faster than demonstrated. That is a safety problem, not
    # just a cosmetic one, so the recording's own clock is the authority.
    recorded_dt = None
    if meta.get("action_hz"):
        # Authoritative: written by the recorder from control.record_hz.
        recorded_dt = 1.0 / float(meta["action_hz"])
    stamps = meta.get("timestamp")
    if recorded_dt is None and stamps is not None and len(stamps) > 1:
        steps = np.diff(stamps)
        # Median, not mean: episodes contain multi-second stalls (homing, a
        # tracking dropout) that would drag an average far off the true period.
        good = steps[np.isfinite(steps) & (steps > 0)]
        # An all-equal or non-finite stamp column leaves nothing to measure;
        # np.median of an empty array is nan, and 1/nan would silently become
        # an infinite replay period.
        if len(good):
            recorded_dt = float(np.median(good))

    print(f"\n\033[1mdemo_{args.demo}\033[0m from {args.dataset}")
    # Duration must use the RECORDED period, not the servo period, or a 25 Hz
    # episode is reported as a fifth of its real length.
    frame_dt = recorded_dt or dt
    print(f"  {meta.get('start_name', '?')} -> {meta.get('goal_name', '?')}, "
          f"{horizon} frames ({horizon * frame_dt:.1f}s at {1 / frame_dt:.0f} Hz)")
    print(f"  first q: {np.round(joint_targets[0], 3).tolist()}")
    print(f"  last  q: {np.round(joint_targets[-1], 3).tolist()}")
    # Mean across the end-effector's joints: with a 20-DoF hand there is no
    # single "closed" channel, so this is how flexed the hand was on average.
    print(f"  hand flexion  mean {float(gripper_traj.mean()):.2f}, "
          f"peak {float(gripper_traj.max()):.2f}  ({gripper_width} DoF)")
    if recorded_dt:
        ratio = recorded_dt / dt
        print(f"  recorded at   {1 / recorded_dt:.1f} Hz "
              f"({recorded_dt * 1000:.2f} ms/frame); replaying at that rate")
        # A decimated recording is by design, not a fault. Only warn when the
        # rate does not match what the config asks for -- that means the loop
        # actually fell behind rather than being sampled deliberately.
        expected = robot_cfg.control.record_decimation
        if abs(ratio - expected) > 0.35 * expected:
            print(f"     \033[33mrecorded {ratio:.1f} servo ticks per action, but "
                  f"control.record_hz asks for {expected} -- the loop fell "
                  f"behind during this recording\033[0m")
    else:
        print("  \033[33mno timestamps in this demo; assuming "
              f"{1 / dt:.0f} Hz\033[0m")

    # The servo program's 0.05 rad reject applies to ONE servo tick. A recorded
    # step spans record_decimation ticks and replay interpolates across them, so
    # the meaningful quantity is the per-tick increment, not the raw difference.
    step = float(np.abs(np.diff(joint_targets, axis=0)).max())
    ticks_per_action = max(1, int(round((recorded_dt or dt) / dt)))
    per_tick = step / ticks_per_action
    print(f"  largest joint step: {step:.4f} rad per action "
          f"-> {per_tick:.4f} rad per servo tick", end="")
    if per_tick > 0.05:
        print("  \033[31m<- above the servo program's 0.05 rad reject threshold\033[0m")
        print("     The recording contains a real discontinuity; replay would stall there.")
    else:
        print("  \033[32mok\033[0m")

    images: dict[str, list[bytes]] = meta.get("images", {})
    if args.images and not images:
        print(f"\033[33mNo images in this demo.\033[0m Recorded before cameras were "
              f"wired in, or with --cameras none.")
    if images:
        counts = {c: len(b) for c, b in images.items()}
        sizes = {c: np.mean([len(x) for x in b]) for c, b in images.items()}
        print(f"  cameras: " + ", ".join(
            f"{c} {n} frames, {sizes[c] / 1024:.0f} kB mean" for c, n in counts.items()))
        bad = {c: n for c, n in counts.items() if n != horizon}
        if bad:
            print(f"  \033[31mimage count != {horizon} actions: {bad}\033[0m")
            print("     This demo is misaligned; do not train on it.")
        if "image_age" in meta:
            print(f"  frame age at record time: mean {meta['image_age'].mean() * 1000:.1f} ms, "
                  f"p95 {np.percentile(meta['image_age'], 95) * 1000:.1f} ms")

    if args.speed > 1.0:
        print("Refusing --speed > 1.0: replaying faster than recorded exceeds the recorded velocities.")
        return 1

    show = args.images and bool(images)
    n_panels = len(images) * (2 if args.with_live else 1)
    if args.image_scale is None:
        # Scale UP as well as down: two 640x480 panels side by side is only
        # 1280 px, which is a small window on any modern screen.
        args.image_scale = min(MAX_SCALE, WINDOW_W / max(n_panels * 640, 1))
    if args.dry_run:
        # Video-only playback: check what a demo actually looked like before
        # deciding whether it is worth putting on the robot.
        return _play_video(images, horizon, frame_dt, args, meta, n_panels) if show else 0

    print(f"\n\033[1mThis will MOVE the robot at {robot_cfg.connection.host}.\033[0m")
    print("  It first moves to the episode's starting configuration, then replays.")
    if not args.yes and not confirm("Proceed?"):
        print("Aborted.")
        return 1

    arm = UR5eArm(robot_cfg)
    actual: list[np.ndarray] = []

    live_rig = None
    win = ""
    if show:
        import cv2  # noqa: PLC0415

        win = _open_window(
            f"demo {args.demo} — {meta.get('start_name', '?')} -> "
            f"{meta.get('goal_name', '?')}", n_panels, args.image_scale)
        if args.with_live:
            from scripts._common import build_camera_rig  # noqa: PLC0415

            try:
                live_rig, _ = build_camera_rig(",".join(images))
                live_rig.start(lock=True)
            except Exception as exc:  # noqa: BLE001
                print(f"\033[33mLive view unavailable ({exc}); showing recorded only.\033[0m")
                live_rig = None

    # Replay is the gate that proves the recorded action space is correct.
    # Replaying only the arm would validate 6 of 26 channels and quietly
    # skip every grasp, so the hand is driven unless explicitly opted out.
    gripper = None
    if not args.no_hand and gripper_width > 1 and hand_urdf(args).exists():
        gripper = build_hand(args, serial_number=args.serial)
    elif not args.no_hand and not hand_urdf(args).exists():
        print(f"\033[33mURDF not found ({hand_urdf(args)}); replaying the arm only.\033[0m")

    try:
        info = arm.connect()
        print(f"\nConnected. PolyScope {info['polyscope']}, {info['safety_status']}.")
        if gripper is not None:
            gripper.connect()
            print(f"Hand connected. {gripper}")
            # Match the episode's opening hand pose before the arm moves, so
            # the fingers are not mid-grasp while travelling to the start.
            gripper.ramp_to(gripper_traj[0], seconds=HAND_PARK_S)

        print("Moving to the episode's initial configuration...")
        state = arm.move_to_joints(joint_targets[0], speed=0.3)
        error = float(np.max(np.abs(state.actual_q - joint_targets[0])))
        print(f"  in position (max joint error {error:.4f} rad)")

        arm.start_servo()
        print("Replaying...\n")

        # One RTDE tick is `dt`; hold each recorded target for as many ticks
        # as it occupied when it was recorded, then slow further if asked.
        period = (recorded_dt or dt) / max(args.speed, 1e-6)
        repeats = max(1, int(round(period / dt)))
        print(f"  {repeats} tick(s) per frame -> {1 / (repeats * dt):.1f} Hz playback\n")
        t_replay = time.monotonic()
        for i in range(horizon):
            # Interpolate between recorded targets instead of holding each one.
            #
            # Recording samples every 5th servo tick, so consecutive actions can
            # be ~0.06 rad apart. Commanding that as a staircase asks the servo
            # program for a 0.06 rad jump and then nothing for 40 ms, which both
            # trips its 0.05 rad jump reject and produces large tracking error --
            # the arm chases a step it was never asked to make during recording,
            # where targets advanced ~0.012 rad per tick. Interpolating restores
            # the original per-tick increments.
            nxt = joint_targets[min(i + 1, horizon - 1)]
            for r in range(repeats):
                state = arm.receive()
                if state.is_faulted:
                    raise SafetyAbort(f"Robot faulted: {', '.join(state.safety_flags)}")
                alpha = (r + 1) / repeats
                arm.command_joints(joint_targets[i] + (nxt - joint_targets[i]) * alpha)
            if gripper is not None:
                # The hand's own LowPass smooths its 25 Hz targets, so it does
                # not need interpolating the way the arm does.
                gripper.command(gripper_traj[i])
            if show:
                # After the servo ticks for this frame, never between them: the
                # inner loop is what paces the arm, and decoding inside it would
                # stretch the period unevenly.
                live = None
                if live_rig is not None:
                    frames, _age = live_rig.grab()
                    live = dict(zip(live_rig.names, frames))
                panel = decode_panels(images, i, args.image_scale, live)
                if panel is not None:
                    cv2.imshow(win, panel)
                    cv2.waitKey(1)
            actual.append(state.actual_q.copy())
            if i % 25 == 0:
                err = float(np.max(np.abs(state.actual_q - joint_targets[i])))
                print(f"\r  frame {i:5d}/{horizon}  max joint error {err * 1000:6.2f} mrad",
                      end="", flush=True)

        elapsed = time.monotonic() - t_replay
        expected = horizon * period
        print(f"\n  took {elapsed:.1f}s for {expected:.1f}s of trajectory "
              f"({elapsed / max(expected, 1e-9):.2f}x)", end="")
        # Decoding and drawing steal time from a loop paced by RTDE, so replay
        # with --images runs slightly SLOW. Slow is safe; report it so a real
        # pacing bug is not mistaken for the display overhead.
        if elapsed > expected * 1.1:
            print("  \033[33m<- display overhead; the arm still tracked every "
                  "target, just later\033[0m")
        else:
            print()

        # Let the arm settle onto the final target before measuring.
        for _ in range(int(0.5 / dt)):  # 0.5 s of settling, in RTDE ticks
            state = arm.receive()
            arm.command_joints(joint_targets[-1])
        print()

    except SafetyAbort as exc:
        print(f"\n\033[31mSAFETY ABORT:\033[0m {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 1
    finally:
        # Hand first: no e-stop, and arm shutdown can block.
        if gripper is not None:
            try:
                gripper.disconnect()
            except Exception as exc:  # noqa: BLE001
                print(f"\033[31mhand disconnect failed: {exc}\033[0m")
        arm.close()
        if live_rig is not None:
            live_rig.stop()
        if show:
            import cv2  # noqa: PLC0415

            cv2.destroyAllWindows()
        print("Disconnected.")

    if actual:
        achieved = np.stack(actual)
        n = min(len(achieved), horizon)
        err = np.abs(achieved[:n] - joint_targets[:n])
        print(f"Tracking error over {n} frames:")
        print(f"  mean {err.mean() * 1000:.2f} mrad   p95 {np.percentile(err, 95) * 1000:.2f} mrad"
              f"   max {err.max() * 1000:.2f} mrad")
        print(f"  per joint (mean, mrad): {np.round(err.mean(axis=0) * 1000, 2).tolist()}")
        if err.mean() * 1000 < 20:
            print("\n\033[32mReplay reproduced the recording.\033[0m The action space is correct;")
            print("proceed to training.")
        else:
            print("\n\033[33mLarge tracking error.\033[0m Raise the pendant speed slider toward 100%,")
            print("then re-run before concluding anything about the data.")

        if args.plot:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
            t = np.arange(n) * dt
            for j, ax in enumerate(axes.ravel()):
                ax.plot(t, joint_targets[:n, j], label="recorded", lw=1.5)
                ax.plot(t, achieved[:n, j], label="replayed", lw=1.0, ls="--")
                ax.set_title(f"joint {j}", fontsize=9)
                ax.grid(alpha=0.3)
            axes[0, 0].legend(fontsize=8)
            axes[-1, 0].set_xlabel("s")
            axes[-1, 1].set_xlabel("s")
            fig.suptitle(f"demo_{args.demo} replay: {meta.get('start_name')} -> {meta.get('goal_name')}")
            fig.tight_layout()
            args.plot.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.plot, dpi=120)
            print(f"\nPlot written to {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

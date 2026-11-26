#!/usr/bin/env python
"""Teleoperate the Wuji Hand from Quest hand tracking. No arm involved.

    python scripts/troubleshooting/teleop_hand.py          # dry run, nothing moves
    python scripts/troubleshooting/teleop_hand.py --run    # drives the hand (asks first)
    python scripts/troubleshooting/teleop_hand.py --run --record data/hand_vocab/v01.npz

One hand drives the Wuji Hand; the other hand's pinch freezes it so you can
lower your arm without the fingers following. SPACE engages; motion starts
disengaged. Right drives by default; --left swaps it.

The hand has no emergency stop. Dry run is the default, arming needs --run plus
a confirmation, the hand opens on tracking loss, and every exit path
de-energises it, including Ctrl-C.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quest_teleop2.config import TeleopCfg  # noqa: E402
from quest_teleop2.keyboard import KeyboardReader  # noqa: E402
from quest_teleop2.quest.quest_input import DualHandInput  # noqa: E402
from quest_teleop2.quest.webxr_server import WebXRServer  # noqa: E402
from quest_teleop2.robot.gripper import MIN_JOINT_RAD, WujiHand2Gripper  # noqa: E402
from quest_teleop2.teleop.hand_retarget import HandRetargeter, urdf_limits  # noqa: E402
from scripts._common import add_hand_args, build_hand, drive_side, hand_side, hand_urdf, human_limits, mirror_left, retarget_config, solve_side  # noqa: E402

#: Seconds without the driving hand tracked before the robot hand opens. Hand-tracking
#: dropouts are routine, so this is a normal event, not an error.
STALE_TIMEOUT_S = 0.5

#: Seconds between recorded vocabulary frames (25 Hz, the policy rate).
RECORD_DT = 0.04

FINGERS = ["thumb ", "index ", "middle", "ring  ", "pinky "]


#: Which joints curl each finger. Taken from the gripper's floor: a joint is
#: floored at straight exactly when more-negative means bent backwards.
CURL_JOINTS = [[j for j in range(4) if np.isfinite(MIN_JOINT_RAD[f, j])]
               for f in range(5)]

#: Abduction glyphs: left of neutral, neutral, right of neutral.
_SPREAD = "<|>"


def _bar(v: float, width: int = 10) -> str:
    n = int(round(np.clip(v, 0.0, 1.0) * width))
    return "#" * n + "-" * (width - n)


def _spread(v: float) -> str:
    """One character for how far an abduction joint sits from its neutral,
    which is normalised 0.5 rather than 0."""
    return _SPREAD[0 if v < 0.42 else 2 if v > 0.58 else 1]


def run_calibration(quest, args, lower, upper, n: int) -> int:
    """Measure the operator's relaxed hand against the robot's joints.

    Reports the RAW solver output -- no knuckle trim, no thumb remap -- because
    those corrections are exactly what is being chosen here. Measuring through
    them would fold the current guess into its own replacement.

    Median, not mean: hand tracking drops and recovers, and one bad frame of a
    half-seen hand would drag a mean somewhere no pose ever was.
    """
    from quest_teleop2.teleop.hand_retarget import HandRetargeter  # noqa: PLC0415

    raw = HandRetargeter(optimizer="adaptive", limits=(lower, upper),
                         hand_side=solve_side(args), mirror=mirror_left(args),
                         human_limits=False, config=retarget_config(args))

    print("\n\033[1mCalibration\033[0m  nothing is energised; this only reads.")
    print("  Hold your hand \033[1mrelaxed and flat\033[0m, fingers together, palm down,")
    print("  as if resting on a table. Keep still.")
    try:
        input("  Press ENTER, then hold the pose... ")
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        return 1

    frames, dropped, deadline = [], 0, time.monotonic() + 30.0
    while len(frames) < n and time.monotonic() < deadline:
        snap = quest.snapshot(engaged=True)
        if snap.hand is None:
            dropped += 1
        else:
            frames.append(raw(snap.hand.joints))
            if len(frames) % 25 == 0:
                print(f"\r  {len(frames)}/{n} frames", end="", flush=True)
        time.sleep(0.01)
    print()
    if len(frames) < n // 2:
        print(f"  \033[31mOnly {len(frames)} frames in 30s ({dropped} without a hand).\033[0m")
        print("  Keep the hand in view of the headset and try again.")
        return 1

    med = np.median(np.stack(frames), axis=0)
    spread = np.percentile(np.stack(frames), 75, axis=0) - np.percentile(
        np.stack(frames), 25, axis=0)
    names = ["thumb ", "index ", "middle", "ring  ", "pinky "]
    print(f"\n  {len(frames)} frames. Median raw solver output, degrees:")
    print(f"  {'':7s}" + "".join(f"{c:>12s}" for c in ("J0 flex", "J1 abd", "J2 pip", "J3 dip")))
    for f in range(5):
        print(f"  {names[f]}" + "".join(f"{np.degrees(med[f, j]):+12.1f}" for j in range(4)))
    if float(np.degrees(spread.max())) > 8.0:
        print(f"\n  \033[33mThe pose moved by up to {np.degrees(spread.max()):.0f} deg "
              f"while sampling.\033[0m Hold stiller\n  and re-run, or these numbers "
              "describe an average of several poses.")

    print("\n\033[1m  Paste into config/hand_calibration.yaml:\033[0m\n")
    trim = med[:, 0].copy()
    trim[0] = 0.0          # the thumb's J0 is a saddle axis, not a knuckle
    print(f"  knuckle_trim: [{', '.join(f'{v:.3f}' for v in trim)}]")
    print("  thumb_j1:")
    print("    rest:")
    print(f"      input: {med[0, 1]:.4f}")
    print(f"      output: {med[0, 1]:.4f}      # start here, then lower to lay the "
          "thumb flatter")
    print("\n  knuckle_trim zeroes each finger's MCP for THIS pose, so a relaxed hand")
    print("  gives flat robot fingers. The thumb's `output` is yours to choose: equal")
    print("  to `input` means no correction, lower lays it into the palm.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true",
                        help="actually drive the hand (default is a dry run)")
    parser.add_argument("--optimizer", default="vector", choices=("vector", "adaptive"),
                        help="vector is ~630 Hz, adaptive ~240 Hz; both are fast enough")
    parser.add_argument("--rate", type=float, default=100.0, help="control Hz")
    parser.add_argument("--cutoff", type=float, default=WujiHand2Gripper.DEFAULT_CUTOFF_HZ,
                        help="realtime controller low-pass cutoff, Hz")
    add_hand_args(parser)
    parser.add_argument("--serial", default="", help="hand serial, if more than one")
    parser.add_argument("--calibrate", action="store_true",
                        help="measure where YOUR relaxed hand lands on the robot's joints "
                             "and print values for config/hand_calibration.yaml. Never "
                             "energises the hand")
    parser.add_argument("--samples", type=int, default=150,
                        help="frames to median over when calibrating (default 150)")
    parser.add_argument("--record", type=Path, default=None,
                        help="save the retargeted 20-D commands to this .npz while ACTIVE, "
                             "for the synergy basis; works in a dry run")
    args = parser.parse_args()
    if args.record is not None and (args.record.suffix != ".npz" or args.record.exists()):
        print(f"\033[31m--record needs a new .npz path; {args.record} is not one.\033[0m")
        return 1
    if args.record is not None:
        args.record.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not hand_urdf(args).exists():
        print(f"\033[31mURDF not found: {hand_urdf(args)}\033[0m")
        return 1
    lower, upper = urdf_limits(hand_urdf(args))

    try:
        retarget = HandRetargeter(optimizer="adaptive", limits=(lower, upper),
                                   hand_side=solve_side(args),
                                          mirror=mirror_left(args),
                                          human_limits=human_limits(args),
                                          config=retarget_config(args))
    except Exception as exc:
        print(f"\033[31mRetargeter failed to load: {exc}\033[0m")
        return 1

    cfg = TeleopCfg.load()
    quest = DualHandInput(gripper_cfg=cfg.gripper, pinch_freezes=True,
                          drive_side=hand_side(args))
    server = WebXRServer(cfg, quest)

    # Built whether or not it is driven: it owns the radians->0..1 mapping,
    # and constructing it touches no hardware.
    conv = build_hand(args, limits=(lower, upper),
                      serial_number=args.serial, cutoff_hz=args.cutoff)

    gripper = None
    if args.run:
        print("\n\033[1mThis will MOVE the hand.\033[0m  It has no emergency stop -- "
              "keep it clear of\n  obstacles and be ready to unplug it.")
        try:
            if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                print("Aborted.")
                return 1
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1
        gripper = conv
    else:
        print("\n\033[33mDRY RUN\033[0m -- retargeting only, the hand is never "
              "commanded. Add --run to drive it.")

    server.start()
    if args.calibrate:
        print(f"\nOpen \033[1mhttp://localhost:{cfg.server.port}\033[0m in the headset "
              "(run setup/quest_usb.sh first).")
        try:
            return run_calibration(quest, args, lower, upper, args.samples)
        finally:
            server.stop()
    print(f"\nOpen \033[1mhttp://localhost:{cfg.server.port}\033[0m in the headset "
          "(run setup/quest_usb.sh first).")
    gate = "left" if drive_side(args) == "right" else "right"
    print(f"SPACE engages/disengages, {gate} pinch freezes, q quits. "
          "Starting DISENGAGED.")
    print("  Bars are CURL only, so a flat hand reads empty and a fist reads full.")
    print("  The trailing <, | or > is that finger's spread: | is neutral.\n")

    engaged = False
    dt = 1.0 / args.rate
    frames = stale = 0
    last_print = 0.0
    rec_t, rec_cmd, rec_joints, last_rec = [], [], [], 0.0
    if args.record is not None:
        print(f"  Recording to {args.record} at {1 / RECORD_DT:.0f} Hz while ACTIVE.\n")

    try:
        if gripper is not None:
            gripper.connect()
        with KeyboardReader() as keys:
            while True:
                loop_start = time.monotonic()

                quit_requested = False
                for key in keys.poll():
                    if key == "q":
                        quit_requested = True
                    elif key == " ":
                        engaged = not engaged
                        print(f"\n  {'ENGAGED' if engaged else 'DISENGAGED'}")
                if quit_requested:
                    break

                snap = quest.snapshot(engaged=engaged)
                hand = snap.hand
                # snap.clutch already folds in `engaged`, the left-pinch freeze
                # and left-hand staleness -- it is the single "may move" signal.
                may_move = snap.clutch

                if hand is None:
                    stale += 1
                    # Open on tracking loss rather than holding: a half-closed
                    # hand left gripping something is worse than an open one.
                    if gripper is not None and stale * dt > STALE_TIMEOUT_S:
                        gripper.command(gripper.open_pose)
                    target = None
                else:
                    stale = 0
                    frames += 1
                    radians = retarget(hand.joints)
                    # The gripper's own conversion, so the preview and the
                    # driven hand cannot disagree about what a pose means.
                    target = conv.to_normalised(radians).reshape(5, 4)
                    if gripper is not None and may_move:
                        gripper.command(target.reshape(-1))
                    if args.record is not None and may_move and loop_start - last_rec >= RECORD_DT:
                        last_rec = loop_start
                        rec_t.append(loop_start)
                        rec_cmd.append(target.reshape(-1).astype(np.float32))
                        rec_joints.append(np.asarray(hand.joints, dtype=np.float32))

                now = time.monotonic()
                if now - last_print > 0.1:
                    last_print = now
                    if target is None:
                        line = "  \033[33mno hand\033[0m" + " " * 60
                    else:
                        state = ("ACTIVE " if may_move else
                                 "frozen " if engaged else "idle   ")
                        # Curl only, so a flat hand reads as an empty bar.
                        bars = " ".join(
                            f"{FINGERS[f]}{_bar(float(target[f][CURL_JOINTS[f]].mean()))}"
                            f"{_spread(float(target[f][1]))}"
                            for f in range(5))
                        line = f"  {state} {bars}"
                        if args.record is not None:
                            line += f"  rec {len(rec_t) * RECORD_DT:5.1f} s"
                    print(f"\r{line}", end="", flush=True)

                sleep = dt - (time.monotonic() - loop_start)
                if sleep > 0:
                    time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        # Order matters: de-energise the hand before tearing down the server, so
        # a slow socket shutdown cannot leave the hand powered.
        if gripper is not None:
            gripper.disconnect()
        server.stop()
        print(f"\n{frames} tracked frames, {retarget.clamped_frames} clamped to limits.")
        if rec_t:
            np.savez_compressed(args.record, hand_cmd=np.stack(rec_cmd),
                                hand_joints=np.stack(rec_joints), t=np.asarray(rec_t),
                                side=hand_side(args))
            print(f"Saved {len(rec_t)} frames ({len(rec_t) * RECORD_DT:.0f} s) to {args.record}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

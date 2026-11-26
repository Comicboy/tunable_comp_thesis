#!/usr/bin/env python
"""Return the arm to the configured home pose, and optionally park the hand.

    python scripts/go_home.py
    python scripts/go_home.py --yes --speed 0.15     # slower, no prompt
    python scripts/go_home.py --hand                 # also park the fingers

Uses a blocking moveJ, so it is safe from any starting configuration -- unlike
servoJ, which rejects large jumps.

Without --hand the fingers are never energised. With it, they ramp to a
slightly-curled neutral pose and are then de-energised, because the hand has no
e-stop. The arm moves first, so anything gripped is released at the home pose.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quest_teleop2.robot.safety import SafetyAbort  # noqa: E402
from quest_teleop2.robot.ur5e_arm import UR5eArm  # noqa: E402
from scripts._common import add_common_args, add_hand_args, build_hand, confirm, connect_hand, hand_side, hand_urdf, load_configs, setup_logging  # noqa: E402


#: Seconds to ramp the fingers from where they are to the neutral pose. The
#: controller's own 5 Hz low-pass would smooth a single step anyway, but a step
#: from a clenched hand is still a fast move on a mechanism with no e-stop.
HAND_RAMP_S = 2.0
HAND_RAMP_HZ = 50.0


def park_hand(args: argparse.Namespace) -> int:
    """Ramp the fingers to the neutral pose, then de-energise. Blocking.

    Runs AFTER the arm's blocking moveJ, so servoing is already stopped and the
    RTDE watchdog is not in play -- this is one of the few places a caller may
    take seconds. The neutral pose is slightly curled rather than flat: a flat
    hand maximises the silhouette, which is the wrong shape to leave it in near
    a table edge.
    """
    urdf = hand_urdf(args)
    if not urdf.exists():
        print(f"\n\033[31mURDF not found: {urdf}\033[0m")
        return 1

    hand = build_hand(args, serial_number=args.serial, dry_run=args.dry_run)
    if not connect_hand(hand):
        if args.dry_run:
            print("  \033[33mNothing to inspect -- power the hand on and retry.\033[0m")
        else:
            # The arm is already home, which was the point of the run. Say so,
            # so this reads as "one of two things did not happen" rather than
            # a crash.
            print("  \033[33mThe arm is homed; the fingers were not parked.\033[0m")
        return 1
    try:
        print(f"\nHand connected. {hand}")
        start = hand.state()
        target = hand.neutral_pose
        print(f"  largest finger move: {float(np.abs(target - start).max()):.3f} "
              f"normalised, ramped over {HAND_RAMP_S:.0f}s")

        hand.ramp_to(target, seconds=HAND_RAMP_S, hz=HAND_RAMP_HZ)

        if args.dry_run:
            report_dry_run(hand, start)
            return 0
        residual = float(np.abs(hand.state() - target).max())
        print(f"  parked (residual {residual:.3f} normalised).")
        return 0
    finally:
        hand.disconnect()
        print("  fingers de-energised.")


def report_dry_run(hand, start: np.ndarray) -> None:
    """Print the frames the parking ramp would have published, and audit them.

    The interesting frame is the FIRST one. A hand that is commanded to where
    it already is cannot lurch; one that is commanded somewhere else does, and
    no amount of ramping afterwards undoes a bad opening step, because the ramp
    moves the target while the filter state is what the hand chases.
    """
    frames = np.asarray(hand.commanded)
    if frames.size == 0:
        print("\n  \033[31mNo frames were produced.\033[0m")
        return
    measured = hand.to_radians(start)
    first, last = frames[0].reshape(5, 4), frames[-1].reshape(5, 4)
    step = np.abs(first - measured)
    floor = hand._lower                               # noqa: SLF001 -- audit

    print(f"\n\033[1m  Dry run: {len(frames)} frames, none sent, motors never "
          f"energised.\033[0m")
    names = ["thumb", "index", "middle", "ring", "pinky"]
    print("\n  measured now            first frame             last frame")
    for i in range(5):
        row = lambda a: " ".join(f"{a[i, j]:+6.3f}" for j in range(4))  # noqa: E731
        print(f"  {names[i]:7s}{row(measured)}   {row(first)}   {row(last)}")

    print(f"\n  opening step from where the hand is : {np.degrees(step.max()):.1f} deg "
          f"(max), {np.degrees(step.mean()):.1f} deg (mean)")
    worst = float(np.min(frames.reshape(-1, 5, 4) - floor))
    print(f"  closest any frame comes to a floor  : {np.degrees(worst):+.1f} deg "
          f"{'(behind straight!)' if worst < -1e-9 else '(never behind straight)'}")
    print(f"  frames held at a floor              : {hand.floored_frames}")

    if np.degrees(step.max()) > 5.0:
        print(f"\n  \033[31mThe first frame is {np.degrees(step.max()):.0f} deg from the "
              f"hand's actual pose.\033[0m That is a lurch,\n  not a ramp -- do not run "
              "this for real.")
    else:
        print("\n  \033[32mThe first frame matches where the hand already is, so there "
              "is\n  nothing to lurch from.\033[0m The ramp then moves it from there.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speed", type=float, default=None,
                        help="rad/s; defaults to robot.yaml home_speed")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    parser.add_argument("--hand", action="store_true",
                        help="also park the fingers at the neutral pose (slightly curled, "
                             "abduction square), then de-energise. Off by default: the hand "
                             "has no e-stop")
    parser.add_argument("--dry-run", action="store_true",
                        help="move NOTHING. Reads the hand, computes every frame the "
                             "parking ramp would publish and prints them, leaving the "
                             "motors de-energised. Implies --hand")
    add_hand_args(parser)
    parser.add_argument("--serial", default="", help="hand serial, if more than one")
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging(args.verbose)
    robot_cfg, _, _ = load_configs(args)
    if args.dry_run:
        # The point of the run is to inspect what WOULD be commanded, so the
        # arm stays put too -- otherwise "dry run" would still move a robot.
        args.hand = True

    home = np.asarray(robot_cfg.home_q, dtype=np.float64)
    speed = args.speed if args.speed is not None else robot_cfg.home_speed

    if args.dry_run:
        # Before the arm is touched at all. A dry run commands nothing, so
        # requiring a powered, braked-off, remote-enabled robot just to read
        # the hand would make the safest run the hardest one to perform.
        print("\n\033[1mDRY RUN.\033[0m Nothing moves. The arm is not contacted and "
              "the hand's\nmotors are never energised.")
        return park_hand(args)

    arm = UR5eArm(robot_cfg)
    try:
        info = arm.connect()
        print(f"\nConnected. PolyScope {info['polyscope']}, {info['safety_status']}.")

        state = arm.receive()
        delta = np.abs(state.actual_q - home)
        print(f"  current  {np.round(state.actual_q, 3).tolist()}")
        print(f"  home     {np.round(home, 3).tolist()}")
        print(f"  largest joint move: {float(delta.max()):.3f} rad "
              f"({np.degrees(float(delta.max())):.0f} deg) at {speed} rad/s")

        arm_at_home = float(delta.max()) < 1e-3
        if arm_at_home and not args.hand:
            print("\nAlready at home.")
            return 0


        # One prompt covering everything about to move, so the operator is not
        # asked twice and cannot approve the arm while missing the fingers.
        if arm_at_home:
            print("\n\033[1mThe arm is already at home.\033[0m")
        else:
            print("\n\033[1mThis will MOVE the arm.\033[0m Check the path is clear -- "
                  "moveJ goes\n  joint-space, so the tool does not travel in a straight "
                  "line.")
        if args.hand:
            print("  \033[1mThe fingers will also move\033[0m to the neutral pose, then "
                  "de-energise.\n  The hand has no e-stop and the red button does not cut "
                  "its power. If it is\n  holding something, that is released at the home "
                  "pose.")
        if not args.yes and not confirm("Proceed?"):
            print("Aborted.")
            return 1

        if not arm_at_home:
            state = arm.move_to_joints(home, speed=speed)
            error = float(np.max(np.abs(state.actual_q - home)))
            print(f"\nAt home (max joint error {error:.4f} rad).")
        arm.zero_force()

        # Arm first, deliberately: anything the hand is gripping is then let go
        # at a known configuration rather than wherever the arm was stopped.
        if args.hand:
            return park_hand(args)
        return 0

    except SafetyAbort as exc:
        print(f"\n\033[31mSAFETY ABORT:\033[0m {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 1
    finally:
        arm.close()
        print("Disconnected.")


if __name__ == "__main__":
    raise SystemExit(main())

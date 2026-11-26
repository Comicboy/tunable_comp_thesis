#!/usr/bin/env python
"""Bisect a PROTECTIVE_STOP: is it the motion, or the servo layer at all?

    python scripts/troubleshooting/servo_hold_test.py                  # hold still for 10 s
    python scripts/troubleshooting/servo_hold_test.py --nudge 0.05     # then creep one joint
    python scripts/troubleshooting/servo_hold_test.py --no-hand        # rule the hand out

Removes one variable at a time, starting from the safest thing possible:
commanding the arm to stay exactly where it is.

  hold faults              -> not about commanded motion. Check the payload/TCP
                              on the pendant, or the servo layer.
  hold clean, nudge faults -> motion-related, and --nudge says at what speed.
                              Compare against safety.max_joint_speed.
  both clean               -> the rollout path differs; suspect the policy's own
                              targets.

The nudge is one joint, one direction, ramped from rest at
safety.max_joint_accel and capped at the speed you pass.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quest_teleop2.robot.safety import SafetyAbort  # noqa: E402
from quest_teleop2.robot.ur5e_arm import UR5eArm  # noqa: E402
from quest_teleop2.teleop.hand_retarget import urdf_limits  # noqa: E402
from scripts._common import add_common_args, add_hand_args, build_hand, confirm, hand_side, hand_urdf, load_configs, setup_logging  # noqa: E402



def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=10.0, help="how long to hold")
    parser.add_argument("--nudge", type=float, default=0.0,
                        help="after holding, creep one joint this many rad/s (try 0.05)")
    parser.add_argument("--joint", type=int, default=4,
                        help="which joint to nudge. 4 is the wrist, where a policy's "
                             "opening target usually deviates most")
    parser.add_argument("--no-hand", action="store_true",
                        help="leave the hand de-energised, to rule it out entirely")
    parser.add_argument("--no-home", action="store_true", help="skip the move to home")
    add_hand_args(parser)
    parser.add_argument("--yes", action="store_true",
                        help="skip the confirmation prompt")
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging(args.verbose)
    robot_cfg, _, _ = load_configs(args)
    dt = robot_cfg.control.dt

    print("\n\033[1mServo hold test\033[0m")
    print(f"  phase 1: command the CURRENT pose for {args.seconds:.0f}s -- zero motion")
    if args.nudge:
        print(f"  phase 2: creep joint {args.joint} at {args.nudge} rad/s, ramped from rest")
    print(f"  hand: {'de-energised' if args.no_hand else 'connected and held still'}")
    print("\n  The arm holds position. Keep the e-stop within reach anyway.")
    if not args.yes and not confirm("Proceed?"):
        return 1

    arm = UR5eArm(robot_cfg)
    gripper = None
    if not args.no_hand and hand_urdf(args).exists():
        gripper = build_hand(args)

    try:
        info = arm.connect()
        print(f"\nConnected. PolyScope {info['polyscope']}, {info['safety_status']}.")
        if gripper is not None:
            gripper.connect()
            # Hold whatever pose it is already in: commanding a new one would
            # add a variable this test exists to remove.
            gripper.command(np.asarray(gripper.state()))
            print(f"Hand connected and held. {gripper}")
        if not args.no_home:
            arm.move_home()
        arm.start_servo()

        state = arm.receive()
        hold_q = state.actual_q.copy()
        print(f"\nHolding {np.round(hold_q, 4).tolist()}")

        t0 = time.monotonic()
        ticks = int(args.seconds / dt)
        for tick in range(ticks):
            state = arm.receive()
            if state.is_faulted:
                elapsed = time.monotonic() - t0
                print(f"\n\n\033[31mFAULTED while holding still\033[0m after "
                      f"{elapsed:.2f}s ({tick} ticks): {', '.join(state.safety_flags)}")
                print("\n  The commanded target never changed, so this is NOT about motion.")
                print("  Check on the pendant, in order:")
                print("    1. Installation -> General -> Payload. A remounted tool, or a")
                print("       wrist camera and bracket added since it was last set, makes")
                print("       the controller's force estimate wrong and it reads the error")
                print("       as an external collision.")
                print("    2. Installation -> General -> TCP, for the same reason.")
                print("    3. The pendant's own protective-stop message, which names the")
                print("       joint and the cause. That is the ground truth this script")
                print("       cannot read over RTDE.")
                return 1
            arm.command_joints(hold_q)
            if tick % 125 == 0:
                err = float(np.max(np.abs(state.actual_q - hold_q)))
                print(f"\r  t={tick * dt:5.1f}s  drift {err * 1000:6.2f} mrad", end="", flush=True)
        print(f"\r  t={args.seconds:5.1f}s  \033[32mheld cleanly, no protective stop\033[0m"
              f"{' ' * 20}")

        if args.nudge:
            print(f"\n\033[1mPhase 2\033[0m: creeping joint {args.joint} at "
                  f"{args.nudge} rad/s (ramped at {robot_cfg.safety.max_joint_accel} rad/s^2)")
            target = hold_q.copy()
            speed = 0.0
            accel_step = robot_cfg.safety.max_joint_accel * dt
            for tick in range(int(5.0 / dt)):
                state = arm.receive()
                if state.is_faulted:
                    moved = abs(target[args.joint] - hold_q[args.joint])
                    print(f"\n\n\033[31mFAULTED while creeping\033[0m after {moved:.4f} rad "
                          f"at {speed:.3f} rad/s: {', '.join(state.safety_flags)}")
                    print(f"\n  Motion-related. This joint tolerates less than "
                          f"{speed:.3f} rad/s, against safety.max_joint_speed="
                          f"{robot_cfg.safety.max_joint_speed}.")
                    print("  Lower max_joint_speed below that, and check the payload anyway.")
                    return 1
                speed = min(args.nudge, speed + accel_step)
                target[args.joint] += speed * dt
                arm.command_joints(target)
                if tick % 125 == 0:
                    print(f"\r  moved {abs(target[args.joint] - hold_q[args.joint]):.4f} rad "
                          f"at {speed:.3f} rad/s", end="", flush=True)
            print(f"\r  \033[32mcrept {abs(target[args.joint] - hold_q[args.joint]):.4f} rad "
                  f"cleanly at {speed:.3f} rad/s\033[0m{' ' * 20}")

        print("\n\033[32mNo protective stop.\033[0m The servo layer and payload are fine at "
              "these speeds;\nthe rollout differs from this test only in the targets the "
              "policy asks for.")
        return 0

    except SafetyAbort as exc:
        print(f"\n\033[31mSAFETY ABORT:\033[0m {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    finally:
        if gripper is not None:
            try:
                gripper.disconnect()
            except Exception as exc:  # noqa: BLE001
                print(f"\033[31mhand disconnect failed: {exc}\033[0m")
        arm.close()
        print("Disconnected.")


if __name__ == "__main__":
    raise SystemExit(main())

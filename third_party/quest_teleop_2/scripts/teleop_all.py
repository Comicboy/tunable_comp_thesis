#!/usr/bin/env python
"""Live teleoperation of the UR5e and the Wuji Hand. Nothing is recorded.

    python scripts/teleop_all.py                  # arm + hand
    python scripts/teleop_all.py --no-hand        # arm only
    python scripts/teleop_all.py --optimizer adaptive

One hand does everything: the wrist pose drives the TCP, the five fingers
drive the robot hand. The OTHER hand's pinch freezes both, so repositioning
your own arm does not leave the fingers closing on whatever they are near.

Right hand drives by default; --left uses the left Wuji Hand and swaps the
operator's roles.

Keep the pendant speed slider at 25% for a first session.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quest_teleop2.robot.gripper import WujiHand2Gripper  # noqa: E402
from quest_teleop2.robot.safety import SafetyAbort  # noqa: E402
from quest_teleop2.teleop.hand_retarget import HandRetargeter, urdf_limits  # noqa: E402
from scripts._common import (
    # noqa: E402,
    add_common_args,
    add_hand_args,
    build_hand,
    build_session,
    confirm,
    connect_hand,
    controls_help,
    drive_side,
    hand_side,
    hand_urdf,
    human_limits,
    load_configs,
    mirror_left,
    print_quest_banner,
    retarget_config,
    setup_logging,
    solve_side,
    add_glove_args,
    build_hand_filter,
    close_hand_filter,
)



#: Seconds to ramp the fingers to the parking pose at startup.
HAND_PARK_S = 2.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-hand", action="store_true",
                        help="arm only, leaving the hand de-energised")
    parser.add_argument("--optimizer", default="vector", choices=("vector", "adaptive"),
                        help="hand retargeting solver")
    parser.add_argument("--cutoff", type=float, default=WujiHand2Gripper.DEFAULT_CUTOFF_HZ,
                        help="hand low-pass cutoff, Hz")
    add_hand_args(parser)
    add_glove_args(parser)
    parser.add_argument("--serial", default="", help="hand serial, if more than one")
    parser.add_argument("--no-home", action="store_true", help="skip the move to the home pose")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    add_common_args(parser, teleop=True, port=True)
    args = parser.parse_args()
    setup_logging(args.verbose)
    robot_cfg, teleop_cfg, _ = load_configs(args)

    gripper = None
    retarget = None
    if not args.no_hand:
        if not hand_urdf(args).exists():
            print(f"\033[31mURDF not found: {hand_urdf(args)}\033[0m")
            return 1
        limits = urdf_limits(hand_urdf(args))
        # The Hand 2 needs a config that points the optimiser at its own URDF;
        # None keeps the first generation's optimiser-keyed selection.
        config = retarget_config(args)
        if args.optimizer != "adaptive":
            print(f"\033[33mOnly an adaptive config exists for this hand; "
                  f"ignoring --optimizer {args.optimizer}.\033[0m")
        # "adaptive", not args.optimizer: the config decides the solver, and
        # passing the ignored value through only made the log claim a solver
        # that was not running.
        try:
            retargeter = HandRetargeter(optimizer="adaptive", limits=limits,
                                          hand_side=solve_side(args),
                                          mirror=mirror_left(args),
                                          human_limits=human_limits(args),
                                          config=config)
        except Exception as exc:
            print(f"\033[31mRetargeter failed to load: {exc}\033[0m")
            return 1
        gripper = build_hand(args, limits=limits, serial_number=args.serial,
                             cutoff_hz=args.cutoff)
        # The session expects a normalised command back.
        retarget = lambda joints: gripper.to_normalised(retargeter(joints))  # noqa: E731

    print(f"\n\033[1mTeleoperation of the robot at {robot_cfg.connection.host}"
          f"{'' if args.no_hand else ' + Wuji Hand'}.\033[0m")
    print(f"  workspace box  x{robot_cfg.safety.workspace.x} "
          f"y{robot_cfg.safety.workspace.y} z{robot_cfg.safety.workspace.z}")
    print(f"  speed cap      {robot_cfg.safety.max_linear_speed} m/s")
    if not args.no_hand:
        print(f"  hand           Wuji Hand 2, 20 DoF, adaptive optimiser, "
              f"{args.cutoff:.1f} Hz low-pass")
        print("\n  \033[33mThe hand has no emergency stop.\033[0m Keep it clear of the "
              "table and be\n  ready to hit the robot e-stop, which cuts arm power but "
              "NOT the hand.")
    side = drive_side(args)
    quest_gate = "left" if side == "right" else "right"
    print(f"\n{controls_help(side)}")
    if not args.yes and not confirm(
            "The arm and hand will move when the clutch is released. Proceed?"):
        print("Aborted.")
        return 1

    if args.no_hand and (args.glove or args.stiffness is not None):
        print("\033[31m--glove and --stiffness act on the hand; drop --no-hand.\033[0m")
        return 1
    try:
        hand_filter = build_hand_filter(args, robot_cfg.control.dt)
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"\033[31mGlove: {exc}\033[0m")
        return 1
    if hand_filter is not None:
        k = hand_filter.params
        print(f"  glove          {hand_filter.glove.info.get('serial')}, "
              + (f"admittance K {k.stiffness:g}, M {k.mass:g}, D {k.d:.3g}" if k else "recorded only")
              + "; learning its rest now: touch nothing, g relearns it")
    arm, server, session = build_session(args, gripper=gripper, hand_retargeter=retarget,
                                         hand_filter=hand_filter)
    try:
        server.start()
        print_quest_banner(teleop_cfg.server.port)

        info = arm.connect()
        print(f"Connected. PolyScope {info['polyscope']}, {info['safety_status']}.")
        if gripper is not None:
            # Before the arm homes, deliberately: discovering the hand is
            # missing after a 100-degree joint move helps nobody.
            if not connect_hand(gripper):
                print("  Fix the hand, or run with \033[1m--no-hand\033[0m for arm only.")
                return 1
            print(f"Hand connected. {gripper}")
            # Connecting only makes the hand hold where it was left. Park it
            # before the arm moves and before servoing: ramp_to blocks, and the
            # RTDE watchdog times out after 0.1 s.
            start = gripper.state()
            print(f"  parking the fingers: {float(np.abs(gripper.neutral_pose - start).max()):.3f} "
                  f"normalised over {HAND_PARK_S:.0f}s")
            gripper.ramp_to(gripper.neutral_pose, seconds=HAND_PARK_S)
        if not args.no_home:
            arm.move_home()
        arm.start_servo()

        print("\nPut the headset on and open the page. "
              "Press SPACE to engage; a "
              f"{quest_gate} pinch freezes.\n")
        session.run()
        return 0

    except SafetyAbort as exc:
        print(f"\n\033[31mSAFETY ABORT:\033[0m {exc}")
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        # De-energise the hand first: it has no e-stop, and arm shutdown can
        # block on the controller. Each step is independent so one failure
        # cannot skip the others.
        if gripper is not None:
            try:
                gripper.disconnect()
            except Exception as exc:  # noqa: BLE001
                print(f"\033[31mhand disconnect failed: {exc}\033[0m")
        arm.close()
        server.stop()
        close_hand_filter(hand_filter)
        print("Disconnected.")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Live teleoperation of the arm only. Nothing is recorded.

    python scripts/troubleshooting/teleop_ur5e.py

For getting comfortable with the clutch and checking the frame mapping before
adding episode management. teleop_all.py is the same thing plus the hand.

If the arm moves the wrong way, you are standing somewhere other than behind
the robot: set retarget.yaw_offset_deg in config/teleop.yaml rather than editing
the rotation matrix.

    Keep the pendant speed slider at 25% for the first session.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quest_teleop2.robot.safety import SafetyAbort  # noqa: E402
from scripts._common import (  # noqa: E402
    controls_help,
    add_common_args,
    add_hand_args,
    build_session,
    confirm,
    load_configs,
    print_quest_banner,
    setup_logging,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gripper", default="none", help="gripper backend (only 'none' exists)")
    add_hand_args(parser, urdf=False)
    parser.add_argument("--no-home", action="store_true", help="skip the move to the home pose")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    add_common_args(parser, teleop=True, port=True)
    args = parser.parse_args()
    setup_logging(args.verbose)
    robot_cfg, teleop_cfg, _ = load_configs(args)

    print(f"\n\033[1mTeleoperation of the robot at {robot_cfg.connection.host}.\033[0m")
    print(f"  workspace box  x{robot_cfg.safety.workspace.x} y{robot_cfg.safety.workspace.y} z{robot_cfg.safety.workspace.z}")
    print(f"  speed cap      {robot_cfg.safety.max_linear_speed} m/s")
    print(f"\n{controls_help(hand_side(args))}")
    if not args.yes and not confirm("The arm will move when you hold the clutch. Proceed?"):
        print("Aborted.")
        return 1

    arm, server, session = build_session(args)
    try:
        server.start()
        print_quest_banner(teleop_cfg.server.port)

        info = arm.connect()
        print(f"Connected. PolyScope {info['polyscope']}, {info['safety_status']}.")
        if not args.no_home:
            arm.move_home()
        arm.start_servo()

        print("\nPut the headset on and open the page. Hold the left grip to move the arm.\n")
        session.run()
        return 0

    except SafetyAbort as exc:
        print(f"\n\033[31mSAFETY ABORT:\033[0m {exc}")
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        arm.close()
        server.stop()
        print("Disconnected.")


if __name__ == "__main__":
    raise SystemExit(main())

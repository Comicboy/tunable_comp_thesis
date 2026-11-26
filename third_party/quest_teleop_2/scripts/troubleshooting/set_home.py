#!/usr/bin/env python
"""Capture the arm's current joint configuration as the home pose.

    python scripts/troubleshooting/set_home.py --show   # print, do not write
    python scripts/troubleshooting/set_home.py

Read-only against the robot: it reads where the arm is now and writes that into
home_q in config/robot.yaml.

Every episode starting from the same configuration is what keeps a dataset
learnable. Position the arm by hand or teleoperate it there, then run this.
Changing home_q invalidates comparisons with anything recorded before.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quest_teleop2.config import CONFIG_DIR  # noqa: E402
from quest_teleop2.robot.rtde_io import RTDEInterface  # noqa: E402
from scripts._common import add_common_args, confirm, load_configs, setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    parser.add_argument("--show", action="store_true", help="only print, do not write")
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging(args.verbose)
    robot_cfg, _, _ = load_configs(args)

    rtde = RTDEInterface(robot_cfg)
    rtde.connect()
    try:
        state = rtde.receive()
        if state is None:
            print("No RTDE data received.")
            return 1
    finally:
        rtde.disconnect()

    q = state.actual_q
    box = robot_cfg.safety.workspace.as_array()
    tcp = state.actual_tcp_pose[:3]
    inside = bool(np.all(tcp >= box[:, 0]) and np.all(tcp <= box[:, 1]))

    print("\n\033[1mCurrent arm configuration\033[0m")
    print(f"  joints  {np.round(q, 4).tolist()}")
    print(f"  tcp     {np.round(state.actual_tcp_pose, 4).tolist()}")
    print(f"  in workspace box: {'yes' if inside else 'NO'}")
    print(f"\n  existing home_q: {robot_cfg.home_q}")

    if not inside:
        print(
            "\n\033[33mThis pose is outside the configured workspace box.\033[0m Homing here is "
            "fine,\n  but the first clutch engage will pull the TCP back inside the box, so the\n"
            "  arm will move. Either widen the box in robot.yaml or pick a pose inside it."
        )

    if args.show:
        return 0
    if not args.yes and not confirm("\nWrite this as home_q?"):
        print("Unchanged.")
        return 1

    path = Path(args.robot_config or (CONFIG_DIR / "robot.yaml"))
    values = "[" + ", ".join(f"{float(v):.5f}" for v in q) + "]"
    text = path.read_text()

    # Rewrite just the home_q line rather than round-tripping through yaml,
    # which would silently strip every comment in the file. The comments here
    # document safety limits, so losing them is worse than it sounds.
    new_text, count = re.subn(
        r"^home_q:.*$", f"home_q: {values}", text, count=1, flags=re.MULTILINE
    )
    if count == 0:
        data = yaml.safe_load(text) or {}
        data["home_q"] = [round(float(v), 5) for v in q]
        new_text = yaml.safe_dump(data, sort_keys=False, default_flow_style=None)
        print("  (no home_q line found; rewrote the file and lost its comments)")
    path.write_text(new_text)

    print(f"\nWrote home_q to {path}")
    print("\nNext: python scripts/record_demos.py --out data/demos.hdf5")
    print("  (drop --no-home now; homing is safe and repeatable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

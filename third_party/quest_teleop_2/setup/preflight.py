#!/usr/bin/env python
"""Check the environment, headset and robot before anything is asked to move.

    python setup/preflight.py
    python setup/preflight.py --skip-robot

Read-only; never commands motion. Run it first, and after any change to the
robot, the network or the headset connection.
"""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._common import add_common_args, load_configs, setup_logging  # noqa: E402

OK = "\033[32m  ok  \033[0m"
WARN = "\033[33m warn \033[0m"
FAIL = "\033[31m fail \033[0m"


class Checks:
    def __init__(self) -> None:
        self.failed = 0
        self.warned = 0

    def report(self, status: str, name: str, detail: str = "") -> None:
        print(f"  [{status}] {name}" + (f"  --  {detail}" if detail else ""))
        if status is FAIL:
            self.failed += 1
        elif status is WARN:
            self.warned += 1


def check_python(c: Checks) -> None:
    import platform

    c.report(OK, "python", f"{sys.version.split()[0]} ({platform.machine()})")
    try:
        import torch

        # Must match pick_device() in quest_teleop/policy/models.py, or this
        # reports a device training will not actually use. On Linux MPS is
        # always False, so checking it alone would print "cpu" on a working GPU.
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = f"cuda ({torch.cuda.get_device_name(0)})"
        else:
            device = "cpu"
        c.report(OK, "torch", f"{torch.__version__}, device {device}")
    except Exception as exc:
        c.report(FAIL, "torch", str(exc))
    # rtde_control/rtde_receive come from ur_rtde. The old "rtde" module was
    # UR's official client, dropped with the URScript backend -- checking for
    # it would pass only on machines that still have the stale package.
    for module in ("numpy", "scipy", "h5py", "aiohttp", "yaml",
                   "rtde_control", "rtde_receive"):
        try:
            __import__(module)
            c.report(OK, module)
        except Exception as exc:
            c.report(FAIL, module, str(exc))


def check_adb(c: Checks, port: int) -> None:
    if shutil.which("adb") is None:
        c.report(FAIL, "adb", "not installed -- brew install --cask android-platform-tools")
        return
    try:
        out = subprocess.run(
            ["adb", "devices"], capture_output=True, text=True, timeout=15
        ).stdout
    except Exception as exc:
        c.report(FAIL, "adb devices", str(exc))
        return

    devices = [line.split() for line in out.splitlines()[1:] if line.strip()]
    if not devices:
        c.report(FAIL, "quest usb", "no device -- check the cable is data-capable and Dev Mode is on")
        return
    state = devices[0][1]
    if state == "unauthorized":
        c.report(FAIL, "quest usb", "unauthorized -- accept the USB debugging prompt in the headset")
        return
    if state != "device":
        c.report(FAIL, "quest usb", f"device state '{state}'")
        return
    c.report(OK, "quest usb", f"{devices[0][0]} connected")

    reverse = subprocess.run(
        ["adb", "reverse", "--list"], capture_output=True, text=True, timeout=15
    ).stdout
    if f"tcp:{port}" in reverse:
        c.report(OK, "adb reverse", f"localhost:{port} forwarded")
    else:
        c.report(WARN, "adb reverse", f"port {port} not forwarded -- run setup/quest_usb.sh {port}")


def check_network(c: Checks, host: str, ports: dict[str, int]) -> bool:
    reachable = True
    for name, port in ports.items():
        try:
            with socket.create_connection((host, port), timeout=3.0):
                c.report(OK, f"robot {name}", f"{host}:{port} open")
        except OSError as exc:
            c.report(FAIL, f"robot {name}", f"{host}:{port} -- {exc.strerror or exc}")
            reachable = False
    return reachable


def check_robot(c: Checks, robot_cfg) -> None:
    from quest_teleop2.robot.dashboard import DashboardClient
    from quest_teleop2.robot.rtde_io import RTDEInterface

    try:
        with DashboardClient(robot_cfg) as dash:
            c.report(OK, "polyscope", dash.polyscope_version())
            mode = dash.robot_mode()
            c.report(OK if mode == "RUNNING" else FAIL, "robot mode", mode)
            safety = dash.safety_status()
            c.report(OK if safety in ("NORMAL", "REDUCED") else FAIL, "safety status", safety)
            remote = dash.is_in_remote_control()
            c.report(
                OK if remote else FAIL,
                "remote control",
                "enabled"
                if remote
                else "LOCAL mode -- URScript on ports 30001-30003 is silently ignored. "
                "Enable Remote Control on the pendant.",
            )
    except Exception as exc:
        c.report(FAIL, "dashboard", str(exc))

    try:
        rtde = RTDEInterface(robot_cfg)
        rtde.connect()
        state = rtde.receive()
        rtde.disconnect()
        if state is None:
            c.report(FAIL, "rtde", "connected but received no data")
        else:
            import numpy as np

            c.report(OK, "rtde", state.describe())
            c.report(OK, "joints", str(np.round(state.actual_q, 3).tolist()))
            c.report(OK, "tcp pose", str(np.round(state.actual_tcp_pose, 3).tolist()))

            box = robot_cfg.safety.workspace.as_array()
            pos = state.actual_tcp_pose[:3]
            inside = bool(np.all(pos >= box[:, 0]) and np.all(pos <= box[:, 1]))
            c.report(
                OK if inside else WARN,
                "tcp in workspace",
                "yes"
                if inside
                else f"TCP {np.round(pos, 3).tolist()} is outside the configured box -- "
                "it will be clamped, so the arm will move on first clutch",
            )
    except Exception as exc:
        c.report(FAIL, "rtde", str(exc))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-robot", action="store_true", help="only check this machine and the headset")
    add_common_args(parser, teleop=True, task=True, port=True)
    args = parser.parse_args()
    setup_logging(args.verbose)
    robot_cfg, teleop_cfg, task_cfg = load_configs(args)

    c = Checks()
    print("\n\033[1mEnvironment\033[0m")
    check_python(c)

    print("\n\033[1mQuest over USB-C\033[0m")
    check_adb(c, teleop_cfg.server.port)

    print("\n\033[1mTask\033[0m")
    c.report(
        OK,
        "variants",
        f"{task_cfg.n_starts} starts x {task_cfg.n_goals} goals = "
        f"{task_cfg.n_starts * task_cfg.n_goals}, obs_dim {task_cfg.obs_dim}",
    )

    if not args.skip_robot:
        print("\n\033[1mRobot network\033[0m")
        conn = robot_cfg.connection
        reachable = check_network(
            c,
            conn.host,
            {"rtde": conn.rtde_port, "script": conn.script_port, "dashboard": conn.dashboard_port},
        )
        if reachable:
            print("\n\033[1mRobot state\033[0m")
            check_robot(c, robot_cfg)
        else:
            print(
                f"\n  Robot at {conn.host} is unreachable. Check the USB-C Ethernet adapter,\n"
                "  and that this machine has an address on the same subnet as the control box."
            )

    print()
    if c.failed:
        print(f"\033[31m{c.failed} check(s) failed.\033[0m Fix these before going further.")
        return 1
    if c.warned:
        print(f"\033[33m{c.warned} warning(s).\033[0m Safe to continue if you understand them.")
        return 0
    print("\033[32mAll checks passed.\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

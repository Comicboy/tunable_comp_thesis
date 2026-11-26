#!/usr/bin/env python
"""Verify the Linux-specific half of the stack. Read-only, no motion.

    python scripts/troubleshooting/check_linux_stack.py
    python scripts/troubleshooting/check_linux_stack.py --skip-hand

setup/preflight.py covers the robot and the Quest; this covers the CUDA build
and the Wuji Hand 2's network path -- route, reachability and SDK discovery.
The hand is on Ethernet, so there is no udev or USB-permission question; what
replaces it is that the hand and the robot share a /24 across two interfaces.

The CUDA check runs a real matmul rather than trusting torch.cuda.is_available():
a build without kernels for this GPU reports it as available and then fails at
the first launch.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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


def check_platform(c: Checks) -> None:
    import platform

    if sys.platform != "linux":
        c.report(WARN, "platform", f"{sys.platform} -- this script targets Linux")
    else:
        c.report(OK, "platform", f"{platform.system()} {platform.release()} ({platform.machine()})")
    c.report(OK, "python", sys.version.split()[0])


def check_torch_cuda(c: Checks) -> None:
    try:
        import torch
    except Exception as exc:
        c.report(FAIL, "torch", str(exc))
        return

    c.report(OK, "torch", f"{torch.__version__}, cuda build {torch.version.cuda}")

    if not torch.cuda.is_available():
        c.report(
            WARN,
            "cuda",
            "not available -- training falls back to CPU. Check the NVIDIA driver "
            "(need 570+ for Blackwell) with nvidia-smi.",
        )
        return

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    arches = torch.cuda.get_arch_list()
    c.report(OK, "gpu", f"{name} ({arch})")

    if arch not in arches:
        c.report(
            FAIL,
            "cuda kernels",
            f"this torch build has no {arch} kernels (has {arches}). It will import "
            "fine and fail at the first kernel launch. Reinstall:\n"
            "        pip install --index-url https://download.pytorch.org/whl/cu128 torch",
        )
        return

    # The real test: actually run something on the device.
    try:
        a = torch.randn(64, 64, device="cuda")
        result = float((a @ a).sum().item())
        if result != result:  # NaN
            raise RuntimeError("matmul produced NaN")
        torch.cuda.synchronize()
        c.report(OK, "cuda kernels", f"{arch} matmul executed on device")
    except Exception as exc:
        c.report(FAIL, "cuda kernels", f"a real matmul failed: {exc}")


def check_wuji_sdk(c: Checks) -> None:
    try:
        import wuji_sdk
    except ImportError as exc:
        c.report(FAIL, "wuji-sdk", f"{exc} -- pip install wuji-sdk")
        return
    c.report(OK, "wuji-sdk", getattr(wuji_sdk, "__version__", "installed"))


def _run(cmd: list[str], timeout: float = 10.0) -> tuple[int, str]:
    """Run a command, returning (returncode, stdout).

    The RETURNCODE is returned alongside the output on purpose. Matching on
    output text is how a diagnostic lies: `"0% packet loss" in output` is True
    for "100% packet loss", so a total failure reads as a pass. Check the exit
    status, which cannot be a substring of anything.

    Diagnostics must never raise: a missing binary is itself a finding, and the
    caller reports it as such rather than crashing the whole check run.
    """
    if shutil.which(cmd[0]) is None:
        return 127, ""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout
    except Exception:  # noqa: BLE001 -- see docstring
        return 1, ""


def _lsusb() -> str:
    if shutil.which("lsusb") is None:
        return ""
    try:
        return subprocess.run(["lsusb"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ""


def check_hand_network(c: Checks) -> None:
    """The Hand 2 is on Ethernet, not USB -- so this is link, route and scan.

    There is no udev or USB-permission question for this hand. What replaces it
    is that the hand and the robot share a /24 across two interfaces, so the
    host route for the hand has to beat the other interface's subnet route.
    """
    host = "192.168.1.111"

    # 1. Is any interface actually carrying the hand's route?
    rc, route = _run(["ip", "route", "get", host])
    if rc != 0 or not route:
        c.report(WARN, "hand route", "could not query the route")
    else:
        dev = re.search(r"\bdev\s+(\S+)", route)
        c.report(OK, "hand route", f"{host} via {dev.group(1) if dev else route.strip()}")

    # 2. Does it answer? This is the check that matters. ping exits non-zero
    #    when no reply arrives, which is the signal -- not the output text.
    rc, ping = _run(["ping", "-c", "2", "-W", "2", host])
    if rc == 0:
        rtt = re.search(r"= ([\d.]+)/([\d.]+)/", ping)
        c.report(OK, "hand reachable", f"{host}" + (f", {rtt.group(2)} ms avg" if rtt else ""))
    else:
        c.report(
            FAIL,
            "hand reachable",
            f"{host} does not answer -- check the hand is powered (white breathing "
            "LED), its Ethernet adapter is plugged in, and `nmcli con up wuji-hand`",
        )
        return

    # 3. Does the SDK enumerate it? Reachable but undiscovered means the hand
    #    is up but not serving, which is a different fault from a dead link.
    try:
        from wuji_sdk import DeviceType, SdkManager
    except ImportError:
        c.report(WARN, "hand scan", "wuji-sdk not importable, skipped")
        return
    try:
        found = [d for d in SdkManager.instance().scan()
                 if d.device_type == DeviceType.WujiHand2]
    except Exception as exc:  # noqa: BLE001
        c.report(FAIL, "hand scan", f"scan raised: {exc}")
        return
    if not found:
        c.report(FAIL, "hand scan", "reachable but not discovered -- is it serving?")
    else:
        d = found[0]
        c.report(OK, "hand scan", f"{d.sn} at {d.ip}")


def check_adb(c: Checks) -> None:
    if shutil.which("adb") is None:
        c.report(FAIL, "adb", "not installed -- sudo apt install android-sdk-platform-tools")
        return
    try:
        out = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=15).stdout
    except Exception as exc:
        c.report(FAIL, "adb", str(exc))
        return

    devices = [ln.split() for ln in out.splitlines()[1:] if ln.strip()]
    if not devices:
        c.report(WARN, "quest usb", "no device -- plug in the headset (needed for teleop only)")
        return
    state = devices[0][1]
    if state == "no":  # "no permissions"
        c.report(FAIL, "quest usb", "no permissions -- udev rules not applied, or not logged back in")
    elif state == "unauthorized":
        c.report(FAIL, "quest usb", "unauthorized -- accept the USB debugging prompt in the headset")
    elif state == "device":
        c.report(OK, "quest usb", f"{devices[0][0]} connected")
    else:
        c.report(WARN, "quest usb", f"state '{state}'")


def check_project(c: Checks) -> None:
    try:
        from quest_teleop2.config import RobotCfg, TaskCfg, TeleopCfg

        robot = RobotCfg.load()
        teleop = TeleopCfg.load()
        task = TaskCfg.load()
    except Exception as exc:
        c.report(FAIL, "configs", str(exc))
        return
    c.report(OK, "configs", f"robot {robot.connection.host}, input mode '{teleop.input.mode}'")
    c.report(
        OK,
        "task",
        f"{task.n_starts} starts x {task.n_goals} goals, obs_dim {task.obs_dim}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-hand", action="store_true",
                        help="skip the USB checks for when the hand is not attached")
    args = parser.parse_args()

    c = Checks()

    print("\n\033[1mEnvironment\033[0m")
    check_platform(c)
    check_torch_cuda(c)

    print("\n\033[1mProject\033[0m")
    check_project(c)

    print("\n\033[1mWuji Hand 2\033[0m")
    check_wuji_sdk(c)
    if args.skip_hand:
        print("  [ skip ] hand network  --  --skip-hand")
    else:
        check_hand_network(c)

    print("\n\033[1mQuest\033[0m")
    check_adb(c)

    print()
    if c.failed:
        print(f"\033[31m{c.failed} check(s) failed.\033[0m")
        return 1
    if c.warned:
        print(f"\033[33m{c.warned} warning(s).\033[0m Safe to continue if you understand them.")
        return 0
    print("\033[32mAll checks passed.\033[0m")
    print("\nNext:  python setup/preflight.py     (the robot half)")
    print("       python scripts/troubleshooting/hand_test.py    (read the hand, no motion)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

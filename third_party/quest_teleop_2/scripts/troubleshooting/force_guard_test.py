#!/usr/bin/env python
"""Check the software force guard against a scale.

    python scripts/troubleshooting/force_guard_test.py --depth 0  # hold still; push a scale on it
    python scripts/troubleshooting/force_guard_test.py            # down 2 cm and back, 8 s cycles

Start with the hand in free air; the sensor is zeroed there. On a force stop the
arm stays put: read the scale, then run go_home.py. Touch the palm, never the fingertips.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quest_teleop2.robot.safety import ForceLimitExceeded, SafetyAbort, SafetyLimiter  # noqa: E402
from quest_teleop2.robot.ur5e_arm import UR5eArm  # noqa: E402
from scripts._common import add_common_args, confirm, load_configs, setup_logging  # noqa: E402

G = 9.81


def depth_at(t: float, depth: float, period: float) -> float:
    """Metres below the start at time t: 0 -> depth -> 0 once per period."""
    return depth * 0.5 * (1.0 - math.cos(2.0 * math.pi * t / period))


def run(arm, limiter, start: np.ndarray, depth: float, period: float, seconds: float) -> None:
    """Servo the TCP along the cycle. A force stop raises out of arm.receive()."""
    t0, last = time.monotonic(), -1.0
    while (now := time.monotonic() - t0) < seconds:
        state = arm.receive()
        target = start.copy()
        target[2] -= depth_at(now, depth, period)
        verdict = limiter.filter(target, state, motion_enabled=True, data_age=0.0)
        arm.command(verdict.command, verdict.enable)
        if now - last > 0.2:
            last = now
            print(f"\r  {depth_at(now, depth, period) * 1000:4.1f} mm down   force "
                  f"{arm.tcp_force:5.1f} N ({arm.tcp_force / G:4.2f} kg)   peak "
                  f"{arm.peak_tcp_force:5.1f} N", end="", flush=True)
    print()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depth", type=float, default=0.02,
                   help="metres below the start; 0 holds still")
    p.add_argument("--period", type=float, default=8.0, help="seconds per down-and-up cycle")
    p.add_argument("--cycles", type=int, default=5)
    p.add_argument("--yes", action="store_true", help="skip the confirmation")
    add_common_args(p)
    args = p.parse_args()
    setup_logging(args.verbose)
    if not 0.0 <= args.depth <= 0.03:
        print("--depth must be between 0 and 0.03 m.")
        return 1
    if args.period < 4.0:
        print("--period must be at least 4 s.")
        return 1
    robot_cfg, _, _ = load_configs(args)
    limit = robot_cfg.safety.max_tcp_force

    print("\n\033[1mForce guard test\033[0m")
    print(f"  guard      {limit:.0f} N = {limit / G:.2f} kg on a scale")
    if args.depth == 0:
        print(f"  motion     none: the arm holds still for {args.cycles * args.period:.0f} s")
    else:
        print(f"  motion     {args.depth * 1000:.0f} mm down and back, {args.period:.0f} s per "
              f"cycle, {args.cycles} cycles")
    print("  The hand must be in free air now. Contact with the palm or a rigid part only.")
    if not args.yes and not confirm("Proceed?"):
        return 1

    arm = UR5eArm(robot_cfg)
    limiter = SafetyLimiter(robot_cfg.safety, robot_cfg.control.dt)
    try:
        info = arm.connect()
        print(f"Connected. PolyScope {info['polyscope']}, {info['safety_status']}.")
        arm.zero_force()
        start = np.asarray(arm.receive().actual_tcp_pose, dtype=np.float64).copy()
        floor = robot_cfg.safety.workspace.z[0]
        if start[2] - args.depth < floor:
            print(f"\033[31mStart z {start[2]:.3f} m minus {args.depth * 1000:.0f} mm passes the "
                  f"z floor {floor:.2f} m. Raise the arm first.\033[0m")
            return 1
        arm.start_servo()
        limiter.reset(arm.receive().actual_tcp_pose)
        arm.peak_tcp_force = 0.0
        run(arm, limiter, start, args.depth, args.period, args.cycles * args.period)
        print(f"No force stop. Peak {arm.peak_tcp_force:.1f} N "
              f"({arm.peak_tcp_force / G:.2f} kg).")
        return 0
    except ForceLimitExceeded as exc:
        print(f"\n\033[1mFORCE STOP\033[0m {exc}")
        print(f"  peak {arm.peak_tcp_force:.1f} N = {arm.peak_tcp_force / G:.2f} kg. "
              "Read the scale now, then lift the arm: python scripts/go_home.py")
        return 0
    except SafetyAbort as exc:
        print(f"\n\033[31mSAFETY ABORT:\033[0m {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    finally:
        arm.close()


if __name__ == "__main__":
    raise SystemExit(main())

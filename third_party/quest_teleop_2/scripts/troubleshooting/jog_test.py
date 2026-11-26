#!/usr/bin/env python
"""First motion: a canned sine on one axis. No headset involved.

    python scripts/troubleshooting/jog_test.py
    python scripts/troubleshooting/jog_test.py --axis y --amplitude 0.03
    python scripts/troubleshooting/jog_test.py --test-watchdog

Exercises the servo path, controller-side IK, the workspace clamp and the speed
cap, with a motion small enough to watch.

    Put the pendant speed slider at 25% before running this.

--test-watchdog stops sending commands mid-motion, which must cause the
controller to halt by itself. That is what protects you if this process is
killed while the arm is moving.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quest_teleop2.robot.safety import SafetyAbort, SafetyLimiter  # noqa: E402
from quest_teleop2.robot.ur5e_arm import UR5eArm  # noqa: E402
from scripts._common import add_common_args, confirm, load_configs, setup_logging  # noqa: E402

AXES = {"x": 0, "y": 1, "z": 2}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=sorted(AXES), default="z",
                        help="which TCP axis to move: x, y or z")
    parser.add_argument("--amplitude", type=float, default=0.02, help="metres, peak (default 2 cm)")
    parser.add_argument("--period", type=float, default=6.0, help="seconds per cycle")
    parser.add_argument("--cycles", type=float, default=3.0,
                        help="how many sine cycles to run")
    parser.add_argument("--no-home", action="store_true", help="start from the current pose")
    parser.add_argument("--test-watchdog", action="store_true", help="also verify the stop-on-silence backstop")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging(args.verbose)
    robot_cfg, _, _ = load_configs(args)

    if args.amplitude > 0.10:
        print(f"Refusing --amplitude {args.amplitude} m. This is a first-motion test; keep it under 10 cm.")
        return 1

    print(f"\n\033[1mThis will MOVE the robot at {robot_cfg.connection.host}.\033[0m")
    print(f"  axis {args.axis}, +/-{args.amplitude * 100:.0f} cm, {args.cycles:g} cycles of {args.period:g}s")
    print("  Set the pendant speed slider to 25% and keep a hand on the e-stop.")
    if not args.yes and not confirm("Proceed?"):
        print("Aborted.")
        return 1

    arm = UR5eArm(robot_cfg)
    limiter = SafetyLimiter(robot_cfg.safety, robot_cfg.control.dt)
    axis = AXES[args.axis]
    errors: list[float] = []

    try:
        info = arm.connect()
        print(f"\nConnected. PolyScope {info['polyscope']}, {info['safety_status']}.")

        if not args.no_home:
            arm.move_home()

        arm.start_servo()
        state = arm.receive()
        origin = state.actual_tcp_pose.copy()
        limiter.reset(origin)
        print(f"Origin TCP: {np.round(origin, 4).tolist()}")
        print("Moving...\n")

        dt = robot_cfg.control.dt
        duration = args.period * args.cycles
        stalled_at = duration * 0.5 if args.test_watchdog else None
        watchdog_fired = False
        t = 0.0

        while t < duration:
            state = arm.receive()
            limiter.check_robot(state)

            # Deliberate silence: stop feeding the RTDE heartbeat and confirm
            # the controller stops the program on its own.
            if stalled_at is not None and t >= stalled_at:
                print("\n  Stalling the command stream to test the watchdog...")
                deadline = time.monotonic() + 4.0
                while time.monotonic() < deadline:
                    try:
                        state = arm.receive()
                    except SafetyAbort:
                        # Stream ended rather than the program stopping. Treated
                        # as inconclusive, exactly as the old `state is None`
                        # branch was -- a dropped connection is not proof the
                        # watchdog did its job.
                        break
                    if not state.is_program_running:
                        watchdog_fired = True
                        break
                print(
                    "  \033[32mwatchdog fired: controller stopped the program\033[0m"
                    if watchdog_fired
                    else "  \033[31mwatchdog did NOT fire\033[0m -- do not rely on it; "
                    "the arm did not stop on its own -- do not rely on it"
                )
                break

            target = origin.copy()
            target[axis] = origin[axis] + args.amplitude * np.sin(2 * np.pi * t / args.period)
            verdict = limiter.filter(target, state, motion_enabled=True, data_age=0.0)
            arm.command(verdict.command, verdict.enable)

            errors.append(float(np.linalg.norm(state.actual_tcp_pose[:3] - verdict.command[:3])))
            t += dt
            if int(t / dt) % 25 == 0:
                print(
                    f"\r  t={t:5.2f}s  target {args.axis}={verdict.command[axis]:+.4f}  "
                    f"actual {args.axis}={state.actual_tcp_pose[axis]:+.4f}  "
                    f"err={errors[-1] * 1000:5.2f} mm  {verdict.reason:<30}",
                    end="",
                    flush=True,
                )

        print("\n")
        if errors:
            arr = np.asarray(errors) * 1000
            print(f"Tracking error: mean {arr.mean():.2f} mm, p95 {np.percentile(arr, 95):.2f} mm, max {arr.max():.2f} mm")
            if arr.mean() > 10:
                print(
                    "\033[33mHigh tracking error.\033[0m Expected for the first run if the pendant "
                    "speed slider is low -- raise it toward 100% and re-check before tuning servoj."
                )
        print(f"Safety: {limiter.stats.summary()}")
        if args.test_watchdog:
            print(f"Watchdog: {'PASS' if watchdog_fired else 'FAIL'}")
        return 0 if (not args.test_watchdog or watchdog_fired) else 1

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

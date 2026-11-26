#!/usr/bin/env python
"""Read-only RTDE stream. Never commands motion.

    python scripts/troubleshooting/stream_state.py
    python scripts/troubleshooting/stream_state.py --check-rate --seconds 10

Cross-check the printed TCP pose against the pendant's own readout: if they
disagree, the workspace box in robot.yaml means something different from what
you think. --check-rate reports loop timing jitter.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quest_teleop2.robot.rtde_io import RTDEInterface  # noqa: E402
from scripts._common import add_common_args, load_configs, setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds")
    parser.add_argument("--check-rate", action="store_true", help="report loop timing jitter")
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging(args.verbose)
    robot_cfg, _, _ = load_configs(args)

    box = robot_cfg.safety.workspace.as_array()
    rtde = RTDEInterface(robot_cfg)
    rtde.connect()

    print(f"\nStreaming from {robot_cfg.connection.host} at {robot_cfg.control.frequency_hz:.0f} Hz.")
    print("This script does not move the robot. Ctrl-C to stop.\n")

    intervals: list[float] = []
    started = time.monotonic()
    last = time.perf_counter()
    n = 0

    try:
        while True:
            state = rtde.receive()
            if state is None:
                print("\nStream ended.")
                break
            now = time.perf_counter()
            intervals.append(now - last)
            last = now
            n += 1

            if args.seconds and time.monotonic() - started > args.seconds:
                break
            if n % 12:  # print at ~10 Hz, not 125
                continue

            pos = state.actual_tcp_pose[:3]
            inside = bool(np.all(pos >= box[:, 0]) and np.all(pos <= box[:, 1]))
            mark = "\033[32min box\033[0m" if inside else "\033[33mOUT of box\033[0m"
            print(
                f"\rq {np.array2string(state.actual_q, precision=3, suppress_small=True):<48} "
                f"tcp {np.array2string(state.actual_tcp_pose, precision=3, suppress_small=True):<52} "
                f"{mark}  {state.describe()}".ljust(190),
                end="",
                flush=True,
            )
    except KeyboardInterrupt:
        pass
    finally:
        rtde.disconnect()
        print()

    if args.check_rate and len(intervals) > 10:
        arr = np.asarray(intervals[5:]) * 1e3  # drop startup, convert to ms
        expected = 1e3 / robot_cfg.control.frequency_hz
        print(
            f"\nTiming over {len(arr)} ticks (expected {expected:.2f} ms):\n"
            f"  mean {arr.mean():.2f} ms   median {np.median(arr):.2f} ms\n"
            f"  p95  {np.percentile(arr, 95):.2f} ms   p99 {np.percentile(arr, 99):.2f} ms\n"
            f"  max  {arr.max():.2f} ms"
        )
        if arr.max() > expected * 4:
            print(
                "\n\033[33mLarge outliers.\033[0m This is not a real-time kernel, so occasional "
                "spikes are normal and the servoj lookahead absorbs them. Sustained\n"
                "spikes mean something else is loading the machine, or the robot is on WiFi."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

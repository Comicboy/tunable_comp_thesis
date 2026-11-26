#!/usr/bin/env python
"""First contact with the Wuji Hand 2.

    python scripts/troubleshooting/hand_test.py            # read only, nothing moves
    python scripts/troubleshooting/hand_test.py --watch    # live joint readout
    python scripts/troubleshooting/hand_test.py --move     # one joint, small slow sine
    python scripts/troubleshooting/hand_test.py --fingers  # each finger in turn

Built to fail safe: reading is the default, motion needs an explicit flag and a
confirmation, amplitude is capped, every motion path returns the joint to where
it started, and only one joint moves at a time.

The hand has no emergency stop -- the robot's red button does not cut its power,
and unplugging it is the only hard stop. Ctrl-C is handled and returns to start
on the way out.

Everything goes through `WujiHand2Gripper`, so the current limit, the MIT gains
and the command low-pass are exactly the ones teleoperation uses. Commands are
issued in NORMALISED units and converted by the gripper, which is what keeps a
"small" amplitude small on every joint regardless of its individual range.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._common import add_hand_args, build_hand, hand_urdf  # noqa: E402
from quest_teleop2.teleop.hand_retarget import urdf_limits  # noqa: E402

#: Never command more than this, whatever the CLI says. The hand clamps to its
#: own limits internally, but this bounds how wrong a mistake here can be
#: before that clamping is even consulted.
MAX_AMPLITUDE_RAD = 0.35

#: Joint the --move test drives. Index finger, proximal joint: it flexes into
#: free space rather than towards the palm or the other fingers.
DEFAULT_FINGER = 1
DEFAULT_JOINT = 1

FINGERS = ("thumb ", "index ", "middle", "ring  ", "pinky ")

GREEN, YELLOW, RED, BOLD, RESET = (
    "\033[32m", "\033[33m", "\033[31m", "\033[1m", "\033[0m")


def describe(radians: np.ndarray) -> str:
    """(5, 4) radians -> one line per finger."""
    return "\n".join(
        f"    {FINGERS[i]} " + "  ".join(f"{v:+.3f}" for v in row)
        for i, row in enumerate(np.asarray(radians).reshape(5, 4))
    )


def read_once(hand) -> np.ndarray:
    """Measured (5, 4) joint positions in radians."""
    return hand.to_radians(hand.state())


def watch(hand, rate: float, normalise: bool) -> None:
    """Live readout until Ctrl-C. Never commands anything."""
    print(f"\n{BOLD}Live joint readout{RESET} at {rate:.0f} Hz. Ctrl-C to stop.")
    print("  Move the fingers by hand and watch the numbers follow.\n")
    period = 1.0 / max(rate, 1.0)
    try:
        while True:
            value = hand.state() if normalise else read_once(hand)
            unit = "normalised 0..1" if normalise else "radians"
            print(f"\033[6A" if watch.printed else "", end="")
            print(f"  [{unit}]")
            print(describe(np.asarray(value).reshape(5, 4)))
            watch.printed = True
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n  stopped.")


watch.printed = False


def _ramped_sine(hand, index: int, amplitude_norm: float, period: float,
                 cycles: float, rate: float) -> None:
    """One joint, one sine, starting and ending exactly where it started.

    The whole 20-vector is commanded every tick -- every other joint is held at
    the pose the hand was already in, so nothing else drifts while one joint
    moves.
    """
    start = hand.state().copy()
    dt = 1.0 / rate
    ticks = int(cycles * period * rate)
    try:
        for k in range(ticks + 1):
            phase = 2.0 * math.pi * k / (period * rate)
            # (1 - cos) starts at 0 and ends at 0 on whole cycles, so the joint
            # eases in and returns rather than stepping at either end.
            offset = amplitude_norm * 0.5 * (1.0 - math.cos(phase))
            target = start.copy()
            target[index] = np.clip(start[index] + offset, 0.0, 1.0)
            hand.command(target)
            time.sleep(dt)
    finally:
        # Always return to the starting pose, including on Ctrl-C. On a clean
        # finish the sine already ended there, so this is a no-op; interrupted
        # mid-swing it eases back rather than letting the joint sit wherever
        # the exception happened to land it.
        here = hand.state().copy()
        steps = max(1, int(0.5 * rate))
        for k in range(1, steps + 1):
            a = k / steps
            hand.command(here + (start - here) * a)
            time.sleep(dt)


def sine_one_joint(hand, finger: int, joint: int, amplitude_rad: float,
                   period: float, cycles: float, rate: float,
                   limits: tuple[np.ndarray, np.ndarray]) -> None:
    lower, upper = limits
    span = float(upper[finger, joint] - lower[finger, joint])
    amplitude_norm = amplitude_rad / span if span > 1e-9 else 0.0
    index = finger * 4 + joint
    print(f"\n{BOLD}{FINGERS[finger].strip()} joint {joint + 1}{RESET}: "
          f"{amplitude_rad:.3f} rad peak ({amplitude_norm:.3f} normalised), "
          f"{period:.1f}s period, {cycles:g} cycles")
    print(f"  joint range {lower[finger, joint]:+.3f} .. {upper[finger, joint]:+.3f} rad")
    _ramped_sine(hand, index, amplitude_norm, period, cycles, rate)
    print("  returned to start.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--watch", action="store_true",
                        help="live joint readout, never commands anything")
    parser.add_argument("--normalise", action="store_true",
                        help="show normalised 0..1 instead of radians")
    parser.add_argument("--move", action="store_true",
                        help="drive one joint (asks first)")
    parser.add_argument("--fingers", action="store_true",
                        help="drive one joint on each finger in turn (asks first)")
    parser.add_argument("--finger", type=int, default=DEFAULT_FINGER,
                        help="finger index 0-4")
    parser.add_argument("--joint", type=int, default=DEFAULT_JOINT,
                        help="joint index 0-3")
    parser.add_argument("--amplitude", type=float, default=0.15,
                        help="radians, peak")
    parser.add_argument("--period", type=float, default=4.0,
                        help="seconds per cycle")
    parser.add_argument("--cycles", type=float, default=2.0,
                        help="how many cycles to run")
    parser.add_argument("--rate", type=float, default=50.0, help="command Hz")
    add_hand_args(parser)
    parser.add_argument("--serial", default="", help="hand serial, if more than one")
    args = parser.parse_args()

    urdf = hand_urdf(args)
    if not urdf.exists():
        print(f"{RED}URDF not found: {urdf}{RESET}")
        return 1
    limits = urdf_limits(urdf)

    amplitude = min(abs(args.amplitude), MAX_AMPLITUDE_RAD)
    if amplitude < abs(args.amplitude):
        print(f"{YELLOW}amplitude capped to {MAX_AMPLITUDE_RAD} rad{RESET}")
    moving = args.move or args.fingers
    if moving:
        print(f"\n{BOLD}The fingers will move.{RESET} The hand has no e-stop and the "
              f"robot's red\n  button does not cut its power. Unplugging it is the only "
              f"hard stop.")
        if input("\n  type 'move' to continue: ").strip() != "move":
            print("Aborted.")
            return 1

    hand = build_hand(args, limits=limits, serial_number=args.serial)
    try:
        hand.connect()
    except Exception as exc:  # noqa: BLE001 -- the message is the useful part
        print(f"{RED}Could not connect: {exc}{RESET}")
        return 1
    try:
        print(f"\n{GREEN}Connected.{RESET} {hand}")
        print(f"  measured pose, radians:\n{describe(read_once(hand))}")

        if args.watch:
            watch(hand, args.rate, args.normalise)
            return 0
        if args.fingers:
            for f in range(5):
                sine_one_joint(hand, f, args.joint, amplitude, args.period,
                               args.cycles, args.rate, limits)
            return 0
        if args.move:
            sine_one_joint(hand, args.finger, args.joint, amplitude,
                           args.period, args.cycles, args.rate, limits)
            return 0

        print(f"\n  Read-only. Add {BOLD}--watch{RESET} for a live readout, "
              f"{BOLD}--move{RESET} to drive one joint.")
        return 0
    except KeyboardInterrupt:
        print("\n  interrupted.")
        return 1
    finally:
        hand.disconnect()
        print("  fingers de-energised.")


if __name__ == "__main__":
    raise SystemExit(main())

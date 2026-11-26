#!/usr/bin/env python
"""Verify the Quest link and measure the button mapping. No robot.

    python scripts/troubleshooting/quest_button_test.py
    python scripts/troubleshooting/quest_button_test.py --write   # save to teleop.yaml

Run before anything that moves the arm. The documented xr-standard button
indices do not match what Quest Touch controllers report -- the thumbstick is on
axes[2]/[3], not axes[0]/[1] -- so measure rather than trust. Also measures your
own pinch threshold.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quest_teleop2.config import CONFIG_DIR  # noqa: E402
from quest_teleop2.quest.quest_input import QuestInput  # noqa: E402
from quest_teleop2.quest.webxr_server import WebXRServer  # noqa: E402
from scripts._common import add_common_args, load_configs, print_quest_banner, setup_logging  # noqa: E402


def bar(value: float, width: int = 12) -> str:
    filled = int(round(abs(value) * width))
    return ("#" * filled).ljust(width)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="save measured values to teleop.yaml")
    parser.add_argument("--seconds", type=float, default=0.0, help="exit after N seconds")
    add_common_args(parser, robot=False, teleop=True, port=True)
    args = parser.parse_args()
    setup_logging(args.verbose)

    _, teleop_cfg, _ = load_configs(args)
    quest = QuestInput(teleop_cfg.controller)
    server = WebXRServer(teleop_cfg, quest)
    server.start()
    print_quest_banner(teleop_cfg.server.port)

    print("Exercise every control. Watch which index changes:")
    print("  trigger, grip, thumbstick press, X, Y, thumbstick directions,")
    print("  and pinch thumb-to-index on the tracked hand.\n")
    print("Ctrl-C when done.\n")

    seen_pressed: dict[int, int] = {}
    moved_axes: dict[int, float] = {}
    pinch_min, pinch_max = float("inf"), 0.0
    prev_pressed: np.ndarray | None = None
    started = time.monotonic()

    try:
        while True:
            time.sleep(0.05)
            if args.seconds and time.monotonic() - started > args.seconds:
                break
            snap = quest.snapshot()

            hand_line = "hand: \033[31mnot tracked\033[0m"
            if snap.hand is not None and snap.hand_age < 0.5:
                d = snap.hand.pinch_distance
                pinch_min = min(pinch_min, d)
                pinch_max = max(pinch_max, d)
                hand_line = f"hand: \033[32mtracked\033[0m  pinch {d * 1000:5.1f} mm"

            ctrl_line = "ctrl: \033[31mnone\033[0m"
            if snap.controller is not None and snap.controller_age < 0.5:
                c = snap.controller
                if prev_pressed is not None and len(prev_pressed) == len(c.pressed):
                    for i in np.flatnonzero(c.pressed & ~prev_pressed):
                        seen_pressed[int(i)] = seen_pressed.get(int(i), 0) + 1
                prev_pressed = c.pressed.copy()
                for i, v in enumerate(c.axes):
                    moved_axes[i] = max(moved_axes.get(i, 0.0), abs(float(v)))

                pressed = "".join(
                    f"\033[32m{i}\033[0m" if p else f"{i}" for i, p in enumerate(c.pressed)
                )
                axes = " ".join(f"{i}:{v:+.2f}" for i, v in enumerate(c.axes))
                ctrl_line = f"ctrl: buttons[{pressed}]  axes {axes}"

            print(f"\r{hand_line}   {ctrl_line}   ".ljust(140), end="", flush=True)
    except KeyboardInterrupt:
        print("\n")

    if server.input_sources:
        print("\033[1mInput sources the headset reported\033[0m")
        for s in server.input_sources:
            kind = "HAND" if s.get("isHand") else "controller"
            print(f"  {s.get('handedness', '?'):6} {kind:11} "
                  f"buttons={s.get('buttons', 0):2} axes={s.get('axes', 0)}  "
                  f"{','.join(s.get('profiles', []))[:60]}")
        kinds = {s.get("handedness"): s.get("isHand") for s in server.input_sources}
        want = teleop_cfg.controller.handedness
        if kinds.get(want) is not False:
            print(
                f"\n  \033[33mNo real {want} CONTROLLER in the session.\033[0m Quest switches "
                "between hand-tracking\n  and controller modes; a tracked hand also exposes a "
                "gamepad (buttons, no axes),\n  which is why it can look like a controller. Pick "
                "the controller up and move it, then re-run.\n"
            )
        print()

    print("\n\033[1mMeasured\033[0m")
    if seen_pressed:
        print("  buttons pressed (index: times):")
        for index in sorted(seen_pressed):
            print(f"    buttons[{index}]  x{seen_pressed[index]}")
    else:
        print("  \033[33mno button presses seen\033[0m")

    if moved_axes:
        print("  axes moved (index: peak deflection):")
        for index in sorted(moved_axes):
            peak = moved_axes[index]
            flag = "  <-- thumbstick" if peak > 0.5 else ""
            print(f"    axes[{index}]  {peak:.2f} {bar(peak)}{flag}")

    if pinch_min < float("inf"):
        print(f"\n  pinch distance ranged {pinch_min * 1000:.1f} .. {pinch_max * 1000:.1f} mm")
        close = pinch_min + 0.30 * (pinch_max - pinch_min)
        open_ = pinch_min + 0.55 * (pinch_max - pinch_min)
        print(f"  suggested pinch_close_m: {close:.4f}")
        print(f"  suggested pinch_open_m:  {open_:.4f}   (the gap is the hysteresis band)")
    else:
        print("\n  \033[33mno hand tracking seen -- pinch not calibrated\033[0m")
        close = open_ = None

    if args.write and close is not None:
        path = args.teleop_config or (CONFIG_DIR / "teleop.yaml")
        text = Path(path).read_text()
        for key, value in (("pinch_close_m", close), ("pinch_open_m", open_)):
            import re

            text = re.sub(
                rf"^(\s*{key}:\s*)[0-9.]+", rf"\g<1>{value:.4f}", text, flags=re.MULTILINE
            )
        Path(path).write_text(text)
        print(f"\n  wrote pinch thresholds to {path}")
        print("  button indices are NOT written automatically -- check them against")
        print("  the 'controller.buttons' section and edit if they differ.")

    server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

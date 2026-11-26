#!/usr/bin/env python3
"""Learn the glove's noise floor: what it reads when the hand moves but touches nothing.

    python scripts/glove/glove_noise.py --seconds 60

With the glove on the robot hand: keep the hand still and untouched for the first 2 s, then
move it through everything a demonstration does (open, close, spread, turn the wrist) without
touching anything, e.g. with scripts/teleop_all.py --glove in another terminal. Each cell's
floor is the most it rose above rest while moving, times --margin; below its floor a cell
reads exactly zero, above it only the excess counts. Writes config/glove_noise.npz, which the
teleop and recording scripts use when it exists.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quest_teleop2.tactile.contact import ContactFilter  # noqa: E402
from quest_teleop2.tactile.glove import GLOVE_PYTHON, GloveClient, collect  # noqa: E402


def floor_from(rest_frames, moving_frames, margin: float, minimum: float, rate_hz: float):
    """(floor (24, 32), most each cell rose while moving) from the two recordings."""
    f = ContactFilter(rate_hz=rate_hz)
    f.calibrate(rest_frames)
    rise = np.zeros(f.rest.shape)
    for p in moving_frames:
        _, d = f.update(p)
        rise = np.maximum(rise, np.nan_to_num(d))
    floor = np.where(f.valid, np.maximum(margin * rise, minimum), np.nan)
    return floor, np.where(f.valid, rise, np.nan)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=60.0, help="how long to record the moving hand")
    ap.add_argument("--margin", type=float, default=1.25, help="floor = margin x the largest rise")
    ap.add_argument("--minimum", type=float, default=0.01, help="smallest floor any cell gets")
    ap.add_argument("--out", type=Path, default=Path("config/glove_noise.npz"))
    ap.add_argument("--glove-python", type=Path, default=GLOVE_PYTHON)
    args = ap.parse_args()

    glove = GloveClient(python=args.glove_python).start()
    try:
        rate = float(glove.info.get("rate_hz", 117.0))
        input("Hand still, nothing touching the glove. ENTER to learn its rest (2 s)... ")
        rest = collect(glove, 2.0)
        input(f"Now move the hand for {args.seconds:.0f} s without touching anything. ENTER to start... ")
        moving = collect(glove, args.seconds)
    finally:
        glove.close()
    floor, rise = floor_from(rest, moving, args.margin, args.minimum, rate)
    v = ~np.isnan(floor)
    print(f"{len(rest)} rest frames, {len(moving)} moving frames")
    print(f"largest rise while moving: median {np.median(rise[v]):.3f}, 95% {np.percentile(rise[v], 95):.3f}, "
          f"max {rise[v].max():.3f} | floor median {np.median(floor[v]):.3f}, max {floor[v].max():.3f}")
    worst = np.argsort(np.nan_to_num(rise).ravel())[::-1][:5]
    print("cells that rose most without contact (row, col: rise): "
          + ", ".join(f"({i // 32}, {i % 32}): {rise.ravel()[i]:.2f}" for i in worst))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, floor=floor, rise=rise, margin=args.margin, minimum=args.minimum,
             seconds=args.seconds, serial=str(glove.info.get("serial", "")))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

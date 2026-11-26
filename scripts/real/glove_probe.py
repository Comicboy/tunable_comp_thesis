#!/usr/bin/env python3
"""Show which glove cells are pressed, by their raw [row, col] in the 24 x 32 grid.

    python scripts/real/glove_probe.py --out logs/real/glove/dip_lines.npz

Used to find where each fingertip region ends: press firmly along a joint crease (e.g. with a
pen across the DIP joint of one finger, the IP joint of the thumb) and read off which cells
light up. Run in the `teleop` env with the glove plugged in; nothing may touch the glove while
the rest is learned (2 s).

While running, one status line shows the cells pressed hardest right now, each with the region
it lies in according to the supervisor's provisional map (`config/glove_regions.yaml` in
third_party/quest_teleop_2, which follows the sensor sheet's finger strips). Ctrl-C ends the run
and prints, per region, the cells that reached at least --keep of that region's peak during the
whole run, with the rows and columns they span: pressing every crease once in one run gives all
five boundary lines at the end.

The pressure shown is the glove reading above each cell's rest, with the shift shared by most
cells removed (quest_teleop2.tactile.contact.ContactFilter); no noise floor is applied. Uses the
supervisor's glove client and contact filter unchanged.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

from quest_teleop2.tactile.contact import COLS, ROWS, ContactFilter, FingerRegions
from quest_teleop2.tactile.glove import GLOVE_PYTHON, ROOT, GloveClient, collect


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regions", type=Path, default=ROOT / "config" / "glove_regions.yaml",
                    help="region map used to label the cells (default: the supervisor's provisional map)")
    ap.add_argument("--top", type=int, default=6, help="how many of the hardest-pressed cells to show")
    ap.add_argument("--show-min", type=float, default=0.01,
                    help="live line: hide cells pressed less than this (sensor noise)")
    ap.add_argument("--keep", type=float, default=0.5,
                    help="at the end, list the cells that reached this share of their region's peak")
    ap.add_argument("--min-peak", type=float, default=0.02,
                    help="regions whose peak stayed below this count as not pressed")
    ap.add_argument("--out", type=Path, default=None, help="save the per-cell maximum here (.npz)")
    ap.add_argument("--glove-python", type=Path, default=GLOVE_PYTHON)
    args = ap.parse_args()

    # Which region each cell belongs to, for the labels (-1 = in no region, e.g. outside the sensors).
    regions = FingerRegions.load(args.regions)
    owner = np.full((ROWS, COLS), -1, dtype=int)
    for i, mask in enumerate(regions.masks):
        owner[mask] = i

    def label(cell: int) -> str:
        i = owner.ravel()[cell]
        return regions.names[i] if i >= 0 else "-"

    glove = GloveClient(python=args.glove_python).start()
    peak = np.zeros((ROWS, COLS))
    filt = ContactFilter(rate_hz=float(glove.info.get("rate_hz", 117.0)))
    try:
        input("Nothing touching the glove. ENTER to learn its rest (2 s)... ")
        filt.calibrate(collect(glove, 2.0))
        print("Press along a crease. Ctrl-C to finish.")
        width = shutil.get_terminal_size((120, 24)).columns - 1
        seq, last_print = None, 0.0
        while True:
            if glove.lost:
                raise ConnectionError("the glove server stopped")
            frame = glove.latest()
            if frame is None or frame.sequence == seq:
                time.sleep(0.002)
                continue
            seq = frame.sequence
            _, excess = filt.update(frame.pressure)  # pressure above rest, NaN outside the sensors
            d = np.nan_to_num(excess)
            peak = np.maximum(peak, d)
            now = time.monotonic()
            if now - last_print >= 0.2:  # 5 Hz is enough to read
                last_print = now
                flat = d.ravel()
                top = [i for i in np.argsort(flat)[::-1][:args.top] if flat[i] >= args.show_min]
                line = " | ".join(f"({i // COLS:2d},{i % COLS:2d}) {flat[i]:.2f} {label(i)}" for i in top)
                print("\r" + (line or "nothing pressed")[:width].ljust(width), end="", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        glove.close()

    # Per region: the cells that came close to the region's own peak = where the crease was pressed.
    print(f"\nCells that reached >= {args.keep:.0%} of their region's peak (row, col):")
    for i, name in enumerate(regions.names):
        p = np.where(regions.masks[i], peak, 0.0)
        top = p.max()
        if top < args.min_peak:
            print(f"  {name:7s} not pressed (peak {top:.3f})")
            continue
        rows, cols = np.nonzero(p >= args.keep * top)
        cells = sorted(zip(rows.tolist(), cols.tolist()))
        print(f"  {name:7s} peak {top:.2f}  rows {rows.min()}-{rows.max()}  cols {cols.min()}-{cols.max()}  "
              + " ".join(f"({r},{c})" for r, c in cells))
    outside = np.where(owner < 0, peak, 0.0)
    if outside.max() >= args.min_peak:
        r, c = np.unravel_index(np.argmax(outside), outside.shape)
        print(f"  (outside every region: peak {outside.max():.2f} at ({r},{c}))")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, peak=peak, rest=filt.rest if filt.ready else np.full((ROWS, COLS), np.nan),
                 regions=str(args.regions), keep=args.keep, serial=str(glove.info.get("serial", "")))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
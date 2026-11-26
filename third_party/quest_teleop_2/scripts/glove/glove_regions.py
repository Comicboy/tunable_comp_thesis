#!/usr/bin/env python3
"""Map the glove's cells to fingers, with the glove on the robot hand.

    python scripts/glove/glove_regions.py

For each finger in turn, press along all of its pads, tip included, for --seconds. A cell
belongs to the finger it was touched most during; valid cells never touched go to the palm.
Writes config/glove_regions.yaml.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quest_teleop2.tactile.contact import FINGERS, ContactFilter, FingerRegions  # noqa: E402
from quest_teleop2.tactile.glove import GLOVE_PYTHON, GloveClient, collect  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=6.0, help="pressing time per finger")
    ap.add_argument("--out", type=Path, default=Path("config/glove_regions.yaml"))
    ap.add_argument("--glove-python", type=Path, default=GLOVE_PYTHON)
    args = ap.parse_args()

    glove = GloveClient(python=args.glove_python).start()
    counts = np.zeros((len(FINGERS), 24, 32))
    try:
        f = ContactFilter(rate_hz=float(glove.info.get("rate_hz", 117.0)))
        input("Nothing touching the glove. ENTER to learn its rest (2 s)... ")
        f.calibrate(collect(glove, 2.0))
        for i, name in enumerate(FINGERS):
            input(f"Press along the {name}'s pads, tip included, for {args.seconds:.0f} s. ENTER to start... ")

            def count(p, i=i):
                lit, _ = f.update(p)
                counts[i] += lit

            collect(glove, args.seconds, count)
            print(f"  {name}: {int((counts[i] > 0).sum())} cells touched")
    finally:
        glove.close()
    valid = f.valid
    owner = np.argmax(counts, axis=0)
    touched = counts.max(axis=0) > 0
    cells = {name: [[int(r), int(c)] for r, c in zip(*np.nonzero(touched & valid & (owner == i)))]
             for i, name in enumerate(FINGERS)}
    cells["palm"] = [[int(r), int(c)] for r, c in zip(*np.nonzero(valid & ~touched))]
    missing = [name for name in FINGERS if not cells[name]]
    if missing:
        print(f"no cells for {', '.join(missing)}: {args.out} left unchanged. Press harder, or longer.")
        return 1
    FingerRegions(cells)
    head = ("# Which glove cells belong to which finger, as [row, col] in the glove's 24 x 32 grid.\n"
            "# Written by scripts/glove/glove_regions.py. The palm is recorded but does not drive\n"
            "# the admittance.\n")
    args.out.write_text(head + yaml.safe_dump(cells, default_flow_style=None, sort_keys=False, width=100))
    print("cells per region: " + ", ".join(f"{k} {len(v)}" for k, v in cells.items()) + f"; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

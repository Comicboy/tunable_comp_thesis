#!/usr/bin/env python3
"""Fingertip regions of the tactile glove, from two glove_probe.py runs.

    python scripts/real/glove_tip_regions.py logs/real/glove/dip_lines.npz logs/real/glove/tip_pads.npz \
        --out config/real/glove_tip_regions.yaml --figure logs/real/glove/tip_regions.png

Inputs (both written by scripts/real/glove_probe.py --out, per-cell peak pressure above rest):
  1. crease run: the crease of each finger's last joint (DIP; IP on the thumb) pressed with a pen;
  2. pad run:    each fingertip pad swept with a fingertip, from the crease to the tip.

Method (agreed 2026-11-25). The glove is a Hand-1 glove on a Hand 2, pulled to the fingertips;
it has sensors on the palm side only. Each finger lies on one strip of the sensor sheet (the
columns of the strip are fixed by the sheet; the supervisor's provisional map, which follows the
sheet layout, gives them). Along a strip the fingertip lies at the high-row end. The joint shows
as a weak row between two rows that light up in the crease run (the glove bridges the gap between
the two segments, the pen presses the sheet onto the segment edges); the fingertip region is every
cell of the strip beyond that joint row whose pad-run peak reaches --threshold of the finger's
own pad-run peak (this drops columns of the strip the narrower Hand-2 finger does not cover).
The joint rows are read off the row profiles this script prints and passed with --joint-row; the
defaults are the ones found on 2026-11-25. All cells outside the five fingertip regions form the
"palm" region (= not a fingertip), which is recorded but never drives the admittance.

Writes the region map (the format of quest_teleop2.tactile.contact.FingerRegions) and, with
--figure, a picture of both runs with the chosen cells, in the raw grid orientation (row 0 at the
top; the supervisor's glove_view.py shows the grid flipped).
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from quest_teleop2.tactile.contact import COLS, FINGERS, ROWS, FingerRegions
from quest_teleop2.tactile.glove import ROOT

# Last-joint row per finger strip, read off the profiles on 2026-11-25 (glove marked on the hand).
JOINT_ROWS = {"thumb": 11, "index": 19, "middle": 20, "ring": 20, "little": 16}


def parse_joint_rows(items: list[str]) -> dict[str, int]:
    rows = dict(JOINT_ROWS)
    for item in items:
        name, _, value = item.partition("=")
        if name not in FINGERS or not value:
            raise SystemExit(f"--joint-row takes NAME=ROW with NAME in {list(FINGERS)}, not {item!r}")
        rows[name] = int(value)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("crease", type=Path, help="glove_probe.py output of the crease run (.npz)")
    ap.add_argument("pads", type=Path, help="glove_probe.py output of the pad run (.npz)")
    ap.add_argument("--strips", type=Path, default=ROOT / "config" / "glove_regions.yaml",
                    help="map whose finger regions give each finger's strip (default: the supervisor's)")
    ap.add_argument("--joint-row", action="append", default=[], metavar="NAME=ROW",
                    help=f"last-joint row of a finger strip (repeatable; defaults {JOINT_ROWS})")
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="keep fingertip cells reaching this share of the finger's pad-run peak")
    ap.add_argument("--min-peak", type=float, default=0.05,
                    help="a finger whose pad-run peak stays below this was not pressed: error")
    ap.add_argument("--out", type=Path, default=Path("config/real/glove_tip_regions.yaml"))
    ap.add_argument("--figure", type=Path, default=None, help="save a picture of both runs here (.png)")
    args = ap.parse_args()

    crease = np.load(args.crease)["peak"]
    pads = np.load(args.pads)["peak"]
    strips = FingerRegions.load(args.strips)
    joint_rows = parse_joint_rows(args.joint_row)
    row_index = np.arange(ROWS)[:, None] * np.ones((1, COLS), dtype=int)

    cells: dict[str, list[list[int]]] = {}
    tips = np.zeros((len(FINGERS), ROWS, COLS), bool)
    print("Row profile per strip (max over the strip's columns, x100): crease run | pad run; * = joint row")
    for i, name in enumerate(FINGERS):
        strip = strips.masks[strips.names.index(name)]
        rows = np.nonzero(strip.any(axis=1))[0]
        prof = " ".join(f"{'*' if r == joint_rows[name] else ' '}r{r}:{crease[r][strip[r]].max() * 100:3.0f}"
                        f"|{pads[r][strip[r]].max() * 100:3.0f}" for r in rows)
        print(f"  {name:6s} {prof}")
        candidate = strip & (row_index > joint_rows[name])
        peak = pads[candidate].max() if candidate.any() else 0.0
        if peak < args.min_peak:
            raise SystemExit(f"{name}: pad-run peak {peak:.3f} beyond row {joint_rows[name]} is below "
                             f"--min-peak {args.min_peak}; was its pad pressed, and is the joint row right?")
        tips[i] = candidate & (pads >= args.threshold * peak)
        cells[name] = [[int(r), int(c)] for r, c in zip(*np.nonzero(tips[i]))]

    rest = ~tips.any(axis=0)
    cells["palm"] = [[int(r), int(c)] for r, c in zip(*np.nonzero(rest))]
    FingerRegions(cells)  # the supervisor's checks: known names, cells inside the grid, no overlaps

    print(f"\nFingertip cells (pad-run peak >= {args.threshold:.0%} of the finger's peak, beyond its joint row):")
    for i, name in enumerate(FINGERS):
        r, c = np.nonzero(tips[i])
        print(f"  {name:6s} {len(r):2d} cells  rows {r.min()}-{r.max()}  cols {c.min()}-{c.max()}  "
              f"pad peak {pads[tips[i]].max():.2f}")

    head = (
        "# Fingertip regions of the tactile glove (Hand-1 glove on the Hand 2, pulled to the fingertips),\n"
        "# as [row, col] in the glove's 24 x 32 grid. Only the five fingers drive the admittance; 'palm' holds\n"
        "# every other cell (palm and the lower phalanges) and is recorded only.\n"
        f"# Written by scripts/real/glove_tip_regions.py on {date.today().isoformat()} from\n"
        f"#   crease run {args.crease}, pad run {args.pads}, strips {args.strips.name},\n"
        f"#   joint rows {joint_rows}, threshold {args.threshold}.\n"
        "# Valid only for the glove position it was measured in (marked on the hand).\n"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(head + yaml.safe_dump(cells, default_flow_style=None, sort_keys=False, width=100))
    print(f"wrote {args.out}")

    if args.figure is not None:
        draw(crease, pads, tips, strips, joint_rows, args.figure)
        print(f"wrote {args.figure}")
    return 0


def draw(crease, pads, tips, strips, joint_rows, path: Path) -> None:
    """Both runs side by side, one sequential scale; chosen cells outlined and named, joint rows dashed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    vmax = max(crease.max(), pads.max())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
    for ax, data, title in zip(axes, (crease, pads), ("Crease run (last joint pressed)", "Pad run (fingertip swept)")):
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xticks(range(0, COLS, 2))
        ax.set_yticks(range(0, ROWS, 2))
        ax.tick_params(labelsize=8, length=0)
        ax.set_xlabel("column", fontsize=9)
        ax.set_ylabel("row", fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i, name in enumerate(FINGERS):
            for r, c in zip(*np.nonzero(tips[i])):
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="#1a1a1a", linewidth=1.2))
            strip = strips.masks[strips.names.index(name)]
            cols = np.nonzero(strip.any(axis=0))[0]
            jr = joint_rows[name]
            ax.plot([cols.min() - 0.5, cols.max() + 0.5], [jr, jr], linestyle="--", color="#666666", linewidth=1)
            ax.text((cols.min() + cols.max()) / 2, jr - 0.3, name, ha="center", va="bottom", fontsize=8,
                    color="#1a1a1a")
    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    cbar.set_label("peak pressure above rest (glove units)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle("Fingertip regions: outlined cells; dashed = last-joint row of each strip; row 0 at the top",
                 fontsize=10, x=0.01, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
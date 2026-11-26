#!/usr/bin/env python3
"""Probe the fingertip map: touch each fingertip in turn and see where the contact lands.

    python scripts/real/glove_tip_check.py --out logs/real/glove/tip_check.npz \
        --figure logs/real/glove/tip_check.png

For each finger in turn (thumb .. little), go over its fingertip as thoroughly as you can -
anywhere between the very tip and the last joint's crease (DIP; IP on the thumb) - and press
ENTER when done; recording of the next finger starts right away. Let go before pressing ENTER.

The glove is filtered as the admittance will see it: the supervisor's ContactFilter with our
noise floor, contacts only (quest_teleop2.tactile.contact, unchanged). Afterwards, per touched
finger:

  - where the contact pressure went: its own fingertip region, another finger's region
    (cross-talk), or no fingertip at all ("not a fingertip": misclassified, since only the
    fingertip was touched);
  - the pressure per row along the finger's strip, fingertip rows marked, so rows that should
    belong to the fingertip but do not (or the reverse) stand out;
  - fingertip cells that were never in contact (parts of the region the finger may not cover);
  - the cells in contact outside the region, strongest first.

"Pressure" is the contact pressure above rest and above the floor, summed over the frames of the
touch, so a cell pressed longer or harder counts more. --figure draws, per finger, where the
contact landed with all fingertip regions outlined. --out saves everything for later analysis.
Run in the teleop env; nothing may touch the glove while the rest is learned (2 s).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np

from quest_teleop2.tactile.contact import COLS, FINGERS, ROWS, ContactFilter, FingerRegions, load_floor
from quest_teleop2.tactile.glove import GLOVE_PYTHON, ROOT, GloveClient, collect


class Touch:
    """Accumulates the contacts of one touch on a background thread until stopped, and shows a live
    status line: which finger is being recorded, and where the contact is right now."""

    def __init__(self, glove: GloveClient, filt: ContactFilter, name: str, tip: np.ndarray,
                 other_tips: np.ndarray) -> None:
        self.glove, self.filt = glove, filt
        self.name, self.tip, self.other_tips = name, tip, other_tips
        self.pressure = np.zeros((ROWS, COLS))  # contact pressure summed over the frames
        self.peak = np.zeros((ROWS, COLS))      # strongest contact pressure per cell
        self.frames = 0                         # frames with contact anywhere
        self.error: Exception | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> Touch:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        if self.error is not None:
            raise self.error

    def _status(self, c: np.ndarray, start: float) -> None:
        line = (f"[REC {self.name.upper()}] {time.monotonic() - start:5.1f} s | now: own fingertip "
                f"{c[self.tip].sum():5.2f}  other fingertips {c[self.other_tips].sum():5.2f}  not a fingertip "
                f"{c[~(self.tip | self.other_tips)].sum():5.2f} | frames with contact {self.frames}")
        width = shutil.get_terminal_size((120, 24)).columns - 1
        sys.stdout.write("\r" + line[:width].ljust(width))
        sys.stdout.flush()

    def _run(self) -> None:
        seq, start, last = None, time.monotonic(), 0.0
        try:
            while not self._stop.is_set():
                if self.glove.lost:
                    raise ConnectionError("the glove server stopped")
                frame = self.glove.latest()
                if frame is None or frame.sequence == seq:
                    time.sleep(0.002)
                    continue
                seq = frame.sequence
                lit, excess = self.filt.update(frame.pressure)
                c = np.where(lit, np.nan_to_num(excess), 0.0)
                self.pressure += c
                np.maximum(self.peak, c, out=self.peak)
                self.frames += bool(lit.any())
                if time.monotonic() - last >= 0.2:  # 5 Hz is enough to read
                    last = time.monotonic()
                    self._status(c, start)
        except Exception as exc:  # noqa: BLE001 -- handed to the main thread in stop()
            self.error = exc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regions", type=Path, default=Path("config/real/glove_tip_regions.yaml"),
                    help="the fingertip map under test")
    ap.add_argument("--noise", type=Path, default=Path("config/real/glove_noise.npz"),
                    help="noise floor from glove_noise.py (runs without one if the file does not exist)")
    ap.add_argument("--strips", type=Path, default=ROOT / "config" / "glove_regions.yaml",
                    help="map whose finger regions give each finger's strip of the sensor sheet")
    ap.add_argument("--fingers", nargs="+", default=list(FINGERS), choices=list(FINGERS),
                    help="which fingers to probe (default all, thumb to little)")
    ap.add_argument("--out", type=Path, default=None, help="save the per-finger results here (.npz)")
    ap.add_argument("--figure", type=Path, default=None, help="save a picture per finger here (.png)")
    ap.add_argument("--glove-python", type=Path, default=GLOVE_PYTHON)
    args = ap.parse_args()

    regions = FingerRegions.load(args.regions)
    strips = FingerRegions.load(args.strips)
    floor = load_floor(args.noise)
    if floor is None:
        print(f"WARNING: no noise floor at {args.noise}; bending artifacts will count as contact.")
    tip = {name: regions.masks[regions.names.index(name)] for name in FINGERS}
    tips_any = np.any([tip[n] for n in FINGERS], axis=0)

    touches: list[Touch] = []
    glove = GloveClient(python=args.glove_python).start()
    try:
        filt = ContactFilter(rate_hz=float(glove.info.get("rate_hz", 117.0)), floor=floor)
        input("Nothing touching the glove, fingers still. ENTER to learn its rest (2 s)... ")
        filt.calibrate(collect(glove, 2.0))
        for name in args.fingers:
            print(f"\n>>> {name.upper()}: go over its fingertip (tip to last joint). Let go, then ENTER when done.")
            others = np.any([tip[o] for o in FINGERS if o != name], axis=0)
            touch = Touch(glove, filt, name, tip[name], others).start()
            try:
                input()
            finally:
                touch.stop()
            touches.append(touch)
            # Clear a status line the thread may have written after ENTER, then report.
            width = shutil.get_terminal_size((120, 24)).columns - 1
            sys.stdout.write("\r" + " " * width + "\r")
            print(f"  {name} done: {touch.frames} frames with contact")
    finally:
        glove.close()

    for name, touch in zip(args.fingers, touches):
        report(name, touch, tip, tips_any, strips)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, fingers=np.array(args.fingers),
                 pressure=np.stack([t.pressure for t in touches]), peak=np.stack([t.peak for t in touches]),
                 frames=np.array([t.frames for t in touches]), regions=str(args.regions),
                 noise=str(args.noise) if floor is not None else "", serial=str(glove.info.get("serial", "")))
        print(f"\nwrote {args.out}")
    if args.figure is not None:
        draw(args.fingers, touches, tip, args.figure)
        print(f"wrote {args.figure}")
    return 0


def report(name: str, touch: Touch, tip, tips_any, strips) -> None:
    pressure, peak = touch.pressure, touch.peak
    total = pressure.sum()
    print(f"\n== {name} touched ({touch.frames} frames with contact)")
    if total <= 0:
        print("   no contact registered: pressed too lightly, or the floor is too high here")
        return
    own = pressure[tip[name]].sum() / total
    others = {o: pressure[tip[o]].sum() / total for o in FINGERS if o != name}
    outside = pressure[~tips_any].sum() / total
    cross = ", ".join(f"{o} {s:.0%}" for o, s in others.items() if s >= 0.005) or "none"
    print(f"   pressure share: own fingertip {own:.0%} | other fingertips {sum(others.values()):.0%} ({cross})"
          f" | not a fingertip {outside:.0%}")

    strip = strips.masks[strips.names.index(name)]
    strip_total = pressure[strip].sum()
    if strip_total > 0:
        rows = np.nonzero(strip.any(axis=1))[0]
        prof = " ".join(f"r{r}{'[T]' if tip[name][r].any() else ''}:{pressure[r][strip[r]].sum() / strip_total:.0%}"
                        for r in rows if pressure[r][strip[r]].sum() / strip_total >= 0.005)
        print(f"   along its strip ([T] = fingertip row; share of the strip's pressure): {prof}")
    print(f"   {strip_total / total:.0%} of the pressure was on this finger's strip")

    never = [(int(r), int(c)) for r, c in zip(*np.nonzero(tip[name] & (peak <= 0)))]
    print(f"   fingertip cells never in contact ({len(never)} of {int(tip[name].sum())}): "
          + (" ".join(f"({r},{c})" for r, c in never) or "none"))

    out_cells = np.where(~tip[name], pressure, 0.0).ravel()
    order = np.argsort(out_cells)[::-1][:6]
    worst = [(i // COLS, i % COLS, out_cells[i] / total) for i in order if out_cells[i] / total >= 0.005]
    print("   strongest cells in contact outside its fingertip (row, col: share): "
          + (", ".join(f"({r},{c}): {s:.0%}" for r, c, s in worst) or "none"))


def draw(fingers, touches, tip, path: Path) -> None:
    """One panel per touched finger: where its contact landed (share of that touch's pressure),
    the touched finger's region outlined bold, the other fingertip regions thin; row 0 at the top."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = len(fingers)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.9 * nrows), constrained_layout=True,
                             squeeze=False)
    im = None
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for ax, name, touch in zip(axes.ravel(), fingers, touches):
        total = touch.pressure.sum()
        share = touch.pressure / total if total > 0 else touch.pressure
        im = ax.imshow(share, cmap="Blues", vmin=0, vmax=max(float(share.max()), 1e-9), interpolation="nearest")
        for other in FINGERS:
            bold = other == name
            for r, c in zip(*np.nonzero(tip[other])):
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                       edgecolor="#1a1a1a" if bold else "#9a9a9a",
                                       linewidth=1.4 if bold else 0.6))
        own = touch.pressure[tip[name]].sum() / total if total > 0 else 0.0
        ax.set_title(f"{name} touched: {own:.0%} in its fingertip", fontsize=10, loc="left")
        ax.set_xticks(range(0, COLS, 4))
        ax.set_yticks(range(0, ROWS, 4))
        ax.tick_params(labelsize=7, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.7)
        cbar.set_label("share of the touch's contact pressure (scale per panel)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    fig.suptitle("Fingertip probe: bold = the touched finger's region, grey = the other fingertips; "
                 "row 0 at the top", fontsize=9, x=0.01, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
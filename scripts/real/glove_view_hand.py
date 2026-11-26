#!/usr/bin/env python3
"""Live view of the tactile glove drawn on the Hand 2, seen from the palm side.

    python scripts/real/glove_view_hand.py
    python scripts/real/glove_view_hand.py --demo        # no glove: synthetic presses, to check the layout

The hand outline comes from the Hand 2 URDF (tunable_comp_real.hand_model, open pose), so the
proportions are the real hand's. Every glove cell is drawn where it lies on the hand:

  - fingertip cells (our map, config/real/glove_tip_regions.yaml): full colour and outlined, on the
    distal segment of their finger;
  - the rest of each finger's strip of the sensor sheet: faint, along the lower segments;
  - the palm block: faint, in the palm.

Colour = contact pressure above rest (and above the noise floor, if one exists), one sequential
scale; `v` switches to the raw pressure above rest (before the contact decision). On the right,
per finger, the force the admittance would see: the sum over its fingertip cells in contact
(glove units until the newton calibration exists), as a number and a bar; below it the pressure
on everything that is not a fingertip.

Placement along a finger: the strip's last-joint row (one row before the fingertip region) sits
on the finger's last joint (DIP; IP on the thumb), the last fingertip row at the fingertip, and
the strip's first row at the finger's first joint; columns run across the finger, rising towards
the little-finger side (as on the sensor sheet). Geometry of the cells is approximate; their
assignment to fingers and fingertips is exact.

Keys: b relearn the rest (touch nothing for 2 s) | v contacts / raw | + - colour scale | q quit.
Uses the supervisor's glove client and contact filter unchanged (quest_teleop2.tactile).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from quest_teleop2.tactile.contact import COLS, FINGERS, ROWS, ContactFilter, FingerRegions, load_floor
from quest_teleop2.tactile.glove import GLOVE_PYTHON, ROOT, GloveClient, collect
from tunable_comp_real.hand_model import HandModel

# -- look ---------------------------------------------------------------------------------- #
SCALE = 3.4                    # pixels per millimetre
MARGIN = 40                    # pixels around the hand
PANEL_W = 300                  # width of the force panel on the right
BG = (255, 255, 255)           # BGR
HAND_FILL = (236, 236, 236)
HAND_EDGE = (205, 205, 205)
INK = (40, 40, 40)
INK_SOFT = (120, 120, 120)
FINGER_WIDTH_MM = {"thumb": 20.0, "index": 17.0, "middle": 17.0, "ring": 16.0, "little": 14.0}
# Sequential blue ramp, light -> dark (RGB), for the pressure.
RAMP = np.array([[239, 243, 250], [198, 219, 239], [107, 174, 214], [33, 113, 181], [8, 48, 107]], float)


def ramp(v: np.ndarray) -> np.ndarray:
    """Values in 0..1 -> BGR colours on the sequential ramp."""
    x = np.clip(v, 0.0, 1.0) * (len(RAMP) - 1)
    i = np.minimum(x.astype(int), len(RAMP) - 2)
    t = (x - i)[..., None]
    rgb = RAMP[i] * (1 - t) + RAMP[i + 1] * t
    return rgb[..., ::-1]


class HandLayout:
    """Where each glove cell is drawn: a quadrilateral in pixels, and what kind of cell it is."""

    def __init__(self, hand: HandModel, tips: FingerRegions, strips: FingerRegions) -> None:
        poses = hand.fk(np.zeros(20))
        # Palm-side view in millimetres: x = x of r_wrist (thumb on the right), y = -z (fingers up).
        self.lines = {f: np.vstack([p.joint_pos, p.tip_pos])[:, [0, 2]] * [1000.0, -1000.0]
                      for f, p in poses.items()}
        pts = np.vstack(list(self.lines.values()) + [np.zeros((1, 2))])
        self.lo = pts.min(axis=0) - 12.0
        hi = pts.max(axis=0) + 12.0
        self.size = (int((hi[0] - self.lo[0]) * SCALE) + 2 * MARGIN, int((hi[1] - self.lo[1]) * SCALE) + 2 * MARGIN)
        self.hi_y = hi[1]

        self.kind = np.full((ROWS, COLS), "", dtype=object)   # 'tip:<finger>', 'strip', 'palm'
        self.quads: dict[tuple[int, int], np.ndarray] = {}
        for name in FINGERS:
            tip = tips.masks[tips.names.index(name)]
            strip = strips.masks[strips.names.index(name)] | tip
            self._place_finger(name, strip, tip)
        palm = strips.masks[strips.names.index("palm")] & (self.kind == "")
        self._place_palm(palm)

    # millimetres (view) -> pixels
    def px(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, float)
        x = (p[..., 0] - self.lo[0]) * SCALE + MARGIN
        y = (self.hi_y - p[..., 1]) * SCALE + MARGIN
        return np.stack([x, y], axis=-1)

    def _place_finger(self, name: str, strip: np.ndarray, tip: np.ndarray) -> None:
        line = self.lines[name]                                     # joints 1..4, then the fingertip
        seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
        s_at = np.concatenate([[0.0], np.cumsum(seg)])              # arc length at each point
        s_joint, s_tip = s_at[3], s_at[4]                           # last joint, fingertip
        rows = np.nonzero(strip.any(axis=1))[0]
        tip_rows = np.nonzero(tip.any(axis=1))[0]
        r_joint, r_last, r_first = tip_rows.min() - 1, tip_rows.max(), rows.min()

        def s_of(r: float) -> float:
            """Row coordinate -> arc length: first row edge at the base, joint row at the last
            joint, last row's far edge at the fingertip."""
            if r >= r_joint:
                return s_joint + (r - r_joint) / (r_last + 0.5 - r_joint) * (s_tip - s_joint)
            return (r - (r_first - 0.5)) / (r_joint - (r_first - 0.5)) * s_joint

        def at(s: float) -> tuple[np.ndarray, np.ndarray]:
            k = int(np.clip(np.searchsorted(s_at, s) - 1, 0, len(seg) - 1))
            d = (line[k + 1] - line[k]) / seg[k]
            return line[k] + d * (s - s_at[k]), np.array([-d[1], d[0]])  # point, left normal

        cols = np.nonzero(strip.any(axis=0))[0]
        c_mid, n_cols = (cols.min() + cols.max()) / 2.0, cols.max() - cols.min() + 1
        w = FINGER_WIDTH_MM[name] / n_cols
        for r, c in zip(*np.nonzero(strip)):
            (p0, n0), (p1, n1) = at(s_of(r - 0.5)), at(s_of(r + 0.5))
            # Columns rise towards the little-finger side = the left normal of the finger in this view.
            a, b = (c - c_mid - 0.5) * w, (c - c_mid + 0.5) * w
            quad = np.array([p0 + n0 * a, p0 + n0 * b, p1 + n1 * b, p1 + n1 * a])
            self.quads[(r, c)] = self.px(quad)
            self.kind[r, c] = f"tip:{name}" if tip[r, c] else "strip"

    def _place_palm(self, palm: np.ndarray) -> None:
        if not palm.any():
            return
        rows, cols = np.nonzero(palm)
        # Span of the palm block from its bulk; the few cells beyond (e.g. column 31 at the
        # little-finger edge) are extrapolated to the side instead of squeezing the block.
        r0, r1 = rows.min(), rows.max()
        c0, c1 = int(np.percentile(cols, 5)), int(np.percentile(cols, 95))
        bases = {f: self.lines[f][0] for f in ("index", "little")}
        top_right, top_left = bases["index"] - [0, 6.0], bases["little"] - [0, 6.0]
        bot_right, bot_left = np.array([18.0, 0.0]), np.array([-30.0, 0.0])

        def corner(r: float, c: float) -> np.ndarray:
            u = (c - (c0 - 0.5)) / (c1 - c0 + 1.0)              # 0 = index side, 1 = little side
            v = (r - (r0 - 0.5)) / (r1 - r0 + 1.0)              # 0 = wrist, 1 = finger bases
            top = top_right + (top_left - top_right) * u
            bot = bot_right + (bot_left - bot_right) * u
            return bot + (top - bot) * v

        for r, c in zip(rows, cols):
            quad = np.array([corner(r - 0.5, c - 0.5), corner(r - 0.5, c + 0.5),
                             corner(r + 0.5, c + 0.5), corner(r + 0.5, c - 0.5)])
            self.quads[(r, c)] = self.px(quad)
            self.kind[r, c] = "palm"

    def outline(self, img: np.ndarray) -> None:
        """Palm and fingers as light shapes under the cells."""
        palm = [self.lines[f][0] for f in ("index", "middle", "ring", "little")]
        palm_poly = np.array(palm + [[-30.0, 0.0], [18.0, 0.0], self.lines["thumb"][0]])
        hull = cv2.convexHull(self.px(palm_poly).astype(np.int32))
        cv2.fillConvexPoly(img, hull, HAND_FILL, cv2.LINE_AA)
        for name, line in self.lines.items():
            pts = self.px(line).astype(np.int32)
            thick = int(FINGER_WIDTH_MM[name] * SCALE)
            cv2.polylines(img, [pts], False, HAND_EDGE, thick + 4, cv2.LINE_AA)
            cv2.polylines(img, [pts], False, HAND_FILL, thick, cv2.LINE_AA)


def render(layout: HandLayout, values: np.ndarray, forces: np.ndarray, scale: float,
           header: str, force_scale: float) -> np.ndarray:
    w, h = layout.size
    img = np.full((max(h, 420), w + PANEL_W, 3), BG, np.uint8)
    layout.outline(img)
    colours = ramp(values / scale)
    for (r, c), quad in layout.quads.items():
        kind = layout.kind[r, c]
        col = colours[r, c]
        if not kind.startswith("tip:"):
            col = col * 0.45 + np.array(HAND_FILL) * 0.55           # faint outside the fingertips
        cv2.fillConvexPoly(img, quad.astype(np.int32), tuple(int(x) for x in col), cv2.LINE_AA)
        edge = INK_SOFT if kind.startswith("tip:") else HAND_EDGE
        cv2.polylines(img, [quad.astype(np.int32)], True, edge, 1, cv2.LINE_AA)
    for name in FINGERS:
        tip_px = layout.px(layout.lines[name][-1])
        cv2.putText(img, name, (int(tip_px[0]) - 18, int(tip_px[1]) - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, INK, 1, cv2.LINE_AA)

    # Force panel.
    x0, y = w + 10, 60
    cv2.putText(img, "fingertip force (contacts)", (x0, y - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK, 1, cv2.LINE_AA)
    bar_w = PANEL_W - 140
    for i, name in enumerate(FINGERS + ("not tip",)):
        v = forces[i]
        cv2.putText(img, name, (x0, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    INK if i < 5 else INK_SOFT, 1, cv2.LINE_AA)
        cv2.rectangle(img, (x0 + 62, y), (x0 + 62 + bar_w, y + 18), (240, 240, 240), -1)
        filled = int(bar_w * min(v / force_scale, 1.0))
        bar_col = tuple(int(x) for x in ramp(np.array(0.85 if i < 5 else 0.35)))
        if filled > 0:
            cv2.rectangle(img, (x0 + 62, y), (x0 + 62 + filled, y + 18), bar_col, -1)
        cv2.putText(img, f"{v:5.2f}", (x0 + 68 + bar_w, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, INK, 1, cv2.LINE_AA)
        y += 34
    cv2.putText(img, f"bar full scale {force_scale:.1f} (glove units)", (x0, y + 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, INK_SOFT, 1, cv2.LINE_AA)

    # Colour scale.
    y += 40
    grad = ramp(np.linspace(0, 1, bar_w)[None, :].repeat(12, axis=0)).astype(np.uint8)
    img[y:y + 12, x0 + 62:x0 + 62 + bar_w] = grad
    cv2.putText(img, "cell pressure (colour)", (x0, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, INK, 1, cv2.LINE_AA)
    cv2.putText(img, "0", (x0 + 50, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, INK_SOFT, 1, cv2.LINE_AA)
    cv2.putText(img, f"{scale:.2f}", (x0 + 68 + bar_w, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, INK_SOFT, 1, cv2.LINE_AA)

    cv2.putText(img, header, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK, 1, cv2.LINE_AA)
    cv2.putText(img, "b rest  v contacts/raw  + - scale  q quit", (12, img.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, INK_SOFT, 1, cv2.LINE_AA)
    return img


def demo_frames(tips: FingerRegions):
    """Synthetic glove frames: a press moving from fingertip to fingertip, plus a little noise."""
    rng = np.random.default_rng(0)
    base = rng.uniform(0.05, 0.2, (ROWS, COLS))
    k = 0
    while True:
        p = base + rng.normal(0, 0.002, base.shape)
        name = FINGERS[(k // 120) % 5]
        m = tips.masks[tips.names.index(name)]
        phase = (k % 120) / 120.0
        p[m] += 0.35 * np.sin(np.pi * phase) * rng.uniform(0.5, 1.0, m.sum())
        yield p.astype(np.float32)
        k += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regions", type=Path, default=Path("config/real/glove_tip_regions.yaml"),
                    help="our fingertip map")
    ap.add_argument("--noise", type=Path, default=Path("config/real/glove_noise.npz"),
                    help="noise floor (used if the file exists)")
    ap.add_argument("--strips", type=Path, default=ROOT / "config" / "glove_regions.yaml",
                    help="map giving each finger's strip and the palm block (default: the supervisor's)")
    ap.add_argument("--urdf", type=Path, default=None, help="Hand 2 URDF (default: the one in third_party)")
    ap.add_argument("--scale", type=float, default=0.30, help="cell pressure at full colour")
    ap.add_argument("--force-scale", type=float, default=2.0, help="finger force at a full bar")
    ap.add_argument("--demo", action="store_true", help="no glove: synthetic presses (layout check)")
    ap.add_argument("--snapshot", type=Path, default=None, help="with --demo: save one frame here and exit")
    ap.add_argument("--glove-python", type=Path, default=GLOVE_PYTHON)
    args = ap.parse_args()

    tips = FingerRegions.load(args.regions)
    strips = FingerRegions.load(args.strips)
    layout = HandLayout(HandModel(args.urdf), tips, strips)
    floor = load_floor(args.noise)
    filt = ContactFilter(rate_hz=117.0, floor=floor)
    show_raw, scale, force_scale = False, args.scale, args.force_scale

    glove = None
    if args.demo:
        source = demo_frames(tips)
        filt.calibrate([next(source) for _ in range(200)])
        rate = 117.0
    else:
        glove = GloveClient(python=args.glove_python).start()
        rate = float(glove.info.get("rate_hz", 117.0))
        filt = ContactFilter(rate_hz=rate, floor=floor)
        print("Learning the glove's rest: touch nothing for 2 s ...")
        filt.calibrate(collect(glove, 2.0))

    window = "glove on Hand 2 (palm side)"
    if args.snapshot is None:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    values = np.zeros((ROWS, COLS))
    forces = np.zeros(6)
    seq, last_draw = None, 0.0
    try:
        while True:
            if args.demo:
                frame = next(source)
                time.sleep(1.0 / rate)
            else:
                if glove.lost:
                    raise ConnectionError("the glove server stopped")
                f = glove.latest()
                if f is None or f.sequence == seq:
                    time.sleep(0.002)
                    continue
                seq, frame = f.sequence, f.pressure
            lit, excess = filt.update(frame)
            d = np.nan_to_num(excess)
            values = d if show_raw else np.where(lit, d, 0.0)
            forces = tips.forces(d, lit)                       # 5 fingertips, then everything else
            now = time.monotonic()
            if now - last_draw < 1 / 30:                       # draw at ~30 Hz
                continue
            last_draw = now
            header = (f"{'demo' if args.demo else f'{glove.rate_hz:.0f} Hz'}   "
                      f"{'raw pressure above rest' if show_raw else 'contacts'}   "
                      f"noise floor: {'yes' if floor is not None else 'none'}")
            img = render(layout, values, forces, scale, header, force_scale)
            if args.snapshot is not None:
                if args.demo and forces[:5].max() > 0.5 * force_scale:
                    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(args.snapshot), img)
                    print(f"wrote {args.snapshot}")
                    return 0
                continue
            cv2.imshow(window, img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("v"):
                show_raw = not show_raw
            elif key in (ord("+"), ord("=")):
                scale *= 0.8
            elif key == ord("-"):
                scale /= 0.8
            elif key == ord("b") and glove is not None:
                print("Relearning the rest: touch nothing for 2 s ...")
                filt.calibrate(collect(glove, 2.0))
    except KeyboardInterrupt:
        pass
    finally:
        if glove is not None:
            glove.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
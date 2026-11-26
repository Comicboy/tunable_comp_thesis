#!/usr/bin/env python3
"""Live view of the glove exactly as the admittance sees it: contacts, finger regions, forces.

    python scripts/glove/glove_view.py

Reads the glove through its server, so it runs alongside teleoperation and recording. Learns
the rest for 2 s first: touch nothing. Uses config/glove_noise.npz and config/glove_regions.yaml.

Keys: b learn the rest again   v contacts or raw pressure   + - colour sensitivity
      f flip   m mirror   r rotate   q quit
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quest_teleop2.tactile.contact import FINGERS, ContactFilter, FingerRegions, load_floor  # noqa: E402
from quest_teleop2.tactile.glove import GLOVE_PYTHON, GloveClient  # noqa: E402

CELL, BAR, SIDE = 22, 48, 190
OFF, IDLE = (38, 38, 38), (70, 52, 40)
COLOURS = [(80, 200, 255), (80, 255, 120), (255, 200, 80), (255, 120, 200), (180, 140, 255), (160, 160, 160)]


def orient(a, flip, mirror, turns):
    a = a[::-1] if flip else a
    a = a[:, ::-1] if mirror else a
    return np.rot90(a, turns)


def picture(values, valid, lit, regions, full, flip, mirror, turns):
    o = lambda a: orient(a, flip, mirror, turns)  # noqa: E731
    v = np.clip(np.nan_to_num(o(values)) / full, 0.0, 1.0)
    img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    if lit is not None:
        img[~o(lit)] = IDLE
    img[~o(valid)] = OFF
    img = cv2.resize(img, (img.shape[1] * CELL, img.shape[0] * CELL), interpolation=cv2.INTER_NEAREST)
    img[::CELL, :] = 0
    img[:, ::CELL] = 0
    for i, name in enumerate(regions.names):
        m = o(regions.masks[i])
        if m.any():
            r, c = np.argwhere(m).mean(axis=0)
            cv2.putText(img, name, (int(c * CELL) - 14, int(r * CELL) + 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, COLOURS[i], 1, cv2.LINE_AA)
    return img


def side_panel(height, force, top):
    p = np.zeros((height, SIDE, 3), np.uint8)
    for i, (name, f) in enumerate(zip(FINGERS + ("palm",), force)):
        y = 30 + i * 34
        cv2.putText(p, f"{name:7s}{f:5.2f}", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOURS[i], 1, cv2.LINE_AA)
        cv2.rectangle(p, (8, y + 6), (8 + int(170 * min(f / top, 1.0)), y + 14), COLOURS[i], -1)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glove-regions", type=Path, default=Path("config/glove_regions.yaml"))
    ap.add_argument("--glove-noise", type=Path, default=Path("config/glove_noise.npz"))
    ap.add_argument("--glove-python", type=Path, default=GLOVE_PYTHON)
    args = ap.parse_args()

    regions = FingerRegions.load(args.glove_regions)
    floor = load_floor(args.glove_noise)
    glove = GloveClient(python=args.glove_python).start()
    print(f"glove {glove.info.get('serial')}; noise floor {'from ' + str(args.glove_noise) if floor is not None else 'none'}")
    filt = ContactFilter(rate_hz=float(glove.info.get("rate_hz", 117.0)), floor=floor)
    flip, mirror, turns, full, raw = True, False, 0, 0.3, False
    rest, seq, lit, d, force = [], None, None, None, np.zeros(6)
    cv2.namedWindow("glove", cv2.WINDOW_NORMAL)
    try:
        while not glove.lost:
            f = glove.latest()
            if f is None:
                cv2.waitKey(20)
                continue
            if f.sequence != seq:
                seq = f.sequence
                if rest is not None:
                    rest.append(f.pressure)
                    if len(rest) >= 234:
                        filt.calibrate(rest)
                        rest = None
                else:
                    lit, d = filt.update(f.pressure)
                    force = regions.forces(d, lit)
            valid = ~np.isnan(f.pressure)
            if rest is not None or raw or d is None:
                img = picture(f.pressure, valid, None, regions, 1.0, flip, mirror, turns)
                head = "learning the rest: touch nothing" if rest is not None else "raw pressure"
            else:
                img = picture(d, valid, lit, regions, full, flip, mirror, turns)
                head = f"{glove.rate_hz:3.0f} Hz   contacts above the noise floor   full scale {full:.2f}"
            bar = np.zeros((BAR, img.shape[1], 3), np.uint8)
            cv2.putText(bar, head, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(bar, "b rest  v view  + - colours  f m r orient  q quit", (8, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
            img = np.vstack([bar, img])
            cv2.imshow("glove", np.hstack([img, side_panel(img.shape[0], force, max(1.0, full * 20))]))
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q") or cv2.getWindowProperty("glove", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("b"):
                rest, lit, d, force = [], None, None, np.zeros(6)
            elif key == ord("v"):
                raw = not raw
            elif key in (ord("+"), ord("=")):
                full = max(full / 2, 0.02)
            elif key == ord("-"):
                full = min(full * 2, 1.0)
            elif key == ord("f"):
                flip = not flip
            elif key == ord("m"):
                mirror = not mirror
            elif key == ord("r"):
                turns = (turns + 1) % 4
        if glove.lost:
            print("the glove server stopped")
    finally:
        glove.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

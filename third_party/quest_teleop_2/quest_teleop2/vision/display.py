"""Drawing helpers shared by the live viewer and the replay viewer.

These live here rather than in a script because two callers need them and a
second copy would drift: `scripts/camera/camera_live.py` draws camera frames while
aiming a mount, and `scripts/replay_demo.py` draws frames decoded back out of a
dataset while the robot re-executes the episode. Same panels, same tiling.

Everything here is DISPLAY ONLY. What is captured, recorded and fed to the
policy stays at the camera's native size and colour order -- `overlay` takes RGB
(what `Camera.latest()` returns) and hands back BGR (what `cv2.imshow` wants),
so a caller holding JPEG-decoded BGR must convert before calling.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: Dead space between tiled panels. Without it two dark scenes read as one
#: image with a seam somewhere in the middle.
GAP = 6

#: Pixels a target must stay inside every frame edge. Not a hard requirement --
#: a consistently clipped object is still consistently recognisable -- but
#: something touching the edge is one bump away from being cut differently in
#: every frame, which is the variance that actually hurts.
MARGIN = 30

Colour = tuple[int, int, int]  # BGR, as OpenCV wants


def sharpness(frame: np.ndarray) -> float:
    """Variance of the Laplacian. Higher is sharper; the scale is arbitrary.

    Focusing by eye on a laptop screen is unreliable, so this is what the live
    viewer actually optimises. Only comparable within one scene: pointing the
    camera somewhere else changes the number for reasons that are not focus.
    """
    import cv2  # noqa: PLC0415

    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def draw_marks(img: np.ndarray, marks: dict[int, tuple[int, int]], scale: float,
               margin: int = MARGIN) -> None:
    """Draw the safe margin box and every stamped position, in place.

    `marks` are in NATIVE frame pixels; `scale` maps them onto the already
    magnified image, so a caller never has to pre-scale its own coordinates.
    """
    import cv2  # noqa: PLC0415

    h, w = img.shape[:2]
    m = int(margin * scale)
    cv2.rectangle(img, (m, m), (w - m, h - m), (0, 140, 255), 1)
    for idx, (x, y) in sorted(marks.items()):
        cx, cy = int(x * scale), int(y * scale)
        inside = m <= cx <= w - m and m <= cy <= h - m
        colour = (80, 255, 80) if inside else (60, 60, 255)
        cv2.drawMarker(img, (cx, cy), colour, cv2.MARKER_CROSS, int(18 * scale), 2)
        cv2.putText(img, str(idx), (cx + int(10 * scale), cy - int(8 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, colour,
                    max(1, int(scale)), cv2.LINE_AA)


def closest_pair(marks: dict[int, tuple[int, int]]) -> tuple[float, str]:
    """Smallest pixel gap between stamped marks, and which pair it was.

    Two marks that land close together in the IMAGE are two marks a vision
    policy will confuse, however far apart they are on the table.
    """
    pts = sorted(marks.items())
    best, which = float("inf"), ""
    for i, (ai, a) in enumerate(pts):
        for bi, b in pts[i + 1:]:
            d = float(np.hypot(a[0] - b[0], a[1] - b[1]))
            if d < best:
                best, which = d, f"{ai}-{bi}"
    return best, which


def overlay(frame: np.ndarray, lines: Sequence[tuple[str, Colour]] = (),
            scale: float = 1.0, grid: int = 0,
            marks: dict[int, tuple[int, int]] | None = None,
            margin: int = MARGIN, border: Colour | None = None) -> np.ndarray:
    """RGB frame -> BGR panel, magnified and annotated.

    Upscale first, THEN draw. Drawing at native size and scaling afterwards
    would blur the text along with the image, which is the whole reason the
    overlay exists.
    """
    import cv2  # noqa: PLC0415

    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if grid:
        h, w = img.shape[:2]
        for i in range(1, grid):
            cv2.line(img, (w * i // grid, 0), (w * i // grid, h), (60, 60, 60), 1)
            cv2.line(img, (0, h * i // grid), (w, h * i // grid), (60, 60, 60), 1)
    if marks:
        draw_marks(img, marks, scale, margin)
    if lines:
        _text_panel(img, lines, scale)
    if border is not None:
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), border, 3)
    return img


def _text_panel(img: np.ndarray, lines: Sequence[tuple[str, Colour]], scale: float) -> None:
    """A darkened box behind the text, in place.

    Darkened rather than outlined: a stroke around a small font is thicker than
    the glyphs themselves and renders as mush. Width follows the longest line
    instead of a constant, so adding a longer readout cannot silently push text
    off the edge of its own backdrop.
    """
    import cv2  # noqa: PLC0415

    font_face, font = cv2.FONT_HERSHEY_SIMPLEX, 0.45 * max(scale, 1.0)
    thick = max(1, int(scale))
    pad, line_h = int(6 * scale), int(20 * scale)
    widest = max(cv2.getTextSize(t, font_face, font, thick)[0][0] for t, _ in lines)
    panel_w = min(img.shape[1], widest + pad * 3)
    panel_h = min(img.shape[0], pad * 2 + line_h * len(lines))
    box = img[0:panel_h, 0:panel_w]
    img[0:panel_h, 0:panel_w] = (box * 0.25).astype(img.dtype)
    y = pad + int(14 * scale)
    for text, colour in lines:
        cv2.putText(img, text, (int(8 * scale), y), font_face, font, colour,
                    thick, cv2.LINE_AA)
        y += line_h


def tile(panels: Sequence[np.ndarray | None], gap: int = GAP) -> np.ndarray | None:
    """Lay BGR panels out left to right. None becomes a 'no frame' placeholder.

    A missing camera has to stay VISIBLE and keep its position in the row --
    dropping it would silently renumber the panels, and the whole point of a
    fixed rig order is that panel 0 is always the same camera.
    """
    import cv2  # noqa: PLC0415

    present = [p for p in panels if p is not None]
    if not present:
        return None
    h = max(p.shape[0] for p in present)
    w = max(p.shape[1] for p in present)

    cols: list[np.ndarray] = []
    for panel in panels:
        if panel is None:
            panel = np.zeros((h, w, 3), np.uint8)
            cv2.putText(panel, "no frame", (int(w * 0.05), h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 255), 2, cv2.LINE_AA)
        if panel.shape[0] < h:
            pad = np.zeros((h - panel.shape[0], panel.shape[1], 3), panel.dtype)
            panel = np.vstack([panel, pad])
        if cols:
            cols.append(np.full((h, gap, 3), 40, np.uint8))
        cols.append(panel)
    return np.hstack(cols)

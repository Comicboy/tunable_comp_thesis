"""Find an object on the table by what changed, not by what it looks like.

`find_cube` in check_scene.py segments on "bright and desaturated" and scores by
how much darker the surround is. Both assumptions are about the scene rather
than the object: it needs a pale object, and it needs the dark box under it. A
coloured object is invisible to it, and moving a white one off the box collapses
the score until some reflection outscores it -- at which point it reports the
reflection, because it has no floor and always returns its best candidate.

This compares the frame against a picture of the empty table instead. Anything
that differs is a candidate, whatever colour it is and wherever it sits, so a
new object needs no code change and no threshold tuning.

The cost is a calibration step: the background has to be recaptured whenever the
camera, the lighting or the table moves. That is a real constraint, but it turns
a silent failure into a loud one -- a stale background reports nothing rather
than reporting the wrong thing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)

DEFAULT_PATH = Path("config/background.png")

#: Lab distance above which a pixel counts as "not the table". Lab because a
#: fixed step there is closer to a fixed perceptual step than it is in RGB, so
#: one threshold works for a white object and a saturated one at once.
DEFAULT_THRESHOLD = 25.0

#: Weight on the lightness difference, relative to chroma. Below 1 because a
#: shadow changes L while leaving a and b nearly alone: halving L's contribution
#: suppresses shadows without losing a white object on a dark table, which is
#: almost pure L difference and still clears the threshold by a wide margin.
LIGHTNESS_WEIGHT = 0.5

#: Blob area bounds in pixels. The upper bound is deliberately generous -- the
#: arm sweeping through frame is excluded by capturing with it parked, not by
#: size -- and the lower one rejects sensor speckle that survives the open.
MIN_AREA_PX = 200
MAX_AREA_PX = 40000

#: Mean Lab distance inside a blob, below which it is not believed. A real
#: object clears this several times over; sensor noise and a slightly stale
#: background do not.
MIN_SCORE = 18.0


@dataclass
class BackgroundModel:
    """A picture of the empty table, and the difference test against it."""

    image: np.ndarray
    """(H, W, 3) uint8 RGB median of several frames of the bare table."""

    threshold: float = DEFAULT_THRESHOLD

    @classmethod
    def capture(cls, grab, n: int = 15, threshold: float = DEFAULT_THRESHOLD) -> BackgroundModel:
        """Build a background from `n` frames.

        Args:
            grab: Callable returning one (H, W, 3) uint8 RGB frame, or None.
            n: Frames to median together. The median rather than the mean so a
                single frame with a hand still in shot cannot bias it.
            threshold: Lab distance counting as a difference.

        Returns:
            A BackgroundModel.
        """
        frames = []
        while len(frames) < n:
            frame = grab()
            if frame is not None:
                frames.append(np.asarray(frame, dtype=np.uint8))
        stack = np.stack(frames).astype(np.float32)
        return cls(image=np.median(stack, axis=0).astype(np.uint8), threshold=threshold)

    def save(self, path: Path | str | None = None) -> Path:
        """Write the background as a PNG, so it can be looked at."""
        import cv2  # noqa: PLC0415

        p = Path(path) if path is not None else DEFAULT_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
        return p

    @classmethod
    def load(cls, path: Path | str | None = None,
             threshold: float = DEFAULT_THRESHOLD) -> BackgroundModel:
        """Read a saved background, with an error that says how to make one."""
        import cv2  # noqa: PLC0415

        p = Path(path) if path is not None else DEFAULT_PATH
        if not p.exists():
            raise FileNotFoundError(
                f"No background at {p}.\n"
                "  Clear the table, park the arm out of view, and run:\n"
                "    python scripts/camera/capture_background.py"
            )
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise ValueError(f"{p} is not a readable image.")
        return cls(image=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), threshold=threshold)

    def difference(self, frame: np.ndarray) -> np.ndarray:
        """(H, W) float32 Lab distance between `frame` and the empty table."""
        import cv2  # noqa: PLC0415

        frame = np.asarray(frame, dtype=np.uint8)
        if frame.shape != self.image.shape:
            raise ValueError(
                f"frame {frame.shape} does not match the background "
                f"{self.image.shape}; recapture the background for this camera."
            )
        a = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
        b = cv2.cvtColor(self.image, cv2.COLOR_RGB2LAB).astype(np.float32)
        d = a - b
        chroma = np.sqrt(d[:, :, 1] ** 2 + d[:, :, 2] ** 2)
        return np.sqrt(chroma ** 2 + (LIGHTNESS_WEIGHT * d[:, :, 0]) ** 2)


def find_object(
    frame: np.ndarray,
    background: BackgroundModel,
    min_area: int = MIN_AREA_PX,
    max_area: int = MAX_AREA_PX,
    min_score: float = MIN_SCORE,
) -> tuple[tuple[int, int, int, int] | None, float]:
    """Locate the largest thing on the table that is not the table.

    Args:
        frame: (H, W, 3) uint8 RGB, same shape as the background.
        background: A BackgroundModel of the empty table.
        min_area: Smallest blob in pixels, to reject speckle.
        max_area: Largest blob in pixels.
        min_score: Mean Lab distance below which the blob is not believed.

    Returns:
        ``((x, y, w, h), score)``, or ``(None, score)`` when nothing is found.
        The score is the mean Lab distance inside the blob, so it is a
        confidence: a real object scores several times ``min_score``.
    """
    import cv2  # noqa: PLC0415

    dist = background.difference(frame)
    mask = (dist > background.threshold).astype(np.uint8)
    # Open kills speckle, close fills the interior of an object whose middle
    # happens to match the table. Order matters: closing first would grow the
    # speckle before the open could remove it.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best, best_score = None, 0.0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (min_area <= area <= max_area):
            continue
        # Chosen by area rather than by score: the biggest thing that is not
        # the table IS the object. Scoring by difference magnitude would prefer
        # a small bright specular highlight over the object itself.
        score = float(dist[labels == i].mean())
        if best is None or area > best[4]:
            best, best_score = (int(x), int(y), int(w), int(h), int(area)), score

    if best is None:
        return None, 0.0
    if best_score < min_score:
        _log.warning(
            "Object rejected: mean difference %.1f below %.1f. The background "
            "is probably stale -- recapture it if the camera or lighting moved.",
            best_score, min_score,
        )
        return None, best_score
    return best[:4], best_score

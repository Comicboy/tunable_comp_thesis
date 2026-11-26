"""Contacts and per-finger forces from the glove's pressure grid.

The glove reads 0.05-0.2 on many cells with nothing touching them and its own noise is tiny,
so a contact is pressure above each cell's own rest: learned while nothing touches the glove,
smoothed briefly, switched on above max(6 noise, `on`) for `hold` frames and off below
max(3 noise, `off`). A lone cell must reach `strong`, a shift shared by most cells is not a
contact, and the rest follows slow drift while a cell is untouched.

With a `floor` (24, 32), learned by scripts/glove/glove_noise.py from the hand moving without
touching anything, a cell is in contact only above its floor, and only the excess counts as
force: below it the cell reads exactly zero.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

FINGERS = ("thumb", "index", "middle", "ring", "little")
ROWS, COLS = 24, 32


class ContactFilter:
    def __init__(self, rate_hz: float = 117.0, on: float = 0.02, off: float = 0.01,
                 k_on: float = 6.0, k_off: float = 3.0, alpha: float = 0.3, hold: int = 3,
                 tau_rest_s: float = 5.0, strong: float = 0.06, min_cells: int = 2,
                 floor: np.ndarray | None = None) -> None:
        self.on, self.off, self.k_on, self.k_off = on, off, k_on, k_off
        self.floor = np.zeros((ROWS, COLS)) if floor is None else np.nan_to_num(np.asarray(floor, float))
        self.alpha, self.hold, self.strong, self.min_cells = alpha, hold, strong, min_cells
        self.beta = 1.0 / (tau_rest_s * rate_hz)
        self.rest = None

    @property
    def ready(self) -> bool:
        return self.rest is not None

    def calibrate(self, frames) -> None:
        """Learn each cell's rest and noise from frames with nothing touching the glove."""
        X = np.stack(list(frames)).astype(np.float64)
        self.valid = ~np.isnan(X[0])
        X = np.nan_to_num(X)
        self.rest = np.median(X, axis=0)
        self.noise = np.maximum(X.std(axis=0), 0.0005)
        self.usable = self.valid & (self.rest < 0.9) & (self.noise < 0.02)
        self.smooth = self.rest.copy()
        self.touched = np.zeros(self.rest.shape, bool)
        self.count = np.zeros(self.rest.shape, int)

    def update(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(contact mask, pressure above rest and above the floor) for one frame, both (24, 32)."""
        x = np.nan_to_num(np.asarray(frame, dtype=np.float64))
        self.smooth += self.alpha * (x - self.smooth)
        d = self.smooth - self.rest
        if self.usable.any():
            d = d - np.median(d[self.usable])
        up = d > np.maximum(np.maximum(self.k_on * self.noise, self.on), self.floor)
        self.count = np.where(up, self.count + 1, 0)
        down = d < np.maximum(np.maximum(self.k_off * self.noise, self.off), self.floor)
        self.touched = ((self.touched & ~down) | (self.count >= self.hold)) & self.usable
        idle = ~self.touched & self.usable
        self.rest[idle] += self.beta * (self.smooth[idle] - self.rest[idle])
        n, labels, stats, _ = cv2.connectedComponentsWithStats(self.touched.astype(np.uint8), connectivity=8)
        keep = np.zeros(n, bool)
        for i in range(1, n):
            keep[i] = stats[i, cv2.CC_STAT_AREA] >= self.min_cells or d[labels == i].max() >= self.strong
        excess = np.clip(d - self.floor, 0.0, None)
        return keep[labels] & self.touched, np.where(self.valid, excess, np.nan)


def load_floor(path: str | Path | None) -> np.ndarray | None:
    """The per-cell floor scripts/glove/glove_noise.py wrote, or None without one."""
    if path is None or not Path(path).exists():
        return None
    floor = np.load(path)["floor"]
    if floor.shape != (ROWS, COLS):
        raise ValueError(f"{path}: floor has shape {floor.shape}, expected ({ROWS}, {COLS})")
    return floor


class FingerRegions:
    """Which glove cells belong to which finger (and the palm), from config/glove_regions.yaml."""

    def __init__(self, cells: dict[str, list]) -> None:
        unknown = set(cells) - set(FINGERS) - {"palm"}
        if unknown:
            raise ValueError(f"unknown region(s) {sorted(unknown)}; use {list(FINGERS)} and palm")
        self.names = list(FINGERS) + ["palm"]
        self.masks = np.zeros((len(self.names), ROWS, COLS), bool)
        for i, name in enumerate(self.names):
            for r, c in cells.get(name) or []:
                if not (0 <= r < ROWS and 0 <= c < COLS):
                    raise ValueError(f"{name}: cell ({r}, {c}) is outside the {ROWS} x {COLS} grid")
                self.masks[i, r, c] = True
        if (self.masks.sum(axis=0) > 1).any():
            raise ValueError("a glove cell belongs to more than one region")

    @classmethod
    def load(cls, path: str | Path) -> FingerRegions:
        return cls(yaml.safe_load(Path(path).read_text()) or {})

    def forces(self, d: np.ndarray, lit: np.ndarray) -> np.ndarray:
        """(6,) contact force per region, thumb..little then palm: pressure above rest summed
        over the region's cells in contact."""
        p = np.where(lit, np.clip(np.nan_to_num(d), 0.0, None), 0.0)
        return (self.masks * p).sum(axis=(1, 2))

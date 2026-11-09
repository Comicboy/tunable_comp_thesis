"""Shared look and statistics of the plotting scripts in this folder (plot_il.py, plot_scripted_lift.py).

Imported by the scripts next to it (their folder is on sys.path when they run). It is not part of the
tunable_comp_thesis package on purpose: importing that package registers the gym tasks and so needs
Isaac Lab, while plotting only needs numpy and matplotlib.

    from plot_style import INK, K_COLOR, suptitle, save, band, wilson, mcnemar, ratio_ci

Colours: stiffness is an ordered magnitude, so K uses one blue ramp (softer = lighter); categories
(fingers, outcomes, controllers) use the categorical slots in a fixed order. Text is always ink or
muted grey, never the series colour.
"""

import math
import os
import textwrap
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# --- colours ---------------------------------------------------------------------------------------
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE, PHASE_BAND = "#ffffff", "#f0efec"
K_COLOR = {150.0: "#5598e7", 500.0: "#184f95"}     # the two IL stiffnesses
K_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]  # more K values (scripted lift), light -> dark
RIGID = "#52514e"                                  # K = inf: dark grey, dashed
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FINGER_COLOR = CATEGORICAL[:5]
FINGER_MARKER = ["o", "s", "^", "D", "v"]
FINGERS = ["thumb", "index", "middle", "ring", "little"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 10.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "legend.frameon": False, "legend.fontsize": 9.5, "lines.linewidth": 1.8,
})


def k_color(k: float) -> str:
    """Colour of one of the IL stiffnesses (150 / 500 N/m); anything else ink."""
    return K_COLOR.get(float(k), INK)


def k_label(k: float) -> str:
    return "K = ∞ (rigid)" if np.isinf(k) else f"K = {k:g} N/m"


# anchors of the stiffness shade (log10 K -> colour): 150 and 500 N/m are exactly K_COLOR, so the
# scripted lift's K series and the two IL stiffnesses read as one scale
_K_ANCHORS = [(np.log10(20.0), "#cfe1f8"), (np.log10(150.0), K_COLOR[150.0]), (np.log10(500.0), K_COLOR[500.0]),
              (np.log10(3000.0), "#0b2a52")]


def k_shade(k: float) -> str:
    """Colour of any stiffness on the blue scale (softer = lighter); K = inf is RIGID (dark grey)."""
    if np.isinf(k):
        return RIGID
    x = float(np.clip(np.log10(max(k, 1e-3)), _K_ANCHORS[0][0], _K_ANCHORS[-1][0]))
    for (x0, c0), (x1, c1) in zip(_K_ANCHORS[:-1], _K_ANCHORS[1:]):
        if x <= x1:
            w = (x - x0) / (x1 - x0)
            a, b = matplotlib.colors.to_rgb(c0), matplotlib.colors.to_rgb(c1)
            return matplotlib.colors.to_hex([(1 - w) * u + w * v for u, v in zip(a, b)])
    return _K_ANCHORS[-1][1]


# --- figure helpers --------------------------------------------------------------------------------
def suptitle(fig, title: str, sub: str | None = None):
    """Bold title and a lighter, wrapped subtitle at the top of the figure.

    They are placed inside the figure and their height is remembered; layout() then keeps the axes
    below them. (Anchoring them above the figure's edge does not work with bbox_inches="tight": the
    crop moves figure-fraction positions, and the texts end up on the panels.)"""
    h = fig.get_figheight() * 72.0  # figure height [points]
    lines = textwrap.fill(sub, int(fig.get_figwidth() * 13)) if sub else ""
    n = lines.count("\n") + 1 if sub else 0
    pad, title_h, line_h = 6.0, 17.0, 13.5
    fig.text(0.01, 1.0 - pad / h, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    if sub:
        fig.text(0.01, 1.0 - (pad + title_h + 3.0) / h, lines, ha="left", va="top", fontsize=10, color=INK_2,
                 linespacing=1.25)
    fig._header_frac = (pad + title_h + 3.0 + n * line_h + 10.0) / h


def layout(fig, bottom: float = 0.0):
    """tight_layout below the title block of suptitle() and above `bottom` (room for a legend below)."""
    fig.tight_layout(rect=(0, bottom, 1, 1.0 - getattr(fig, "_header_frac", 0.0)))


def save(fig, out_dir: str, name: str):
    """PNG only."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


def quiet(fn, *a, **k):
    """Call fn without numpy's 'all-NaN slice' warnings (columns where no episode has data)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(*a, **k)


def aligned(eps, fn, key: str, lag: np.ndarray) -> list:
    """fn(e) of every episode (T, ...) sampled at e[key] + lag (NaN outside the episode): one row per episode."""
    rows = []
    for e in eps:
        y = np.asarray(fn(e))
        idx = e[key] + lag
        ok = (idx >= 0) & (idx < len(y))
        w = np.full((len(lag),) + y.shape[1:], np.nan)
        w[ok] = y[idx[ok]]
        rows.append(w)
    return rows


def band(rows) -> tuple:
    """Median, 25th and 75th percentile over the rows; NaN where fewer than max(3, n/4) rows have data."""
    m = np.array(rows)
    enough = (~np.isnan(m)).sum(0) >= max(3, len(rows) // 4)
    return tuple(np.where(enough, quiet(np.nanpercentile, m, p, 0), np.nan) for p in (50, 25, 75))


def legend_below(fig, ax, ncol: int, y: float = -0.04):
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y))


# --- statistics ------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """95 % Wilson interval of a success rate k / n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(c - h, 0.0), min(c + h, 1.0)


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple:
    """Exact two-sided McNemar test for paired booleans: (p, only a, only b)."""
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = n10 + n01
    if n == 0:
        return 1.0, n10, n01
    p = 2 * sum(math.comb(n, j) for j in range(min(n10, n01) + 1)) / 2 ** n
    return min(p, 1.0), n10, n01


def ratio_ci(lo: np.ndarray, hi: np.ndarray, rng, boot: int = 2000, min_median: float = 0.0) -> tuple:
    """median(hi) / median(lo) with a 95 % percentile bootstrap (resampled within each group).

    nan if either median is below min_median (e.g. a finger that is mostly not in contact)."""
    if min(np.median(lo), np.median(hi)) < min_median or np.median(lo) == 0:
        return float("nan"), float("nan"), float("nan")
    r = np.median(hi) / np.median(lo)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.array([np.median(rng.choice(hi, len(hi))) / np.median(rng.choice(lo, len(lo))) for _ in range(boot)])
    b[~np.isfinite(b)] = np.inf
    return float(r), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
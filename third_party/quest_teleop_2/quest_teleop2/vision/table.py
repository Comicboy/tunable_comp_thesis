"""Turning a cube seen from above into a place the arm can be sent.

`calibrate_table.py` fits the homography; this reads it back and applies it.

Why a homography rather than a calibrated camera: the cube sits on a table, so
it is confined to a plane, and the projective map between two planes is exactly
a homography -- 8 degrees of freedom, determined by four point pairs, needing no
intrinsics, no distortion model and no depth sensor. The price is that it is
only valid for things ON the table. A cube held in the air reprojects to
nonsense, which is why this is used once at the start of an attempt and never
during the grasp itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)

DEFAULT_PATH = Path("config/table.yaml")

#: Detector score below which a blob is not believed. `find_cube` returns the
#: best-scoring candidate with no floor of its own, so with the real object
#: absent it happily returns a reflection or a pale part of the robot -- which
#: then becomes an approach target. Measured over 43 genuine detections the
#: worst scored 53.9 (p5 56.0, median 82.8), so this sits well below anything
#: real while rejecting the weak matches a false positive produces.
MIN_DETECTOR_SCORE = 40.0

#: How far outside the calibration points a detection is still trusted, metres.
#: The homography is only fitted where the points are; past them it
#: extrapolates, and a projective map extrapolates badly. Objects legitimately
#: sit a little beyond the fitted span -- the recorded demos reach 7 mm past it
#: -- so this is slack for that, not room to grow the working area.
BOUNDS_MARGIN_M = 0.05


@dataclass
class TableMap:
    """Pixel -> table-plane position, in the robot's base frame."""

    homography: np.ndarray  # (3, 3)
    camera: str
    rms_error_m: float
    #: The attitude every scripted approach inherits: a full TCP pose captured
    #: with the hand held the way an operator grasps from. Only its orientation
    #: and z are used -- x and y come from the camera. None when the
    #: calibration predates template capture, which makes any approach
    #: impossible rather than merely worse, so callers should say so plainly.
    template_pose: np.ndarray | None = None
    #: Constant (x, y) shift applied to every approach, base frame, metres.
    #: The TCP is the grasp point but the wrist camera is mounted somewhere
    #: else on the hand, so centring the palm over the object does not centre
    #: the camera over it. Tuned by eye against the wrist view and kept here so
    #: recording and evaluation inherit the same aim point -- it is part of the
    #: intended target, not error, and is reported apart from the injected
    #: noise for that reason.
    #:
    #: Declared last on purpose: appending a field is safe, but inserting one
    #: silently reassigns every positional construction of this class.
    approach_offset: np.ndarray | None = None

    #: (2, 2) [[x_min, x_max], [y_min, y_max]] the calibration points span,
    #: plus BOUNDS_MARGIN_M. A detection outside this is refused rather than
    #: clamped: it is not a slightly-wrong position, it is a different object.
    #: None for a calibration written before these were recorded, which
    #: disables the check rather than inventing a box.
    bounds: np.ndarray | None = None

    @classmethod
    def load(cls, path: Path | str | None = None) -> TableMap:
        import yaml  # noqa: PLC0415

        p = Path(path) if path is not None else DEFAULT_PATH
        if not p.exists():
            raise FileNotFoundError(
                f"No table calibration at {p}.\n"
                "  Run: python scripts/camera/calibrate_table.py"
            )
        d = yaml.safe_load(p.read_text())
        tmpl = d.get("template_pose")
        off = d.get("approach_offset")
        pts = [q["table"] for q in (d.get("points") or []) if "table" in q]
        bounds = None
        if pts:
            arr = np.asarray(pts, dtype=np.float64)
            bounds = np.stack([arr.min(axis=0) - BOUNDS_MARGIN_M,
                               arr.max(axis=0) + BOUNDS_MARGIN_M], axis=1)
        return cls(
            homography=np.asarray(d["homography"], dtype=np.float64),
            camera=str(d.get("camera", "static")),
            rms_error_m=float(d.get("rms_error_m", float("nan"))),
            template_pose=(None if tmpl is None
                           else np.asarray(tmpl, dtype=np.float64).reshape(6)),
            approach_offset=(np.zeros(2) if off is None
                             else np.asarray(off, dtype=np.float64).reshape(2)),
            bounds=bounds,
        )

    def require_template(self) -> np.ndarray:
        """The approach template, or an error that says how to make one."""
        if self.template_pose is None:
            raise ValueError(
                "This calibration has no template_pose, so there is no approach "
                "attitude to move to.\n"
                "  Re-run: python scripts/camera/calibrate_table.py\n"
                "  and press p with the hand held the way you grasp."
            )
        return self.template_pose

    def aim(self, cube_xy: np.ndarray) -> np.ndarray:
        """Where to send the TCP for a cube at `cube_xy`: the object plus the
        camera offset. Kept separate from the injected approach noise so the
        two never get confused in a report."""
        off = np.zeros(2) if self.approach_offset is None else self.approach_offset
        return np.asarray(cube_xy, dtype=np.float64).reshape(2) + off

    def to_table(self, pixel: tuple[float, float] | np.ndarray) -> np.ndarray:
        """One (u, v) pixel -> (x, y) metres in the base frame."""
        u, v = (float(x) for x in np.asarray(pixel, dtype=np.float64).reshape(2))
        p = self.homography @ np.array([u, v, 1.0])
        if abs(p[2]) < 1e-12:
            raise ValueError(f"pixel {(u, v)} maps to infinity; is it inside the table?")
        return np.array([p[0] / p[2], p[1] / p[2]])


#: Cached background, and whether we have already tried to load one. Loading is
#: attempted once per process: retrying per frame would re-stat the file at the
#: control rate and log the same failure every tick.
_BACKGROUND = None
_BACKGROUND_TRIED = False

#: A locator that replaces both detectors, e.g. SAM 3. See set_locator.
_LOCATOR = None


def set_locator(locator) -> None:
    """Answer locate_cube with `locator.locate(frame, table)`; None restores the detectors."""
    global _LOCATOR
    _LOCATOR = locator


def active_locator():
    return _LOCATOR


def use_background(background) -> None:
    """Override the background model, or pass None to force the legacy path."""
    global _BACKGROUND, _BACKGROUND_TRIED
    _BACKGROUND, _BACKGROUND_TRIED = background, True


def detect_object(frame: np.ndarray) -> tuple[tuple[int, int, int, int] | None, float, float]:
    """Run whichever detector is available. Returns (box, score, floor).

    Prefers background subtraction: it depends on neither the object's colour
    nor what it is standing on, and its shape filters are an area range rather
    than the squareness test that drops a cube yawed past 27.5 degrees. Falls
    back to the legacy brightness detector when no background has been
    captured, so an existing cell keeps working untouched.
    """
    global _BACKGROUND, _BACKGROUND_TRIED

    if not _BACKGROUND_TRIED:
        _BACKGROUND_TRIED = True
        from .detect import BackgroundModel  # noqa: PLC0415

        try:
            _BACKGROUND = BackgroundModel.load()
            _log.info("Using background subtraction for object detection.")
        except (FileNotFoundError, ValueError) as exc:
            _log.info("No background model (%s); using the brightness detector. "
                      "Run scripts/camera/capture_background.py for the better one.",
                      type(exc).__name__)

    if _BACKGROUND is not None:
        from .detect import MIN_SCORE, find_object  # noqa: PLC0415

        try:
            box, score = find_object(frame, _BACKGROUND)
            return box, score, MIN_SCORE
        except ValueError as exc:
            # Shape mismatch: the background belongs to a different camera or a
            # different resolution. Falling back silently would hide that.
            _log.error("Background unusable (%s); falling back.", exc)
            _BACKGROUND = None

    from scripts.camera.check_scene import find_cube  # noqa: PLC0415

    box, score = find_cube(frame)
    return box, score, MIN_DETECTOR_SCORE


def set_background(model) -> None:
    """Pin the detector's background, bypassing the one-time load from disk.

    Exists so a caller is not at the mercy of whether `config/background.png`
    happens to be present: a test that silently changes detector depending on
    the working tree tests whichever one it got. Pass None to force the
    brightness path.
    """
    global _BACKGROUND, _BACKGROUND_TRIED

    _BACKGROUND, _BACKGROUND_TRIED = model, True


def active_detector() -> str:
    """Which detector `locate_cube` is currently using.

    Recorded per episode, because the two disagree about where an object's
    centre is: the brightness path segments only the pale object, while
    background subtraction sees the object and whatever travels under it as one
    blob. A dataset that does not say which produced a position cannot later be
    checked for the seam.
    """
    if _LOCATOR is not None:
        return _LOCATOR.name
    return "background" if _BACKGROUND is not None else "brightness"


def locate_cube(frame: np.ndarray, table: TableMap) -> tuple[np.ndarray, float] | None:
    """Find the cube in a top-down frame and return its (x, y) and detector score.

    None when nothing is found -- the arm blocking the view is the usual cause,
    so callers should park it clear before asking.
    """
    if _LOCATOR is not None:
        return _LOCATOR.locate(frame, table)
    box, score, floor = detect_object(frame)
    if box is None:
        return None
    if score < floor:
        _log.warning(
            "Detection rejected: score %.1f below %.1f. The object is probably "
            "not in view and this is a reflection or the robot.",
            score, floor,
        )
        return None
    x, y, w, h = box
    xy = table.to_table((x + w / 2.0, y + h / 2.0))
    if table.bounds is not None and (
        np.any(xy < table.bounds[:, 0]) or np.any(xy > table.bounds[:, 1])
    ):
        _log.warning(
            "Detection rejected: (%.3f, %.3f) is outside the calibrated table "
            "region (x%s y%s). The homography extrapolates past its points, so "
            "this position is not trustworthy.",
            xy[0], xy[1],
            np.round(table.bounds[0], 3).tolist(),
            np.round(table.bounds[1], 3).tolist(),
        )
        return None
    return xy, score


@dataclass
class Approach:
    """A planned pre-grasp: where the object is, and where to go instead."""

    target: np.ndarray   #: (6,) pose to move to, floored, noise included
    cube_xy: np.ndarray  #: (2,) where the camera saw the object
    aim_xy: np.ndarray   #: (2,) object + the camera offset, before noise
    noise: np.ndarray    #: (3,) the injected error, metres
    score: float         #: detector score, for spotting a marginal detection


def plan_approach(frame: np.ndarray, table: TableMap, template: np.ndarray,
                  standoff: float, noise_xy: float, noise_z: float,
                  rng: np.random.Generator, raise_m: float = 0.0) -> Approach | None:
    """Locate the object and choose a deliberately imperfect pre-grasp pose.

    None when nothing is found -- the arm blocking the top-down view is the
    usual cause, so callers should park it clear first.

    Shared by recording and deployment on purpose. The injected error is the
    experiment: a policy trained on approaches that arrived perfectly has never
    seen the states it will face, so the two paths have to sample it the same
    way or the rollout is not measuring what the demos taught.

    z error is one-sided UPWARD, and the target is floored at the taught grasp
    height. Symmetric error would put some approaches below the height an
    operator taught by hand, and that is the one direction with a table in it.

    `raise_m` lifts the approach for an object taller than the calibration cube,
    keeping the standoff above its top.
    """
    found = locate_cube(frame, table)
    if found is None:
        return None
    cube_xy, score = found
    noise = np.array([rng.uniform(-noise_xy, noise_xy),
                      rng.uniform(-noise_xy, noise_xy),
                      rng.uniform(0.0, noise_z)])
    aim = table.aim(cube_xy)
    grasp_z = float(np.asarray(template, dtype=np.float64).reshape(6)[2])
    target = pre_grasp_pose(aim, template, grasp_z + standoff + raise_m, noise)
    if target[2] < grasp_z:
        target = target.copy()
        target[2] = grasp_z
    return Approach(target=target, cube_xy=np.asarray(cube_xy, dtype=np.float64),
                    aim_xy=np.asarray(aim, dtype=np.float64), noise=noise,
                    score=float(score))


def shift_approach(plan: Approach, shift_xy, window: float) -> Approach:
    """The same approach aimed `shift_xy` further, arriving within +-window of the aim.

    The injected error already spreads the arrival over that window, so a shifted aim
    sends the hand nowhere it has not been arriving. `noise` becomes the arrival offset.
    """
    arrival = np.clip(plan.noise[:2] + np.asarray(shift_xy, dtype=np.float64), -window, window)
    target, noise = plan.target.copy(), plan.noise.copy()
    target[:2] += arrival - noise[:2]
    noise[:2] = arrival
    return Approach(target=target, cube_xy=plan.cube_xy, aim_xy=plan.aim_xy, noise=noise,
                    score=plan.score)


def cube_overlay(table: TableMap, label: str = ""):
    """A callable that draws `find_cube`'s verdict on an RGB frame.

    For presentation footage. It re-runs the detector on whatever frame the
    writer thread happened to take, so what is drawn is the same call the
    approach makes -- a separate opinion drawn over the picture would be worse
    than none, because it would look authoritative while being neither what the
    robot saw nor what it decided.
    """
    import cv2  # noqa: PLC0415

    from scripts.camera.check_scene import find_cube  # noqa: PLC0415

    def draw(frame: np.ndarray) -> np.ndarray:
        if _LOCATOR is not None:
            return _LOCATOR.draw(frame)
        out = np.ascontiguousarray(frame).copy()
        box, score = find_cube(frame)
        if box is None:
            cv2.putText(out, "no object", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 80, 80), 2)
            return out
        x, y, w, h = box
        cv2.rectangle(out, (x, y), (x + w, y + h), (80, 255, 80), 2)
        cv2.circle(out, (int(x + w / 2), int(y + h / 2)), 4, (80, 255, 80), -1)
        xy = table.to_table((x + w / 2.0, y + h / 2.0))
        cv2.putText(out, f"{xy[0]:+.3f}, {xy[1]:+.3f} m", (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 80), 2)
        cv2.putText(out, f"{label}score {score:.0f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 80), 2)
        return out

    return draw


def pre_grasp_pose(cube_xy: np.ndarray, template: np.ndarray, height: float,
                   noise: np.ndarray | None = None) -> np.ndarray:
    """A TCP pose above the cube, optionally displaced.

    `template` supplies the orientation and is normally the pose an operator
    grasps from, so the approach arrives in a familiar attitude rather than one
    derived from nothing.

    `noise` is the deliberate approach error. It is not a nuisance to be
    minimised: the grasp policy's job is to absorb it, and the size of it is the
    x-axis of the tolerance curve. Injecting it at demo time is what teaches the
    policy that the approach cannot be trusted.
    """
    pose = np.asarray(template, dtype=np.float64).copy().reshape(6)
    pose[0], pose[1] = float(cube_xy[0]), float(cube_xy[1])
    pose[2] = height
    if noise is not None:
        pose[:3] += np.asarray(noise, dtype=np.float64).reshape(3)
    return pose

"""Quest hand tracking -> Wuji Hand joint angles.

The retargeting itself is Wuji's own `wuji-retargeting` (NLOPT, vector or
adaptive-analytical optimiser, configs shipped for this exact hand). This module
is only the adapter: WebXR's 25 joints in, MediaPipe's 21 out, joint angles back.

Why the vendor's retargeter rather than the hand-rolled one in
`wujihandpy/wujihand_teleop/teleop.py`: that version drove the thumb from
three-point position bend angles, which conflate flexion with abduction, and its
`_thumb_abduction()` returned `arcsin(abs(...))` -- the `abs` made negative
abduction unreachable when the joint's range is [-0.166, +0.934]. Measured over
the vendor's own recording, `wuji-retargeting` exercises thumb J2 across ~90% of
its range instead.

Frame convention: none required. `Retargeter.retarget()` subtracts the wrist and
then estimates the palm frame from the keypoints themselves (an SVD plane fit
through wrist / index-MCP / middle-MCP), so it is translation- and
rotation-invariant. Raw WebXR world coordinates in metres go straight in; no
palm-frame construction is needed on this side.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)

#: WebXR's 25 joints include a metacarpal for every finger; MediaPipe's 21 has
#: one only for the thumb. Dropping index/middle/ring/pinky metacarpals (WebXR
#: 5, 10, 15, 20) leaves exactly MediaPipe's layout, in MediaPipe's order.
XR_TO_MEDIAPIPE: tuple[int, ...] = (
    0,                    # 0  wrist
    1, 2, 3, 4,           # 1-4   thumb:  metacarpal, proximal, distal, tip
    6, 7, 8, 9,           # 5-8   index:  MCP, PIP, DIP, tip
    11, 12, 13, 14,       # 9-12  middle
    16, 17, 18, 19,       # 13-16 ring
    21, 22, 23, 24,       # 17-20 pinky
)

N_MEDIAPIPE = len(XR_TO_MEDIAPIPE)
N_WUJI_JOINTS = 20  # 5 fingers x 4

#: Mirrors a right-hand solution to the left: only the four fingers'
#: abduction flips sign.
MIRROR_TO_LEFT = np.ones((5, 4))
MIRROR_TO_LEFT[1:, 1] = -1.0

#: Subtracted from each finger's MCP flexion so a relaxed hand reads level.
#: The thumb stays 0: its joint 0 is a saddle axis, not a knuckle.
#: Overridden by config/hand_calibration.yaml.
KNUCKLE_TRIM = np.zeros((5, 4))
KNUCKLE_TRIM[:, 0] = (0.0, -0.210, 0.0, 0.169, 0.287)

#: Thumb opposition remap, (input, output) knots in radians, interpolated.
#: The outer knots are fixed points at the joint's limits: the lower stops
#: ordinary motion clipping, the upper keeps adduction reaching. Only the
#: middle knot is a calibration, and hand_calibration.yaml overrides it.
THUMB_J1_KNOTS = np.array([
    # input   output
    [-0.166, -0.166],   # lower limit: fixed point, so nothing clips
    [ 0.582,  0.322],   # resting pose: brought into the palm plane
    [ 0.934,  0.934],   # upper limit: fixed point, so adduction still reaches
])

#: Fraction of the way from the resting-pose knot's input to the UPPER limit at
#: which the corrected output sits, derived from the measured v1 knots above:
#: (0.322 - -0.166) / (0.934 - -0.166) against (0.582 - -0.166) / (0.934 - -0.166).
#: Kept as a ratio so the shape of the correction survives a change of hand.
_THUMB_REST_IN = (0.582 - -0.166) / (0.934 - -0.166)
_THUMB_REST_OUT = (0.322 - -0.166) / (0.934 - -0.166)


#: Operator-specific corrections, if calibrated. See the file's own header.
CALIBRATION_PATH = Path(__file__).resolve().parents[2] / "config" / "hand_calibration.yaml"


def load_calibration(path: Path | str | None = None) -> dict | None:
    """Read `hand_calibration.yaml`, or None if it is absent.

    Absent is a supported state, not an error: the built-in defaults are what
    this ran on before the file existed, so deleting it restores them exactly.
    """
    p = Path(path) if path is not None else CALIBRATION_PATH
    if not p.exists():
        return None
    import yaml  # noqa: PLC0415 -- only needed when a calibration exists

    data = yaml.safe_load(p.read_text()) or {}
    trim = data.get("knuckle_trim")
    if trim is not None and len(trim) != 5:
        raise ValueError(f"{p}: knuckle_trim needs 5 values (thumb..pinky), got {len(trim)}")
    return data


def calibrated_trim(data: dict | None) -> np.ndarray:
    """`KNUCKLE_TRIM` with the calibrated MCP offsets substituted in."""
    trim = KNUCKLE_TRIM.copy()
    if data and data.get("knuckle_trim") is not None:
        trim[:, 0] = np.asarray(data["knuckle_trim"], dtype=np.float64)
    return trim


#: Bounds on a per-joint gain. Wide enough for any real correction, narrow
#: enough that a typo cannot turn a small tracked motion into a full-range one.
_GAIN_LIMITS = (0.2, 3.0)


def calibrated_gain(data: dict | None) -> np.ndarray:
    """(5, 4) per-joint travel multipliers, 1.0 where uncalibrated.

    Keyed by joint name. Applied about zero, after the trim, so the rest pose
    stays put and only the travel away from it scales.
    """
    gain = np.ones((5, 4))
    named = (data or {}).get("joint_gain") or {}
    index = {n: (f, j) for f, row in enumerate(HAND2_JOINT_NAMES)
             for j, n in enumerate(row)}
    for name, value in named.items():
        if name not in index:
            raise ValueError(
                f"joint_gain: unknown joint {name!r}. Expected one of "
                f"{sorted(index)}"
            )
        v = float(value)
        if not _GAIN_LIMITS[0] <= v <= _GAIN_LIMITS[1]:
            raise ValueError(
                f"joint_gain[{name}] = {v} is outside {_GAIN_LIMITS}; a gain "
                "this far from 1.0 is a typo more often than an intention"
            )
        gain[index[name]] = v
    return gain


def thumb_j1_knots(lower: float, upper: float) -> np.ndarray:
    """`THUMB_J1_KNOTS` rescaled to the mounted hand's actual J1 range.

    The constants above were measured on the first-generation hand, whose
    thumb J1 runs [-0.166, +0.934]. Both outer knots are FIXED POINTS at those
    limits by design -- that is what stops the lower end clipping and keeps
    adduction reaching the fingers.

    Hardcoding them breaks on any other hand. The Wuji Hand 2's J1 is
    [-1.484, +0.698], so the v1 upper knot sits 0.236 rad (13.5 deg) PAST the
    limit: every output above 0.698 is clipped and the thumb saturates exactly
    where a pinch needs it. Anchoring the outer knots to the real limits
    restores the fixed-point property on whichever hand is fitted.

    The middle knot keeps its measured SHAPE (where the resting pose sits as a
    fraction of the range, and how far it is brought down), because that was a
    hardware measurement of the operator's flat hand. It should be re-measured
    on the Hand 2 -- the number that moves is where the resting thumb sits, not
    whether adduction reaches.
    """
    span = upper - lower
    return np.array([
        [lower, lower],
        [lower + _THUMB_REST_IN * span, lower + _THUMB_REST_OUT * span],
        [upper, upper],
    ])

_NA = -np.inf
HUMAN_MIN = np.array([
    # J0   J1     J2      J3
    [_NA, _NA, -0.100, -0.350],   # thumb: J3 rests hyperextended
    [_NA, _NA, -0.100, -0.250],   # index
    [_NA, _NA, -0.020, -0.250],   # middle: rested visibly hyperextended
    [_NA, _NA, -0.100, -0.250],   # ring
    [_NA, _NA, -0.100, -0.250],   # pinky
])

#: Shipped configs tuned for Apple Vision Pro input -- headset skeleton data,
#: which is what the Quest produces too, rather than camera-derived MediaPipe.
_CONFIGS = {
    "vector": "example/config/vector/vector_avp.yaml",
    "adaptive": "example/config/adaptive_analytical_avp.yaml",
}
_DEFAULT_REPO = Path(__file__).resolve().parents[2] / "wuji-retargeting"


def xr_to_mediapipe(joints: np.ndarray) -> np.ndarray:
    """(25, 3) WebXR joint positions -> (21, 3) MediaPipe-ordered keypoints."""
    arr = np.asarray(joints, dtype=np.float64)
    if arr.shape != (25, 3):
        raise ValueError(f"expected (25, 3) WebXR joints, got {arr.shape}")
    return arr[list(XR_TO_MEDIAPIPE)]


class HandRetargeter:
    """Wraps `wuji_retargeting.Retargeter` with the WebXR adapter and clamping.

    Import of `wuji_retargeting` is deferred to construction so that `--help`,
    the offline tests and anything that merely imports this module keep working
    without the retargeting stack installed.
    """

    def __init__(
        self,
        optimizer: str = "vector",
        hand_side: str = "right",
        mirror: bool = False,
        human_limits: bool = True,
        repo: Path | None = None,
        limits: tuple[np.ndarray, np.ndarray] | None = None,
        config: Path | str | None = None,
        calibration: Path | str | None = None,
    ) -> None:
        """`config` overrides the optimiser-keyed default.

        Used for the Wuji Hand 2, whose config points the optimiser at the
        hand2 URDF and supplies the `link_naming` block its anatomical link
        names need. The vendor ships no Hand 2 config for WebXR input, so that
        file lives in this repo rather than the vendored clone.
        """
        if config is None and optimizer not in _CONFIGS:
            raise ValueError(
                f"unknown optimizer {optimizer!r}; choose from {sorted(_CONFIGS)}"
            )
        repo = Path(repo) if repo is not None else _DEFAULT_REPO
        config = Path(config) if config is not None else repo / _CONFIGS[optimizer]
        if not config.exists():
            raise FileNotFoundError(
                f"retargeting config not found: {config}\n"
                "    git clone --recurse-submodules "
                "https://github.com/wuji-technology/wuji-retargeting.git"
            )

        from wuji_retargeting import Retargeter  # noqa: PLC0415 -- deferred on purpose

        self.optimizer = optimizer
        self.hand_side = hand_side
        self.mirror = mirror
        self.human_limits = human_limits
        self._retargeter = Retargeter.from_yaml(str(config), hand_side)
        self._perm = self._qpos_perm(hand_side, config)
        self._lower, self._upper = limits if limits is not None else (None, None)
        # Resolved once; a file read per frame would stall the control loop.
        self.calibration = load_calibration(calibration)
        self._trim = calibrated_trim(self.calibration)
        self._gain = calibrated_gain(self.calibration)
        self._knots = (THUMB_J1_KNOTS if self._lower is None
                       else thumb_j1_knots(self._lower[0, 1], self._upper[0, 1]))
        rest = (self.calibration or {}).get("thumb_j1", {}).get("rest")
        if rest is not None:
            # Only the middle knot moves; the outer two are the joint limits.
            self._knots = self._knots.copy()
            self._knots[1] = (float(rest["input"]), float(rest["output"]))
            if not (self._knots[0, 0] < self._knots[1, 0] < self._knots[2, 0]):
                raise ValueError(
                    f"thumb_j1.rest.input {self._knots[1, 0]:+.3f} must lie between the "
                    f"limits {self._knots[0, 0]:+.3f} and {self._knots[2, 0]:+.3f}; "
                    "np.interp needs increasing x and would silently flatten the remap"
                )
        if self.calibration is not None:
            tuned = {n: float(self._gain[f, j])
                     for f, row in enumerate(HAND2_JOINT_NAMES)
                     for j, n in enumerate(row) if self._gain[f, j] != 1.0}
            _log.info("hand calibration: trim %s, thumb rest %+.3f -> %+.3f rad%s",
                      np.round(self._trim[:, 0], 3).tolist(),
                      self._knots[1, 0], self._knots[1, 1],
                      f", gain {tuned}" if tuned else "")
        self.clamped_frames = 0
        _log.info("HandRetargeter: %s optimiser, %s hand%s, config %s",
                  optimizer, hand_side,
                  " (mirrored to LEFT)" if mirror else "", config.name)

    def _qpos_perm(self, hand_side: str, config: Path) -> np.ndarray:
        """Index array mapping the optimiser's qpos order onto thumb..pinky.

        `retarget()` returns qpos in Pinocchio's order -- index, middle, pinky,
        ring, thumb -- not the device's. Uses the vendor's own helpers, and
        like them refuses rather than moving the wrong joints when names
        cannot align.
        """
        names = list(self._retargeter.optimizer.robot.dof_joint_names)
        # The vendor's helpers, so there is one implementation.
        helpers = Path(_DEFAULT_REPO) / "example"
        if str(helpers) not in sys.path:
            sys.path.insert(0, str(helpers))
        from utils.config_paths import (  # noqa: PLC0415
            mjcf_joint_order, qpos_reorder_perm, resolve_mjcf_path)

        mjcf_path = resolve_mjcf_path(Path(config))
        perm = qpos_reorder_perm(names, mjcf_joint_order(mjcf_path))
        if mjcf_path is not None and perm is None:
            # A declared mjcf_path means the remap is required, and None here
            # is indistinguishable from "no remap needed".
            raise RuntimeError(
                "config declares optimizer.mjcf_path but the URDF<->MJCF joint "
                "names could not be aligned; refusing to retarget rather than "
                f"move the wrong joints.\n    config: {config}\n    mjcf: {mjcf_path}"
            )
        if perm is None:
            perm = np.arange(N_WUJI_JOINTS)
        perm = np.asarray(perm, dtype=int)

        # Everything downstream indexes rows positionally as thumb..pinky.
        prefix = "r_" if hand_side == "right" else "l_"
        wanted = [f"{prefix}{n}" for row in HAND2_JOINT_NAMES for n in row]
        got = [names[i] for i in perm]
        if got != wanted:
            raise RuntimeError(
                "the device joint order is not thumb..pinky, which every clamp "
                "and limit in this module assumes.\n"
                f"    device order: {got}\n    expected:     {wanted}"
            )
        if len(set(perm.tolist())) != N_WUJI_JOINTS:
            raise RuntimeError(f"joint permutation is not a bijection: {perm.tolist()}")
        if not np.array_equal(perm, np.arange(N_WUJI_JOINTS)):
            _log.info("qpos remap active: optimiser order -> thumb..pinky %s",
                      perm.tolist())
        return perm

    def __call__(self, xr_joints: np.ndarray) -> np.ndarray:
        """(25, 3) WebXR positions -> (5, 4) Wuji joint targets in radians."""
        kp = xr_to_mediapipe(xr_joints)
        qpos = np.asarray(self._retargeter.retarget(kp), dtype=np.float64)
        if qpos.shape != (N_WUJI_JOINTS,):
            raise RuntimeError(
                f"retargeter returned {qpos.shape}, expected ({N_WUJI_JOINTS},). "
                "The dataset action width and this adapter both assume 20 DoF."
            )
        # Into thumb..pinky order before anything is indexed by finger.
        target = qpos[self._perm].reshape(5, 4)
        if self.mirror:
            target = target * MIRROR_TO_LEFT
        if self.human_limits:
            target = target - self._trim
            target[0, 1] = np.interp(target[0, 1], self._knots[:, 0], self._knots[:, 1])
            # After the trim, so only travel away from rest scales.
            target = target * self._gain
            target = np.maximum(target, HUMAN_MIN)

        if not np.isfinite(target).all():
            raise RuntimeError("retargeter produced non-finite joint targets")

        if self._lower is not None:
            clipped = np.clip(target, self._lower, self._upper)
            # wujihandpy clamps internally too, but a clamp we can see is a clamp
            # we can diagnose -- silent saturation reads as "the mapping is bad".
            if not np.allclose(clipped, target):
                self.clamped_frames += 1
            target = clipped
        return target


#: Wuji Hand 2 names its joints anatomically (`r_thumb_cmc_flex`) where the
#: first-generation hand used positional names (`finger1_joint1`). Same 20 DoF
#: in the same (5, 4) order -- thumb..pinky, and within a finger
#: flexion, abduction, then the two distal joints -- so only the spelling
#: changes. Listed explicitly rather than trusting document order, because a
#: silently mis-ordered limits array would mis-scale every recorded action.
HAND2_JOINT_NAMES: tuple[tuple[str, ...], ...] = (
    ("thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip"),
    ("index_finger_mcp_flex", "index_finger_mcp_abd",
     "index_finger_pip", "index_finger_dip"),
    ("middle_finger_mcp_flex", "middle_finger_mcp_abd",
     "middle_finger_pip", "middle_finger_dip"),
    ("ring_finger_mcp_flex", "ring_finger_mcp_abd",
     "ring_finger_pip", "ring_finger_dip"),
    ("pinky_mcp_flex", "pinky_mcp_abd", "pinky_pip", "pinky_dip"),
)


def urdf_limits(urdf: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """(5, 4) lower and upper joint limits from a Wuji Hand URDF.

    Handles both generations. The first-generation URDF names joints
    `finger{1-5}_joint{1-4}` (optionally side-prefixed); the Hand 2 URDF uses
    anatomical names, mapped through `HAND2_JOINT_NAMES`.

    These limits define the 0..1 normalisation that every recorded action is
    stored in, so the two generations are NOT interchangeable: all 20 joints
    differ, by up to 1.32 rad. Converting old data means going through radians.
    """
    import xml.etree.ElementTree as ET  # noqa: PLC0415 -- only needed here

    root = ET.parse(urdf).getroot()
    found: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        found[joint.get("name") or ""] = (float(limit.get("lower")),
                                          float(limit.get("upper")))

    lower = np.zeros((5, 4))
    upper = np.zeros((5, 4))

    def fill(name_for) -> int:
        seen = 0
        for f in range(5):
            for j in range(4):
                for candidate in name_for(f, j):
                    if candidate in found:
                        lower[f, j], upper[f, j] = found[candidate]
                        seen += 1
                        break
        return seen

    # First generation: bare or side-prefixed positional names.
    seen = fill(lambda f, j: (f"finger{f + 1}_joint{j + 1}",
                              f"right_finger{f + 1}_joint{j + 1}",
                              f"left_finger{f + 1}_joint{j + 1}"))
    if seen != N_WUJI_JOINTS:
        # Hand 2: anatomical names, side-prefixed `r_` / `l_`.
        seen = fill(lambda f, j: (f"r_{HAND2_JOINT_NAMES[f][j]}",
                                  f"l_{HAND2_JOINT_NAMES[f][j]}",
                                  HAND2_JOINT_NAMES[f][j]))
    if seen != N_WUJI_JOINTS:
        raise ValueError(
            f"{urdf}: matched {seen} of {N_WUJI_JOINTS} finger joints under "
            "either the first-generation or the Hand 2 naming scheme"
        )
    return lower, upper

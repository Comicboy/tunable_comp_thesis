"""Wrist pose -> UR5e TCP target.

Ported from the relative-to-anchor scheme in the Isaac quest3_device.py
(`advance()`), which is the right design for this: the operator's hand and the
robot's TCP each have their own home captured at the moment the clutch engages,
and only the *displacement* between them is mapped. That is what makes the
clutch double as "reposition your own arm without moving the robot".

The one substantive change from the Isaac version is the output convention. UR
poses carry orientation as a rotation vector, so `Rotation.as_rotvec()` is
emitted directly and the wxyz/xyzw quaternion juggling disappears.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ..config import RetargetCfg
from ..transforms import (
    change_basis_rotvec,
    make_pose,
    pose_to_rotation,
    quat_xyzw_to_rotation,
    rotation_delta,
)


def _deadband(vector: np.ndarray, threshold: float) -> np.ndarray:
    """Shrink a vector toward zero by `threshold`, clamping at zero.

    Continuous at the boundary, unlike a hard "zero it if small" test -- which
    would make the arm twitch as the hand crosses in and out of the dead zone.
    """
    if threshold <= 0.0:
        return vector
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= threshold:
        return np.zeros_like(vector)
    return vector * ((magnitude - threshold) / magnitude)


class WristRetargeter:
    """Maps tracked-wrist motion onto TCP motion, anchored at clutch engagement."""

    def __init__(self, cfg: RetargetCfg) -> None:
        self._cfg = cfg
        self._rotation = cfg.rotation_matrix()  # XR -> robot base
        self._home_wrist_pos: np.ndarray | None = None
        self._home_wrist_rot: Rotation | None = None
        self._home_tcp_pos: np.ndarray | None = None
        self._home_tcp_rot: Rotation | None = None

    @property
    def anchored(self) -> bool:
        return self._home_wrist_pos is not None

    def anchor(self, wrist_pos: np.ndarray, wrist_quat: np.ndarray, tcp_pose: np.ndarray) -> None:
        """Capture both homes. Call on every clutch engagement and after any
        tracking dropout, so recovery resumes smoothly instead of snapping."""
        self._home_wrist_pos = np.asarray(wrist_pos, dtype=np.float64).copy()
        self._home_wrist_rot = quat_xyzw_to_rotation(wrist_quat)
        tcp = np.asarray(tcp_pose, dtype=np.float64)
        self._home_tcp_pos = tcp[:3].copy()
        self._home_tcp_rot = pose_to_rotation(tcp)

    def clear(self) -> None:
        self._home_wrist_pos = None
        self._home_wrist_rot = None
        self._home_tcp_pos = None
        self._home_tcp_rot = None

    def target(self, wrist_pos: np.ndarray, wrist_quat: np.ndarray) -> np.ndarray | None:
        """Absolute TCP target in the robot base frame, or None if not anchored."""
        if not self.anchored:
            return None
        assert self._home_wrist_rot is not None
        assert self._home_tcp_pos is not None
        assert self._home_tcp_rot is not None

        # Position: hand displacement, re-expressed in the base frame.
        delta = np.asarray(wrist_pos, dtype=np.float64) - self._home_wrist_pos
        delta = _deadband(delta, self._cfg.deadband_m)
        target_pos = self._home_tcp_pos + self._rotation @ delta * self._cfg.pos_scale

        # Orientation: the wrist's rotation since anchoring, re-expressed in the
        # base frame and applied to the home TCP orientation in the world frame
        # (left-multiplied), matching the Isaac implementation.
        wrist_rot = quat_xyzw_to_rotation(wrist_quat)
        delta_rot = rotation_delta(wrist_rot, self._home_wrist_rot)
        rotvec = _deadband(delta_rot.as_rotvec(), self._cfg.deadband_rad)
        base_delta = change_basis_rotvec(
            self._rotation, Rotation.from_rotvec(rotvec), self._cfg.rot_scale
        )
        target_rot = base_delta * self._home_tcp_rot

        return make_pose(target_pos, target_rot)

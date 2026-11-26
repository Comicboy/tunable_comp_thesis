"""Pose conventions and conversions.

UR poses are `[x, y, z, rx, ry, rz]`: position in metres, orientation as a
rotation vector (axis-angle). WebXR gives quaternions in xyzw order, which is
also scipy's order -- so unlike the Isaac pipeline there is no wxyz/xyzw
juggling anywhere in this codebase. Everything is xyzw at the boundary and a
rotation vector on the wire to the robot.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def pose_to_rotation(pose: np.ndarray) -> Rotation:
    """Orientation part of a UR pose as a scipy Rotation."""
    return Rotation.from_rotvec(np.asarray(pose, dtype=np.float64)[3:6])


def make_pose(position: np.ndarray, rotation: Rotation) -> np.ndarray:
    """Assemble a UR pose from a position and a scipy Rotation."""
    return np.concatenate([np.asarray(position, dtype=np.float64), rotation.as_rotvec()])


def quat_xyzw_to_rotation(quat: np.ndarray) -> Rotation:
    """WebXR / scipy quaternion (xyzw) -> Rotation."""
    return Rotation.from_quat(np.asarray(quat, dtype=np.float64))


def rotation_delta(current: Rotation, reference: Rotation) -> Rotation:
    """Rotation taking `reference` to `current`, in the world frame."""
    return current * reference.inv()


def change_basis_rotvec(matrix: np.ndarray, rotation: Rotation, scale: float = 1.0) -> Rotation:
    """Re-express a rotation in another frame, optionally scaling the angle.

    Applies the change of basis to the axis-angle *vector* rather than
    conjugating the matrix. For a proper rotation the two are equivalent, and
    doing it on the rotation vector makes the `scale` factor meaningful: it
    scales the angle while leaving the axis alone. Ported from the same trick in
    quest3_device.py.
    """
    return Rotation.from_rotvec(matrix @ rotation.as_rotvec() * scale)


def angle_between(a: Rotation, b: Rotation) -> float:
    """Magnitude of the rotation taking `b` to `a`, radians."""
    return float(np.linalg.norm((a * b.inv()).as_rotvec()))


def clamp_position(position: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, bool]:
    """Clamp a position into an axis-aligned box. Returns (clamped, was_clamped)."""
    pos = np.asarray(position, dtype=np.float64)
    out = np.clip(pos, box[:, 0], box[:, 1])
    return out, bool(np.any(out != pos))


def limit_step(
    desired: np.ndarray,
    previous: np.ndarray,
    max_linear: float,
    max_angular: float,
) -> tuple[np.ndarray, bool]:
    """Limit how far a pose may move from the previous one in a single tick.

    Position and orientation are limited independently, each scaled down to its
    own ceiling rather than being rejected -- so fast hand motion produces
    slower robot motion in the same direction, not a stall.
    """
    desired = np.asarray(desired, dtype=np.float64)
    previous = np.asarray(previous, dtype=np.float64)
    limited = desired.copy()
    hit = False

    delta_pos = desired[:3] - previous[:3]
    dist = float(np.linalg.norm(delta_pos))
    if dist > max_linear > 0.0:
        limited[:3] = previous[:3] + delta_pos * (max_linear / dist)
        hit = True

    r_prev = pose_to_rotation(previous)
    r_des = pose_to_rotation(desired)
    delta_rot = rotation_delta(r_des, r_prev).as_rotvec()
    angle = float(np.linalg.norm(delta_rot))
    if angle > max_angular > 0.0:
        capped = Rotation.from_rotvec(delta_rot * (max_angular / angle))
        limited[3:6] = (capped * r_prev).as_rotvec()
        hit = True

    return limited, hit

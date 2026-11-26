"""UR5e forward kinematics on the host, so joint targets can be checked against the workspace box.

Universal Robots' nominal parameters. With the TCP offset read from the pendant they put the
TCP within 1.7 mm of the controller's own reading over 27,000 recorded frames; asking the
controller instead costs 16 ms, two control cycles.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

A = np.array([0.0, -0.425, -0.3922, 0.0, 0.0, 0.0])
D = np.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996])
ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])
_CA, _SA = np.cos(ALPHA), np.sin(ALPHA)


def pose_matrix(pose) -> np.ndarray:
    """(4, 4) transform of a pose [x, y, z, rx, ry, rz]."""
    pose = np.asarray(pose, dtype=np.float64).reshape(6)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def flange(q) -> np.ndarray:
    """Base-to-flange transform: (4, 4) for q (6,), (n, 4, 4) for q (n, 6)."""
    q = np.asarray(q, dtype=np.float64)
    single = q.ndim == 1
    q = np.atleast_2d(q)
    T = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
    c, s = np.cos(q), np.sin(q)
    for i in range(6):
        M = np.zeros((len(q), 4, 4))
        M[:, 0, 0], M[:, 0, 1], M[:, 0, 2], M[:, 0, 3] = c[:, i], -s[:, i] * _CA[i], s[:, i] * _SA[i], A[i] * c[:, i]
        M[:, 1, 0], M[:, 1, 1], M[:, 1, 2], M[:, 1, 3] = s[:, i], c[:, i] * _CA[i], -c[:, i] * _SA[i], A[i] * s[:, i]
        M[:, 2, 1], M[:, 2, 2], M[:, 2, 3], M[:, 3, 3] = _SA[i], _CA[i], D[i], 1.0
        T = T @ M
    return T[0] if single else T


def tcp_position(q, tool: np.ndarray) -> np.ndarray:
    """TCP position: (3,) for q (6,), (n, 3) for q (n, 6). `tool` is the flange-to-TCP (4, 4)."""
    return (flange(q) @ tool)[..., :3, 3]


def box_excess(position, box: np.ndarray) -> np.ndarray:
    """How far a position lies outside the box (3, 2), metres; 0 inside. Elementwise over (n, 3)."""
    p = np.asarray(position, dtype=np.float64)
    out = np.maximum(np.maximum(box[:, 0] - p, p - box[:, 1]), 0.0)
    return out.max(axis=-1)

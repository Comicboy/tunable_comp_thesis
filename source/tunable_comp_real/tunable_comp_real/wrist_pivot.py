"""WristPivotRetargeter: the operator's wrist drives the robot hand's wrist, not the TCP.

The supervisor's WristRetargeter (quest_teleop2/teleop/retarget.py) maps the operator's wrist onto
the pendant TCP, relative to an anchor taken when the clutch engages:

    p*_TCP = p_TCP,0 + R_xb dp,        R*_TCP = dR R_TCP,0

so a pure wrist rotation turns the robot hand about the TCP. On this robot the TCP is 150 mm out
along the flange axis (inside the hand, around the palm), so the hand's base and the mount swing
around while the palm stays put -- not how the operator's own hand moves. The simulation had the
same issue with the grasp point and tracked the hand's wrist instead (teleop_lift.py, --ik_point
wrist; Balázs, 2026-11-26: same here).

This subclass keeps his law for the wrist point W of the robot hand and converts it to the TCP
pose the arm is commanded with. With d = the TCP's position relative to W, in the TCP frame (a
fixed vector, since W and the TCP are both fixed to the hand), p_TCP = p_W + R_TCP d, so

    p*_TCP = p_TCP,0 + R_xb dp + (R*_TCP - R_TCP,0) d,      R*_TCP unchanged.

Without rotation the extra term is zero (pure translation is exactly his); with d = 0 the class is
exactly his. W is placed on the flange axis at the hand's base, `wrist_offset` metres out from the
flange face (measured with a ruler); pivot_offset() turns that and the pendant's TCP offset into d.
Subclassed, not edited (option A', notes section 4).
"""

from __future__ import annotations

import numpy as np

from quest_teleop2.config import RetargetCfg
from quest_teleop2.teleop.retarget import WristRetargeter
from quest_teleop2.transforms import pose_to_rotation


def pivot_offset(tool: np.ndarray, wrist_offset: float) -> np.ndarray:
    """d: the TCP relative to the wrist point, in the TCP frame [m].

    tool: the pendant's TCP in the flange frame, (4, 4) (UR5eArm reads it at connect);
    wrist_offset: the hand's base, metres out from the flange face along the flange axis.
    d = R_tool^T (p_tool - p_W), p_W = (0, 0, wrist_offset)."""
    tool = np.asarray(tool, dtype=float)
    p_w = np.array([0.0, 0.0, float(wrist_offset)])
    return tool[:3, :3].T @ (tool[:3, 3] - p_w)


class WristPivotRetargeter(WristRetargeter):
    """His WristRetargeter, with wrist rotations pivoting about a point fixed to the hand."""

    def __init__(self, cfg: RetargetCfg, offset: np.ndarray | None = None) -> None:
        super().__init__(cfg)
        self._d = np.zeros(3)
        self._r0 = None
        if offset is not None:
            self.set_offset(offset)

    @property
    def offset(self) -> np.ndarray:
        """d, the TCP relative to the pivot in the TCP frame [m]; zeros = his behaviour."""
        return self._d.copy()

    def set_offset(self, offset: np.ndarray) -> None:
        """Set d. Only between engagements: a change while anchored would move the target."""
        if self.anchored:
            raise RuntimeError("set the pivot while the clutch is released (not anchored)")
        self._d = np.asarray(offset, dtype=float).reshape(3).copy()

    def anchor(self, wrist_pos, wrist_quat, tcp_pose) -> None:
        super().anchor(wrist_pos, wrist_quat, tcp_pose)
        self._r0 = pose_to_rotation(np.asarray(tcp_pose, dtype=float))

    def clear(self) -> None:
        super().clear()
        self._r0 = None

    def target(self, wrist_pos, wrist_quat) -> np.ndarray | None:
        pose = super().target(wrist_pos, wrist_quat)
        if pose is None or self._r0 is None or not self._d.any():
            return pose
        pose = np.asarray(pose, dtype=float).copy()
        pose[:3] += pose_to_rotation(pose).apply(self._d) - self._r0.apply(self._d)
        return pose
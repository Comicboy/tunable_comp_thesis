#!/usr/bin/env python3
"""Offline checks of tunable_comp_real.wrist_pivot (no robot, no headset).

    python scripts/real/test_wrist_pivot.py

The supervisor's WristRetargeter and his transforms are the real ones; the operator's wrist is
made up. Checks:
  1. pivot_offset: d = R_tool^T (p_tool - p_W) (his TCP 150 mm out, pivot a out -> (0, 0, 0.150 - a));
  2. offset 0 -> exactly his retargeter;
  3. pure translation -> exactly his (the extra term needs a rotation);
  4. pure wrist rotations about several axes -> the robot's wrist point stays where it was
     anchored, the TCP swings around it (his: the TCP stays), orientation identical to his;
  5. translation + rotation -> the wrist point moves exactly as his TCP would;
  6. the offset cannot change while anchored.
Prints PASS / FAIL per check; exit code 1 if any fails.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from quest_teleop2.config import RetargetCfg
from quest_teleop2.teleop.retarget import WristRetargeter
from quest_teleop2.transforms import make_pose, pose_to_rotation
from tunable_comp_real.wrist_pivot import WristPivotRetargeter, pivot_offset

failures = 0


def check(ok: bool, text: str) -> None:
    global failures
    failures += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}")


def wrist_point(pose: np.ndarray, d: np.ndarray) -> np.ndarray:
    """The pivot's position from a TCP pose: p_W = p_TCP - R_TCP d."""
    return pose[:3] - pose_to_rotation(pose).apply(d)


def main() -> int:
    cfg = RetargetCfg()
    tool = np.eye(4)
    tool[2, 3] = 0.150                                    # the pendant TCP on this robot
    a = 0.03
    d = pivot_offset(tool, a)
    print("1. pivot_offset")
    check(np.allclose(d, [0, 0, 0.120]), f"TCP 150 mm, pivot 30 mm out -> d = {np.round(1000 * d, 3)} mm")
    tool_r = np.eye(4)
    tool_r[:3, :3] = Rotation.from_rotvec([0, 0, np.pi / 2]).as_matrix()
    tool_r[:3, 3] = [0.01, 0.0, 0.15]
    d_r = pivot_offset(tool_r, a)
    check(np.allclose(tool_r[:3, :3] @ d_r, tool_r[:3, 3] - [0, 0, a]),
          f"rotated TCP: d in the TCP frame, R d = p_TCP - p_W ({np.round(1000 * d_r, 3)} mm)")

    # Robot at home: TCP pointing forward (+x of the base), as his home_q.
    tcp0 = make_pose(np.array([0.72, 0.10, 0.28]), Rotation.from_rotvec([0, np.pi / 2, 0]))
    w_pos0 = np.array([0.1, 1.2, -0.3])                  # operator's wrist in the XR frame
    w_rot0 = Rotation.from_rotvec([0.1, -0.2, 0.3])

    def pair(offset):
        his, ours = WristRetargeter(cfg), WristPivotRetargeter(cfg, offset)
        for r in (his, ours):
            r.anchor(w_pos0, w_rot0.as_quat(), tcp0)
        return his, ours

    rng = np.random.default_rng(0)
    print("2.-5. against his WristRetargeter")
    his, ours = pair(np.zeros(3))
    ok = True
    for _ in range(50):
        p = w_pos0 + rng.normal(0, 0.05, 3)
        q = (Rotation.from_rotvec(rng.normal(0, 0.4, 3)) * w_rot0).as_quat()
        ok &= np.allclose(his.target(p, q), ours.target(p, q), atol=1e-12)
    check(ok, "offset 0: identical to his for 50 random wrist poses")

    his, ours = pair(d)
    ok = True
    for _ in range(50):
        p = w_pos0 + rng.normal(0, 0.05, 3)
        ok &= np.allclose(his.target(p, w_rot0.as_quat()), ours.target(p, w_rot0.as_quat()), atol=1e-12)
    check(ok, "pure translation: identical to his")

    w0 = wrist_point(tcp0, d)
    still, swing, expect, orient = [], [], [], []
    r0 = pose_to_rotation(tcp0)
    for axis in np.eye(3):
        for angle in (-0.6, -0.3, 0.3, 0.6):
            q = (Rotation.from_rotvec(axis * angle) * w_rot0).as_quat()
            t_his, t_ours = his.target(w_pos0, q), ours.target(w_pos0, q)
            still.append(np.linalg.norm(wrist_point(t_ours, d) - w0))
            swing.append(np.linalg.norm(t_ours[:3] - tcp0[:3]))
            expect.append(np.linalg.norm(pose_to_rotation(t_his).apply(d) - r0.apply(d)))
            orient.append(np.linalg.norm((pose_to_rotation(t_ours) * pose_to_rotation(t_his).inv()).as_rotvec()))
    check(max(still) < 1e-12, f"pure rotations about 3 axes: the wrist point stays (max {max(still):.1e} m)")
    check(np.allclose(swing, expect, atol=1e-12) and max(swing) > 0.05,
          f"... while the TCP swings around it by |(R* - R0) d| (up to {1000 * max(swing):.0f} mm; 0 for a twist "
          "about the hand's own axis; his TCP stays)")
    check(max(orient) < 1e-12, "orientation identical to his")

    ok = True
    for _ in range(50):
        p = w_pos0 + rng.normal(0, 0.05, 3)
        q = (Rotation.from_rotvec(rng.normal(0, 0.4, 3)) * w_rot0).as_quat()
        t_his, t_ours = his.target(p, q), ours.target(p, q)
        ok &= np.allclose(wrist_point(t_ours, d) - w0, t_his[:3] - tcp0[:3], atol=1e-12)
    check(ok, "translation + rotation: the wrist point moves exactly as his TCP would")

    print("6. offset changes")
    try:
        ours.set_offset(np.zeros(3))
        check(False, "set_offset while anchored is refused")
    except RuntimeError:
        check(True, "set_offset while anchored is refused")
    ours.clear()
    ours.set_offset(np.zeros(3))
    check(np.allclose(ours.offset, 0), "after clear() (clutch released) it can change")

    print(f"\n{'all checks passed' if failures == 0 else f'{failures} check(s) FAILED'}")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
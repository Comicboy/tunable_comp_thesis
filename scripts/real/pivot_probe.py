#!/usr/bin/env python3
"""Which point of the robot stays still when the operator only rotates the wrist? Read-only.

    # terminal 1: any teleoperation (his teleop_all.py, or our record_real.py)
    # terminal 2:
    python scripts/real/pivot_probe.py --seconds 20

While it records, keep your wrist in place and only rotate your hand: flex up/down, side to side,
twist, each a few times (about +-30 deg). The arm follows; this script only listens (RTDE receive,
like the supervisor's preflight) and commands nothing.

For every sample k the flange pose (R_k, t_k) comes from the measured joints (his UR5e kinematics).
The pivot p_f (in the flange frame) and the point it stays at, c (base frame), solve
    R_k p_f + t_k = c   for all k            (least squares)
Printed: the pivot in the flange frame [mm] (z = distance out along the flange axis), how still
it stayed, and for comparison how far the flange and the pendant TCP moved. Rotations about at
least two different axes are needed; with one axis the pivot's position along it is undefined
(reported).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import rtde_receive
from scipy.spatial.transform import Rotation

from quest_teleop2.config import RobotCfg
from quest_teleop2.robot.kinematics import flange, pose_matrix


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--host", default=None, help="robot IP (default: his config/robot.yaml)")
    args = ap.parse_args()

    host = args.host or RobotCfg.load().connection.host
    r = rtde_receive.RTDEReceiveInterface(host)
    print(f"Recording {args.seconds:g} s: keep your wrist in place, only rotate your hand "
          "(flex, side to side, twist).")
    qs, tcps = [], []
    t_end = time.monotonic() + args.seconds
    while time.monotonic() < t_end:
        qs.append(r.getActualQ())
        tcps.append(r.getActualTCPPose())
        time.sleep(1.0 / args.hz)
    r.disconnect()

    F = flange(np.asarray(qs))                                # (n, 4, 4) base -> flange
    R, t = F[:, :3, :3], F[:, :3, 3]
    n = len(R)
    # Unknowns x = [p_f (3), c (3)]:  R_k p_f - c = -t_k
    A = np.zeros((3 * n, 6))
    A[:, :3] = R.reshape(3 * n, 3)
    A[:, 3:] = -np.tile(np.eye(3), (n, 1))
    x, *_ = np.linalg.lstsq(A, -t.reshape(-1), rcond=None)
    sv = np.linalg.svd(A, compute_uv=False)
    p_f, c = x[:3], x[3:]

    def spread(points: np.ndarray) -> float:
        """RMS distance from their mean, mm."""
        return 1000 * float(np.sqrt(np.mean(np.sum((points - points.mean(0)) ** 2, axis=1))))

    pivot_world = np.einsum("kij,j->ki", R, p_f) + t
    tool = np.linalg.inv(F[0]) @ pose_matrix(tcps[0])          # pendant TCP in the flange frame
    tcp_world = np.asarray(tcps)[:, :3]
    rot = Rotation.from_matrix(R)
    angles = np.degrees((rot * rot[0].inv()).magnitude())

    np.set_printoptions(precision=1, suppress=True)
    print(f"\n{n} samples; wrist rotated up to {angles.max():.0f} deg from the first sample")
    if angles.max() < 10:
        print("Too little rotation to locate a pivot; rotate more and run again.")
        return 1
    print(f"pivot in the flange frame [mm]: {1000 * p_f}   (z = distance out along the flange axis)")
    print(f"  how still the pivot stayed: {spread(pivot_world):.1f} mm RMS")
    print(f"  flange centre moved:        {spread(t):.1f} mm RMS")
    print(f"  pendant TCP ({1000 * tool[2, 3]:.0f} mm out) moved: {spread(tcp_world):.1f} mm RMS")
    if sv[-1] / sv[0] < 1e-3:
        print("  (rotation about one axis only: the pivot's position along that axis is undefined;"
              " rotate in more directions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
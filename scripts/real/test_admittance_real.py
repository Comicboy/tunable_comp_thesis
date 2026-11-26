#!/usr/bin/env python3
"""Offline checks of tunable_comp_real.admittance (no hand, no glove).

    python scripts/real/test_admittance_real.py

The counterpart of the simulation's scripts/tools/test_admittance.py, at the real control period
(8 ms, 125 Hz):
  1. step response of the laws alone: steady state e = f / K, settling time, no overshoot for the
     critically damped dynamic law, the 30 mm clamp, K = inf, the stability limit;
  2. the fingertip layer on the Hand 2 model: a constant pad force yields the fingertip along
     the pad normal by f / K, the DLS joint offset reproduces that offset, the flexion joints
     open, and a dropout (force 0) lets the offset decay smoothly back to zero.
Prints PASS / FAIL per check; exit code 1 if any fails.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from tunable_comp_real.admittance import AdmittanceParams, DynamicAdmittance, FingertipAdmittance, StaticAdmittance, dls
from tunable_comp_real.hand_model import FINGERS, HandModel

DT = 1.0 / 125.0
failures = 0


def check(ok: bool, text: str) -> None:
    global failures
    failures += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}")


def laws() -> None:
    print("1. Laws alone, force 1 N on one point, dt = 8 ms")
    p = AdmittanceParams()
    dyn = DynamicAdmittance(p, DT, num_points=1)
    check(abs(dyn.max_stable_stiffness - 0.3 * (0.83 / DT) ** 2) / dyn.max_stable_stiffness < 0.01,
          f"stability limit {dyn.max_stable_stiffness:.0f} N/m (= M (0.83 / dt)^2)")
    f = np.array([[0.0, -1.0, 0.0]])
    for K in (150.0, 500.0):
        dyn.reset()
        e = np.array([dyn.compute(f, K)[0, 1] for _ in range(int(2.0 / DT))])  # 2 s
        target = -1.0 / K
        tau = math.sqrt(p.mass / K)
        settle = np.argmax(np.abs(e - target) <= 0.05 * abs(target)) * DT
        overshoot = max(0.0, (target - e.min()) / abs(target))
        check(abs(e[-1] - target) / abs(target) < 1e-3, f"dynamic K={K:g}: steady state {e[-1] * 1000:.3f} mm "
              f"(f/K = {target * 1000:.3f} mm)")
        check(abs(settle - 4.7 * tau) < 3 * DT + 0.25 * 4.7 * tau,
              f"dynamic K={K:g}: 95 % settling {settle * 1000:.0f} ms (4.7 tau = {4.7 * tau * 1000:.0f} ms)")
        check(overshoot < 0.01, f"dynamic K={K:g}: overshoot {overshoot:.1%} (critically damped)")
    sta = StaticAdmittance(p, DT, num_points=1)
    e = sta.compute(f, 500.0)[0, 1]
    check(abs(e - (-1.0 / 500.0)) < 1e-12, f"static K=500: e = {e * 1000:.3f} mm at once (no extra filter)")
    dyn.reset()
    for _ in range(500):
        e = dyn.compute(np.array([[0.0, -50.0, 0.0]]), 150.0)
    check(abs(np.linalg.norm(e) - 0.03) < 1e-9, f"clamp: |e| = {np.linalg.norm(e) * 1000:.1f} mm at 50 N, K=150")
    dyn.reset()
    e = dyn.compute(f, math.inf)
    check(np.all(e == 0.0), "K = inf: no offset")


def fingertips(urdf: Path | None) -> None:
    print("2. Fingertip layer on the Hand 2 model (dynamic law, K = 500 N/m, contact point = distal COM)")
    hand = HandModel(urdf)
    adm = FingertipAdmittance(hand, AdmittanceParams(), DT)
    try:
        adm.set_stiffness(5000.0)
        check(False, "K above the stability limit refused")
    except ValueError:
        check(True, "K above the stability limit refused")
    adm.set_stiffness(500.0)
    # A mid-closed pose: flexion joints at 0.6 rad, abductions 0; reference = measured pose.
    q = np.zeros(20)
    for f in FINGERS:
        sl = hand.finger_slice(f)
        q[sl] = [0.6, 0.0, 0.6, 0.6] if f != "thumb" else [0.3, -0.3, 0.6, 0.6]
    F = np.zeros(5)
    F[1] = 1.0  # 1 N on the index pad
    for _ in range(int(2.0 / DT)):
        q_cmd = adm.step(q, q, F)
    e = adm.offset[1]
    n = adm.normal[1]
    check(abs(np.linalg.norm(e) - 1.0 / 500.0) / (1.0 / 500.0) < 1e-3 and float(e @ n) < 0,
          f"index offset |e| = {np.linalg.norm(e) * 1000:.3f} mm against the pad normal (f/K = 2.000 mm)")
    sl = hand.finger_slice("index")
    J = hand.jacobian(q, "index", hand.distal_com["index"])
    achieved = J @ (q_cmd[sl] - q[sl])
    err = np.linalg.norm(achieved - e) / np.linalg.norm(e)
    along = float(achieved @ e) / float(e @ e)
    print(f"  [info] with lambda = {adm.p.ik_damping}: the joint offset moves the contact point by "
          f"{along:.0%} of e along e (error {err:.1%}); singular values of J "
          f"{np.round(np.linalg.svd(J, compute_uv=False), 4)} m/rad")
    exact = dls(J, e, 1e-4)
    check(np.linalg.norm(J @ exact - e) / np.linalg.norm(e) < 0.01,
          "DLS implementation: with lambda -> 0 the joint offset reproduces e to < 1 %")
    dq = adm.joint_offset[sl]
    check(float(achieved @ n) < 0, f"index opens: its contact point moves against the pad normal "
          f"(dq = {np.round(dq * 1000, 1)} mrad)")
    others = np.abs(adm.joint_offset[[i for i in range(20) if not (sl.start <= i < sl.stop)]]).max()
    check(others == 0.0, "the other fingers do not move")
    for _ in range(int(1.0 / DT)):
        adm.step(q, q, np.zeros(5))  # glove dropout: force 0
    left = np.linalg.norm(adm.offset[1])
    check(left < 1e-5, f"dropout: offset decays to {left * 1e6:.2f} um within 1 s (no jump)")
    # Near the straight floor: the opening is clipped at 0 rad, the command never hyperextends.
    adm.set_stiffness(150.0)
    q_straight = np.zeros(20)
    for _ in range(int(1.0 / DT)):
        q_cmd = adm.step(q_straight, q_straight, F * 3)
    check(np.all(q_cmd >= hand.command_lower - 1e-12), "straight finger: command stays at the 0 rad floor")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", type=Path, default=None, help="Hand 2 URDF (default: the one in third_party)")
    args = ap.parse_args()
    laws()
    fingertips(args.urdf)
    print(f"\n{'all checks passed' if failures == 0 else f'{failures} check(s) FAILED'}")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
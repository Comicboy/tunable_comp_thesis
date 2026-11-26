#!/usr/bin/env python3
"""Check tunable_comp_real.hand_model on the Hand 2 URDF, without the hand.

    python scripts/real/check_hand_model.py --figure logs/real/hand_model.png

Prints the joint table (device order, URDF limits), the fingertip positions and pad normals at
the zero pose, and three checks:
  1. the Jacobian against finite differences of the forward kinematics, at random poses;
  2. flexing a finger moves its fingertip along its pad normal, i.e. towards the palm side;
  3. the pad normals at the zero pose point to the palm side (where the thumb base sits).
--figure draws the hand seen from the palm side at the open pose and half closed, with the pad
normals, for a visual check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tunable_comp_real.hand_model import FINGERS, PAD_AXIS, HandModel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", type=Path, default=None, help="Hand 2 URDF (default: the one in third_party)")
    ap.add_argument("--figure", type=Path, default=None, help="save a palm-side picture here (.png)")
    args = ap.parse_args()

    hand = HandModel(args.urdf)
    print(f"URDF: {hand.urdf}\n\nJoints (device order): URDF limits and the 0..1 command range [rad]:")
    for i, (n, lo, hi, clo) in enumerate(zip(hand.joint_names, hand.lower, hand.upper, hand.command_lower)):
        print(f"  {i:2d} {FINGERS[i // 4]:6s} j{i % 4 + 1}  {n:26s} URDF [{lo:+.3f}, {hi:+.3f}]  "
              f"command [{clo:+.3f}, {hi:+.3f}]")

    q0 = np.zeros(20)
    print("\nZero pose: fingertip position [mm] and pad normal (r_wrist frame)")
    for f, pose in hand.fk(q0).items():
        print(f"  {f:6s} tip {np.round(pose.tip_pos * 1000, 1)}  normal {np.round(pose.pad_normal, 3)}")

    # 1. Jacobian vs finite differences.
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(20):
        q = hand.lower + rng.uniform(0.1, 0.9, 20) * (hand.upper - hand.lower)
        for f in FINGERS:
            J = hand.jacobian(q, f, point=np.array([0.0, 0.005, 0.01]))
            sl = hand.finger_slice(f)
            for k in range(4):
                dq = np.zeros(20)
                dq[sl.start + k] = 1e-6
                p1 = hand.finger(q + dq, f)
                p0 = hand.finger(q - dq, f)
                pt = np.array([0.0, 0.005, 0.01])
                num = ((p1.tip_pos + p1.tip_R @ pt) - (p0.tip_pos + p0.tip_R @ pt)) / 2e-6
                worst = max(worst, float(np.abs(num - J[:, k]).max()))
    print(f"\n1. Jacobian vs finite differences, 20 random poses: largest difference {worst:.2e} m/rad "
          f"({'OK' if worst < 1e-6 else 'FAIL'})")

    # 2. Flexion moves the fingertip along its pad normal (flexion joints: 1, 3, 4).
    print("2. Fingertip motion per rad of flexion, projected on the pad normal (> 0 = towards the palm side):")
    for f in FINGERS:
        J = hand.jacobian(q0, f)
        n = hand.finger(q0, f).pad_normal
        proj = [float(n @ J[:, k]) for k in (0, 2, 3)]
        print(f"   {f:6s} j1 {proj[0]:+.3f}  j3 {proj[1]:+.3f}  j4 {proj[2]:+.3f} m/rad")

    # 3. Pad normals point to the palm side: the side of the palm plane where the thumb base lies.
    poses = hand.fk(q0)
    bases = np.array([poses[f].joint_pos[0] for f in ("index", "middle", "ring", "little")])
    centre = bases.mean(axis=0)
    # Palm plane through the four finger bases and the mean fingertip direction.
    along = np.mean([poses[f].tip_pos - poses[f].joint_pos[0] for f in ("index", "middle", "ring", "little")], axis=0)
    across = bases[0] - bases[-1]
    palm_n = np.cross(across, along)
    palm_n /= np.linalg.norm(palm_n)
    if palm_n @ (poses["thumb"].joint_pos[0] - centre) < 0:
        palm_n = -palm_n  # orient towards the thumb base side
    print(f"3. Palm-side direction (towards the thumb base) {np.round(palm_n, 3)}; "
          "pad normal . palm-side direction at the zero pose:")
    print("   " + "  ".join(f"{f} {poses[f].pad_normal @ palm_n:+.2f}" for f in FINGERS))
    print(f"   (PAD_AXIS = {PAD_AXIS} in the distal link frame)")

    if args.figure is not None:
        draw(hand, args.figure)
        print(f"\nwrote {args.figure}")
    return 0


def draw(hand: HandModel, path: Path) -> None:
    """The hand seen from the palm side (looking along -y of r_wrist), open and half closed. A right
    palm seen from the front has its thumb on the right: a check of the palm side = +y."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    half = hand.lower + 0.5 * (hand.upper - hand.lower)
    half[1::4] = 0.0  # abduction joints straight
    poses = {"open (all joints 0)": np.zeros(20), "flexion joints at mid-range": half}
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), constrained_layout=True)
    for ax, (title, q) in zip(axes, poses.items()):
        for f, pose in hand.fk(q).items():
            pts = np.vstack([np.zeros(3), pose.joint_pos, pose.tip_pos]) * 1000
            # View from the palm side: camera on +y looking along -y with the fingers up (-z), so the
            # horizontal axis is +x of r_wrist (a right palm seen from the front: thumb on the right).
            ax.plot(pts[:, 0], -pts[:, 2], "-o", color="#1a1a1a", markersize=3, linewidth=1.5)
            tip = pose.tip_pos * 1000
            n = pose.pad_normal * 12
            ax.annotate("", xy=(tip[0] + n[0], -(tip[2] + n[2])), xytext=(tip[0], -tip[2]),
                        arrowprops={"arrowstyle": "->", "color": "#2a6fdb", "lw": 1.2})
            ax.text(tip[0], -tip[2] + 6, f, ha="center", fontsize=8)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("x of r_wrist [mm] (seen from the palm side)", fontsize=8)
        ax.set_ylabel("−z of r_wrist [mm] (towards the fingertips)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(color="#e6e6e6", linewidth=0.6)
    fig.suptitle("Hand 2 (right) from its URDF; blue = pad normal (12 mm), projected on the view plane",
                 fontsize=9, x=0.01, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
"""Map where the fingertips go for a grid of thumb poses (nothing is touched).

The hand stays at its start pose, high above the cube. Every environment holds one thumb
pose (joint1, joint2, curl = joint3 = joint4) for the whole run, while fingers 2-5 step
together (parallel, joint2 = 0) through FINGER_CLOSURES. At the end of each stage every
fingertip position (relative to the grasp point) and orientation (the tip link's x, y, z
axes) is recorded in the robot base frame (the same frame as the lift script's tables).
The orientations tell which way each fingertip pad faces. The result is printed and saved
to logs/hand_probe/probe_<time>.npz.

    python scripts/tools/probe_hand_kinematics.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Map fingertip positions for a grid of thumb poses.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import itertools
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils.math import matrix_from_quat, quat_inv, quat_mul, subtract_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg

import tunable_comp_thesis.tasks  # noqa: F401  (registers the gym tasks)

TASK = "Isaac-Lift-Cube-UR5Wuji-Scripted-v0"

# the grid: one env per thumb pose (values outside the joint limits are clamped and reported)
THUMB_J1 = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]  # limits [0.05, 1.60]
THUMB_J2 = [-0.14, 0.2, 0.5, 0.93]  # limits [-0.14, 0.93]
THUMB_CURL = [0.6, 0.9, 1.2]  # joint3 = joint4
FINGER_CLOSURES = [0.5, 0.6, 0.7, 0.8]  # fingers 2-5: joints 1, 3, 4 (joint2 = 0)
STAGE_TIME = 1.5  # [s] per finger closure, long enough for the joints to settle


def main():
    grid = list(itertools.product(THUMB_J1, THUMB_J2, THUMB_CURL))
    n = len(grid)
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=n)
    env = gym.make(TASK, cfg=env_cfg)
    uw = env.unwrapped
    arm = uw.action_manager.get_term("arm_action")
    hand = uw.action_manager.get_term("hand_action")
    robot = uw.scene["robot"]
    dev, dt = uw.device, uw.step_dt
    names = hand.joint_names
    hand_ids = robot.find_joints(names, preserve_order=True)[0]
    tip_ids = [robot.find_bodies(f"right_finger{f}_tip_link")[0][0] for f in range(1, 6)]
    limits = robot.data.soft_joint_pos_limits[0, hand_ids].cpu().numpy()  # (20, 2)

    print("\nhand joint limits [rad]:")
    for name, (lo, hi) in zip(names, limits):
        print(f"  {name:24s} [{lo:+.2f}, {hi:+.2f}]")

    def target(stage: int) -> torch.Tensor:
        """Absolute joint targets (n, 20) for a finger-closure stage, clamped to the limits."""
        rows = []
        for j1, j2, curl in grid:
            row = []
            for name in names:  # e.g. "right_finger3_joint2"
                finger = int(name.split("finger")[1][0])
                joint = int(name[-1])
                if finger == 1:
                    row.append({1: j1, 2: j2, 3: curl, 4: curl}[joint])
                else:
                    row.append(0.0 if joint == 2 else FINGER_CLOSURES[stage])
            rows.append(row)
        q = np.clip(np.asarray(rows), limits[:, 0], limits[:, 1])
        return torch.tensor(q, dtype=torch.float32, device=dev)

    env.reset()
    p_start, quat = arm._compute_frame_pose()  # the hand stays here, far above the cube
    steps = int(round(STAGE_TIME / dt))
    tips, tip_axes, q_reached, q_target = [], [], [], []
    with torch.inference_mode():
        for stage in range(len(FINGER_CLOSURES)):
            q = target(stage)
            for _ in range(steps):
                env.step(torch.cat([p_start, quat, q], dim=-1))
            ee_pos, _ = arm._compute_frame_pose()
            tip_w = robot.data.body_pos_w[:, tip_ids]  # (n, 5, 3)
            root_p = robot.data.root_pos_w[:, None].expand(-1, 5, -1).reshape(-1, 3)
            root_q = robot.data.root_quat_w[:, None].expand(-1, 5, -1).reshape(-1, 4)
            tip_b, _ = subtract_frame_transforms(root_p, root_q, tip_w.reshape(-1, 3))
            tips.append((tip_b.reshape(n, 5, 3) - ee_pos[:, None]).cpu().numpy() * 1000.0)  # [mm]
            # tip link orientation in the base frame; columns of R = the link's x, y, z axes
            tip_q_w = robot.data.body_quat_w[:, tip_ids].reshape(-1, 4)
            tip_q_b = quat_mul(quat_inv(root_q), tip_q_w)
            tip_axes.append(matrix_from_quat(tip_q_b).reshape(n, 5, 3, 3).cpu().numpy())
            q_reached.append(robot.data.joint_pos[:, hand_ids].cpu().numpy())
            q_target.append(q.cpu().numpy())
            print(f"  stage {stage}: fingers 2-5 at {FINGER_CLOSURES[stage]:.1f} rad done")
    env.close()

    tips, q_reached, q_target = np.asarray(tips), np.asarray(q_reached), np.asarray(q_target)  # (S, n, ...)
    tip_axes = np.asarray(tip_axes)  # (S, n, 5, 3, 3)
    out_dir = os.path.join("logs", "hand_probe")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"probe_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.npz")
    np.savez_compressed(path, tips_mm=tips, tip_axes=tip_axes, q_reached=q_reached, q_target=q_target, joint_names=np.asarray(names),
                        joint_limits=limits, thumb_grid=np.asarray(grid), finger_closures=np.asarray(FINGER_CLOSURES))

    # --- fingers 2-5 (do not depend on the thumb; median over the envs) ---------------------------
    print("\nfingers 2-5, tip rel. grasp point [mm] (base frame):")
    for s, c in enumerate(FINGER_CLOSURES):
        med = np.median(tips[s][:, 1:], axis=0)
        print(f"  closure {c:.1f}: " + "  ".join(f"f{f + 2} ({p[0]:5.0f},{p[1]:5.0f},{p[2]:5.0f})" for f, p in enumerate(med)))

    # --- thumb, per pose (stage 0: fingers least curled, least chance of a collision) ------------
    thumb_cols = [i for i, nm in enumerate(names) if "finger1_" in nm]
    print("\nthumb tip rel. grasp point [mm] (base frame), per thumb pose; 'blocked' = a joint is >0.05 rad off its target")
    print(f"{'env':>3} {'j1':>5} {'j2':>5} {'curl':>5} | {'x':>6} {'y':>6} {'z':>6} |")
    for i, (j1, j2, curl) in enumerate(grid):
        p = tips[0, i, 0]
        off = np.abs(q_reached[0, i, thumb_cols] - q_target[0, i, thumb_cols]).max()
        flag = "blocked" if off > 0.05 else ""
        print(f"{i:>3} {j1:5.2f} {j2:5.2f} {curl:5.2f} | {p[0]:6.0f} {p[1]:6.0f} {p[2]:6.0f} | {flag}")
    print(f"\n[INFO] saved {path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
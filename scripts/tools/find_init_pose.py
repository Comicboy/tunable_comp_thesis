"""Finds arm joint angles that put the grasp point at a given position, with the start orientation.

Used to choose the start pose of the imitation-learning env (START_JOINTS in il_env_cfg.py): the
start pose is given as joint angles, but what matters is where the hand is. This script starts
from the IL env's current start pose and runs damped-least-squares IK in the simulation until the
grasp point reaches the target position while the hand keeps its start orientation (fingers +x,
palm down). Because the IK moves continuously from the current pose, the arm stays on the same
branch (elbow up). It then prints the joint angles to copy into START_JOINTS.

    python scripts/tools/find_init_pose.py --z 0.15              # lower the start, keep x and y
    python scripts/tools/find_init_pose.py --x 0.50 --y 0.11 --z 0.15

x, y, z are the grasp point in the robot base frame [m]; omitted values stay as in the current start
pose. The hand is held open (its default pose) the whole time. Physics on the CPU.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Find arm joint angles for a grasp-point position.")
parser.add_argument("--x", type=float, default=None, help="grasp point x, base frame [m] (default: unchanged)")
parser.add_argument("--y", type=float, default=None, help="grasp point y [m] (default: unchanged)")
parser.add_argument("--z", type=float, default=None, help="grasp point z [m] (default: unchanged)")
parser.add_argument("--max_steps", type=int, default=1000, help="IK steps at most (50 per second of sim time)")
parser.add_argument("--tol_mm", type=float, default=0.5, help="stop when closer than this [mm] ...")
parser.add_argument("--tol_deg", type=float, default=0.1, help="... and the orientation within this [deg]")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import math

import gymnasium as gym
import torch

from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    matrix_from_quat,
    skew_symmetric_matrix,
    subtract_frame_transforms,
)
from isaaclab_tasks.utils import parse_env_cfg

import tunable_comp_thesis.tasks  # noqa: F401  (registers the gym tasks)
from tunable_comp_thesis.tasks.manager_based.lift_compliant.config.ur5_wuji.il_env_cfg import ARM_JOINTS, EE_OFFSET

TASK = "Isaac-Lift-Cube-UR5Wuji-IL-v0"
PALM = "right_palm_link"
DAMPING = 0.05  # lambda of the damped least squares
MAX_STEP = 0.02  # [rad] per step: slow and smooth, the arm drives follow easily


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    env = gym.make(TASK, cfg=env_cfg)
    uw = env.unwrapped
    robot = uw.scene["robot"]
    hand = uw.action_manager.get_term("hand_action")
    dev = uw.device

    arm_ids = robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
    hand_ids = robot.find_joints(hand.joint_names, preserve_order=True)[0]
    palm_id = robot.find_bodies(PALM)[0][0]
    jac_row = palm_id - 1 if robot.is_fixed_base else palm_id
    offset = torch.tensor([EE_OFFSET], dtype=torch.float, device=dev)
    unit = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev)
    limits = robot.data.soft_joint_pos_limits[:, arm_ids]

    def grasp_pose():
        d = robot.data
        p, q = subtract_frame_transforms(d.root_pos_w, d.root_quat_w, d.body_pos_w[:, palm_id], d.body_quat_w[:, palm_id])
        return combine_frame_transforms(p, q, offset, unit)

    def jacobian():
        """Grasp-point Jacobian (6 x 6, arm joints), base frame; see teleop_lift.py."""
        d = robot.data
        J = robot.root_physx_view.get_jacobians()[:, jac_row, :, :][:, :, arm_ids].clone()
        r_w = (matrix_from_quat(d.body_quat_w[:, palm_id]) @ offset.unsqueeze(-1)).squeeze(-1)
        J[:, 0:3] += torch.bmm(-skew_symmetric_matrix(r_w), J[:, 3:6])
        R_bw = matrix_from_quat(d.root_quat_w).transpose(1, 2)
        J[:, 0:3], J[:, 3:6] = torch.bmm(R_bw, J[:, 0:3]), torch.bmm(R_bw, J[:, 3:6])
        return J

    with torch.inference_mode():
        env.reset()
        q0 = robot.data.default_joint_pos
        q_arm, q_hand = q0[:, arm_ids].clone(), q0[:, hand_ids].clone()
        p_start, quat_start = grasp_pose()
        target = p_start.clone()
        for k, v in enumerate((args_cli.x, args_cli.y, args_cli.z)):
            if v is not None:
                target[0, k] = v
        print(f"\n[pose] start: grasp point {[round(v, 3) for v in p_start[0].tolist()]} m, "
              f"arm {[round(v, 4) for v in q_arm[0].tolist()]} rad")
        print(f"[pose] target: grasp point {[round(v, 3) for v in target[0].tolist()]} m, start orientation")

        for k in range(args_cli.max_steps):
            p, q = grasp_pose()
            pos_err, rot_err = compute_pose_error(p, q, target, quat_start, rot_error_type="axis_angle")
            done = pos_err.norm() < args_cli.tol_mm / 1000 and rot_err.norm() < math.radians(args_cli.tol_deg)
            if done and k > 0:
                break
            e = torch.cat([pos_err, rot_err], dim=-1).unsqueeze(-1)
            J = jacobian()
            A = J @ J.transpose(1, 2) + DAMPING**2 * torch.eye(6, device=dev)
            dq = (J.transpose(1, 2) @ torch.linalg.solve(A, e)).squeeze(-1).clamp(-MAX_STEP, MAX_STEP)
            q_arm = torch.clamp(robot.data.joint_pos[:, arm_ids] + dq, limits[..., 0], limits[..., 1])
            env.step(torch.cat([q_arm, q_hand], dim=-1))
        # let the arm settle on the last targets, then measure
        for _ in range(50):
            env.step(torch.cat([q_arm, q_hand], dim=-1))
        p, q = grasp_pose()
        pos_err, rot_err = compute_pose_error(p, q, target, quat_start, rot_error_type="axis_angle")
        q_final = robot.data.joint_pos[0, arm_ids].tolist()
        R = matrix_from_quat(q)[0]

    ok = pos_err.norm().item() < 2 * args_cli.tol_mm / 1000 and rot_err.norm().item() < math.radians(2 * args_cli.tol_deg)
    print(f"\n[pose] {'reached' if ok else 'NOT reached'} after {k} IK steps: grasp point "
          f"{[round(v, 4) for v in p[0].tolist()]} m, error {1000 * pos_err.norm().item():.2f} mm, "
          f"{math.degrees(rot_err.norm().item()):.2f} deg")
    print(f"[pose] fingers point along {[round(v, 2) for v in R[:, 2].tolist()]}, "
          f"palm faces {[round(v, 2) for v in R[:, 0].tolist()]}")
    print("[pose] arm joints [rad]:")
    for name, v in zip(ARM_JOINTS, q_final):
        print(f"    {name:<22} {v:+.4f}")
    print("\n[pose] for il_env_cfg.py:\nSTART_JOINTS = {")
    for name, v in zip(ARM_JOINTS, q_final):
        print(f'    "{name}": {v:.4f},')
    print("}")
    if not ok:
        print("[pose] WARNING: the target was not reached (out of reach, joint limit, or more steps needed: --max_steps)")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
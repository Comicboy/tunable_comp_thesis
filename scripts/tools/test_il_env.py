"""Checks the imitation-learning env (Isaac-Lift-Cube-UR5Wuji-IL-v0) before anything is built on it.

For a few resets, every env holds its start pose (the action = the default joint angles) for 1 s,
then the script checks:

  1. the action has 26 entries: 6 arm joints (in ARM_JOINTS order) + 20 hand joints;
  2. the arm stays at its start pose (absolute joint actions are interpreted correctly);
  3. the fingers point along base +x and the palm faces -z (the new start pose);
  4. the cube rests on the table (z ~ 0.021 m, not moving) inside its 10 x 10 cm square,
     and the squares of the resets are actually different (randomisation works).

    python scripts/tools/test_il_env.py                 # 4 envs, 3 resets, physics on the CPU
    python scripts/tools/test_il_env.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check the imitation-learning env.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--resets", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import torch

from isaaclab.utils.math import matrix_from_quat, quat_inv, quat_mul, subtract_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg

import tunable_comp_thesis.tasks  # noqa: F401  (registers the gym tasks)
from tunable_comp_thesis.tasks.manager_based.lift_compliant.config.ur5_wuji.il_env_cfg import (
    ARM_JOINTS,
    CUBE_CENTER,
    CUBE_HALF_RANGE,
)

TASK = "Isaac-Lift-Cube-UR5Wuji-IL-v0"
HOLD_S = 1.0  # [s] hold the start pose before checking
CUBE_REST = 0.021  # [m] rest height of the cube centre


def check(ok: bool, text: str) -> bool:
    print(f"  [{'OK' if ok else 'FAIL'}] {text}")
    return ok


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(TASK, cfg=env_cfg)
    uw = env.unwrapped
    robot, cube = uw.scene["robot"], uw.scene["object"]
    arm, hand = uw.action_manager.get_term("arm_action"), uw.action_manager.get_term("hand_action")
    palm_id = robot.find_bodies("right_palm_link")[0][0]
    n, steps = uw.num_envs, int(round(HOLD_S / uw.step_dt))

    all_ok = True
    print(f"\n[test] action terms {uw.action_manager.active_terms}, dims {uw.action_manager.action_term_dim}")
    all_ok &= check(uw.action_manager.total_action_dim == 26, f"total action dim {uw.action_manager.total_action_dim} == 26")
    all_ok &= check(list(arm._joint_names) == ARM_JOINTS, f"arm joint order {list(arm._joint_names)}")

    # the action that holds the start pose: default angles, in each term's joint order
    arm_ids = robot.find_joints(list(arm._joint_names), preserve_order=True)[0]
    hand_ids = robot.find_joints(list(hand._joint_names), preserve_order=True)[0]
    q0 = robot.data.default_joint_pos
    hold = torch.cat([q0[:, arm_ids], q0[:, hand_ids]], dim=-1)
    print(f"[test] start angles of the arm [rad]: {[round(v, 3) for v in q0[0, arm_ids].tolist()]}")

    cube_xy = []
    with torch.inference_mode():
        for r in range(args_cli.resets):
            env.reset()
            for _ in range(steps):
                env.step(hold)
            print(f"\n[test] reset {r + 1}/{args_cli.resets}, after holding {HOLD_S:.1f} s:")

            # 2. arm still at its start pose
            drift = (robot.data.joint_pos[:, arm_ids] - q0[:, arm_ids]).abs().max().item()
            all_ok &= check(drift < 1e-2, f"arm drift from the start pose {drift:.2e} rad (< 1e-2)")

            # 3. palm orientation in the base frame: +z along the fingers, +x out of the palm
            q_rel = quat_mul(quat_inv(robot.data.root_quat_w), robot.data.body_quat_w[:, palm_id])
            R = matrix_from_quat(q_rel)
            fingers, palm = R[:, :, 2], R[:, :, 0]
            all_ok &= check(bool((fingers[:, 0] > 0.99).all()), f"fingers along +x: {[round(v, 2) for v in fingers[0].tolist()]}")
            all_ok &= check(bool((palm[:, 2] < -0.99).all()), f"palm faces -z: {[round(v, 2) for v in palm[0].tolist()]}")
            grasp = uw.scene["ee_frame"].data.target_pos_source[:, 0]  # grasp point, base frame
            print(f"  grasp point (env 0, base frame): {[round(v, 3) for v in grasp[0].tolist()]} m")

            # 4. cube at rest inside its square, base frame
            p, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, cube.data.root_pos_w)
            v = cube.data.root_lin_vel_w.norm(dim=-1)
            dx, dy = p[:, 0] - CUBE_CENTER[0], p[:, 1] - CUBE_CENTER[1]
            inside = (dx.abs() <= CUBE_HALF_RANGE + 1e-3) & (dy.abs() <= CUBE_HALF_RANGE + 1e-3)
            for i in range(n):
                print(f"  env {i}: cube at ({p[i, 0]:.3f}, {p[i, 1]:.3f}, {p[i, 2]:.4f}) m, speed {v[i]:.4f} m/s")
            all_ok &= check(bool(inside.all()), f"cube inside x {CUBE_CENTER[0]}+-{CUBE_HALF_RANGE}, "
                                                f"y {CUBE_CENTER[1]}+-{CUBE_HALF_RANGE}")
            all_ok &= check(bool(((p[:, 2] - CUBE_REST).abs() < 2e-3).all()), f"cube at rest height ~{CUBE_REST} m")
            all_ok &= check(bool((v < 1e-2).all()), "cube not moving (< 1 cm/s)")
            cube_xy.append(p[:, :2].clone())

    if len(cube_xy) > 1:
        spread = torch.stack(cube_xy).std(dim=0).mean().item()
        all_ok &= check(spread > 5e-3, f"cube position differs between resets (std {1000 * spread:.1f} mm)")

    print(f"\n[test] {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
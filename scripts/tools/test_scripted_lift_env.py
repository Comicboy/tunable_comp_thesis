"""Check the scripted-lift environment: can the arm put the grasp point where we ask?

Holds the start pose, moves the grasp point 10 cm up, then to 15 cm above the cube,
keeping the start orientation, and prints the tracking error after each phase.

    python scripts/tools/test_scripted_lift_env.py            (add --headless to skip the GUI)
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check the scripted-lift environment.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg

import tunable_comp_thesis.tasks  # noqa: F401  (registers the gym tasks)

TASK = "Isaac-Lift-Cube-UR5Wuji-Scripted-v0"


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device)
    env = gym.make(TASK, cfg=env_cfg)
    uw = env.unwrapped
    am = uw.action_manager
    arm = am.get_term("arm_action")
    hand = am.get_term("hand_action")
    robot, cube = uw.scene["robot"], uw.scene["object"]
    n, dev = uw.num_envs, uw.device
    print(f"\naction terms: {list(zip(am.active_terms, am.action_term_dim))}")

    env.reset()
    ee_pos0, ee_quat0 = arm._compute_frame_pose()  # grasp point in the base frame (offset included)
    cube_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, cube.data.root_pos_w)
    print(f"grasp point at start (base frame) [m]: {ee_pos0[0].tolist()}")
    print(f"cube centre (base frame)          [m]: {cube_b[0].tolist()}")

    hand_ids = robot.find_joints(hand.joint_names, preserve_order=True)[0]
    q_hand = robot.data.default_joint_pos[:, hand_ids]  # absolute angles, so the default pose itself

    up = ee_pos0 + torch.tensor([0.0, 0.0, 0.10], device=dev)
    above_cube = cube_b + torch.tensor([0.0, 0.0, 0.15], device=dev)
    phases = [("hold start", ee_pos0, 1.0), ("10 cm up", up, 2.0), ("15 cm above cube", above_cube, 3.0)]

    with torch.inference_mode():
        for name, target, duration in phases:
            for _ in range(int(duration / uw.step_dt)):
                actions = torch.cat([target, ee_quat0, q_hand], dim=-1)
                env.step(actions)
            pos, quat = arm._compute_frame_pose()
            err_mm = (pos - target).norm(dim=-1).max().item() * 1e3
            # angle between the current and the start orientation
            err_deg = torch.rad2deg(2 * torch.acos((quat * ee_quat0).sum(-1).abs().clamp(max=1.0))).max().item()
            print(f"{name:>18}: position error {err_mm:6.2f} mm, orientation error {err_deg:5.2f} deg")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
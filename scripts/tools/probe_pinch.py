"""How well can the Wuji thumb pinch against a given finger? Searches thumb and finger poses in sim.

Question behind it: thumb-index pinches work, but a "centred" grasp needs the thumb to oppose the
middle and/or ring finger, and the thumb's opposition joint (right_finger1_joint2, limit 0.93 rad)
seems to stop short of that. This probe measures how close the thumb pad can get to the target
finger's pad, and how well the two pads face each other.

Nothing is grasped: the hand stays at the IL start pose, 20 cm above the table. Every environment
holds one thumb pose (joint1 = flexion, joint2 = opposition, curl = joint3 = joint4) for the whole
run, while the target finger (--finger: index, middle or ring) steps through a grid of closures
(joint1 = knuckle, joint3 = joint4 = middle/distal) and of its spread joint (joint2; positive
tilts it towards the index side, i.e. towards the thumb). All other fingers stay open and straight.
After each stage the script measures, for every environment:

  d       distance between the thumb pad and the target finger's pad [mm]
          (pad point = tip link origin + PAD_DEPTH along the tip's pad normal, its local +x)
  facing  -n_thumb . n_finger: 1 = the pads face each other exactly, 0 = at right angles,
          -1 = they point the same way (e.g. the thumb pressing with its back)
  aim     how well each pad points at the other pad, the smaller of the two cosines
          (1 = both pads point straight at each other)

A proper pinch is small d with facing and aim close to 1. Self-collision is on, so touching pads
stop each other (d then stays at a few mm instead of 0); an open, straight index finger may also be
in the thumb's way when it reaches across to the middle or ring finger. The best poses are printed;
everything is saved to logs/hand_probe/pinch_<finger>_<time>.npz.

    python scripts/tools/probe_pinch.py --headless --finger middle
    python scripts/tools/probe_pinch.py --headless --finger ring
    python scripts/tools/probe_pinch.py --headless --finger index
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Search thumb pinch poses against one finger of the Wuji hand.")
parser.add_argument("--finger", choices=["index", "middle", "ring"], default="middle",
                    help="the finger the thumb should pinch against")
parser.add_argument("--top", type=int, default=15, help="how many of the best poses to print")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import itertools
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils.math import matrix_from_quat
from isaaclab_tasks.utils import parse_env_cfg

import tunable_comp_thesis.tasks  # noqa: F401  (registers the gym tasks)
from tunable_comp_thesis.tasks.manager_based.lift_compliant.config.ur5_wuji.il_env_cfg import ARM_JOINTS

TASK = "Isaac-Lift-Cube-UR5Wuji-IL-v0"

# thumb grid: one env per pose (joint limits: j1 [0.05, 1.60], j2 [-0.14, 0.93], j3/j4 [-0.46, 1.56])
THUMB_J1 = [0.3, 0.6, 0.9, 1.2, 1.5]  # flexion
THUMB_J2 = [0.3, 0.6, 0.8, 0.93]  # opposition (towards the fingers); 0.93 = upper limit
THUMB_CURL = [-0.3, 0.0, 0.3, 0.6]  # joint3 = joint4; straight or slightly bent back faces the fingers best
# target-finger stages, run one after the other in every env
FINGER_J1 = [0.2, 0.5, 0.8, 1.1]  # knuckle flexion
FINGER_J34 = [0.2, 0.5, 0.8, 1.1]  # middle and distal joint
FINGER_SPREAD = [0.0, 0.15, 0.29]  # spread (joint2), towards the index side / the thumb; limit about +0.30
FINGER_NUMBER = {"index": 2, "middle": 3, "ring": 4}
STAGE_TIME = 1.5  # [s] per finger stage: the soft finger drives need ~1 s to settle
PAD_DEPTH = 0.007  # [m] from a tip link origin to its pad surface, along the tip's local +x


def main():
    thumb_grid = list(itertools.product(THUMB_J1, THUMB_J2, THUMB_CURL))
    f = FINGER_NUMBER[args_cli.finger]
    finger_grid = list(itertools.product(FINGER_J1, FINGER_J34, FINGER_SPREAD))
    n = len(thumb_grid)

    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=n)
    env_cfg.episode_length_s = 1e4  # no time-out during the probe
    env = gym.make(TASK, cfg=env_cfg)
    uw = env.unwrapped
    robot = uw.scene["robot"]
    hand = uw.action_manager.get_term("hand_action")
    dev, dt = uw.device, uw.step_dt
    names = hand.joint_names
    arm_ids = robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
    hand_ids = robot.find_joints(names, preserve_order=True)[0]
    thumb_tip = robot.find_bodies("right_finger1_tip_link")[0][0]
    finger_tip = robot.find_bodies(f"right_finger{f}_tip_link")[0][0]
    col = {name: k for k, name in enumerate(names)}  # joint name -> column in the hand action

    def hand_targets(fj1: float, fj34: float, spread: float) -> torch.Tensor:
        """(n, 20) absolute hand references: each env's thumb pose, the target finger's closure and
        spread, all other fingers open and straight."""
        q = torch.zeros(n, len(names), device=dev)
        for i, (j1, j2, curl) in enumerate(thumb_grid):
            q[i, col["right_finger1_joint1"]] = j1
            q[i, col["right_finger1_joint2"]] = j2
            q[i, col["right_finger1_joint3"]] = curl
            q[i, col["right_finger1_joint4"]] = curl
        q[:, col[f"right_finger{f}_joint1"]] = fj1
        q[:, col[f"right_finger{f}_joint2"]] = spread
        q[:, col[f"right_finger{f}_joint3"]] = fj34
        q[:, col[f"right_finger{f}_joint4"]] = fj34
        return q

    def pads():
        """Pad points (n, 3) and pad normals (n, 3) of the thumb and the target finger, world frame."""
        out = []
        for b in (thumb_tip, finger_tip):
            nrm = matrix_from_quat(robot.data.body_quat_w[:, b])[:, :, 0]  # local +x = pad normal
            out.append((robot.data.body_pos_w[:, b] + PAD_DEPTH * nrm, nrm))
        return out

    steps = int(round(STAGE_TIME / dt))
    S = len(finger_grid)
    d = np.zeros((S, n))
    facing = np.zeros((S, n))
    aim = np.zeros((S, n))
    q_hand = np.zeros((S, n, len(names)))

    with torch.inference_mode():
        env.reset()
        q_arm = robot.data.default_joint_pos[:, arm_ids].clone()
        print(f"\n[pinch] thumb vs {args_cli.finger} finger: {n} thumb poses x {S} finger stages, {STAGE_TIME} s each")
        for s, (fj1, fj34, spread) in enumerate(finger_grid):
            action = torch.cat([q_arm, hand_targets(fj1, fj34, spread)], dim=-1)
            for _ in range(steps):
                env.step(action)
            (p_t, n_t), (p_i, n_i) = pads()
            gap = p_i - p_t
            dist = gap.norm(dim=-1).clamp(min=1e-6)
            u = gap / dist.unsqueeze(-1)  # thumb pad -> index pad
            d[s] = 1000 * dist.cpu().numpy()
            facing[s] = (-(n_t * n_i).sum(-1)).cpu().numpy()
            aim[s] = torch.minimum((n_t * u).sum(-1), (-n_i * u).sum(-1)).cpu().numpy()
            q_hand[s] = robot.data.joint_pos[:, hand_ids].cpu().numpy()
            print(f"  stage {s + 1:2d}/{S}: {args_cli.finger} j1 {fj1:.1f}, j3=j4 {fj34:.1f}, spread {spread:.2f}"
                  f" -> closest pads {d[s].min():6.1f} mm, best facing {facing[s].max():5.2f}")

    # --- the best pinches: close pads that face and point at each other ---------------------------------
    # score: distance plus a penalty for pads that do not face/aim at each other (20 mm per unit)
    score = d + 20.0 * (1.0 - facing) + 20.0 * (1.0 - aim)
    order = np.dstack(np.unravel_index(np.argsort(score, axis=None), score.shape))[0][: args_cli.top]
    print(f"\n[pinch] best {args_cli.top} poses (d = pad distance; facing, aim: 1 = pads exactly opposite)")
    print(f"{'#':>3} {'thumb j1':>8} {'j2(opp)':>8} {'curl':>6} | {args_cli.finger + ' j1':>9} {'j3=j4':>6} "
          f"{'spread':>6} | {'d [mm]':>7} {'facing':>7} {'aim':>6}")
    for r, (s, i) in enumerate(order):
        j1, j2, curl = thumb_grid[i]
        fj1, fj34, spread = finger_grid[s]
        print(f"{r + 1:>3} {j1:8.2f} {j2:8.2f} {curl:6.2f} | {fj1:9.2f} {fj34:6.2f} {spread:6.2f} | "
              f"{d[s, i]:7.1f} {facing[s, i]:7.2f} {aim[s, i]:6.2f}")

    # how much does the opposition joint limit matter? best result per opposition value
    print("\n[pinch] best per thumb opposition value (joint2):")
    for j2 in THUMB_J2:
        envs = [i for i, (_, b, _) in enumerate(thumb_grid) if b == j2]
        sub = score[:, envs]
        s, k = np.unravel_index(np.argmin(sub), sub.shape)
        i = envs[k]
        print(f"    j2 = {j2:4.2f}: d {d[s, i]:6.1f} mm, facing {facing[s, i]:5.2f}, aim {aim[s, i]:5.2f}  "
              f"(closest pads at this j2: {d[:, envs].min():5.1f} mm)")

    close = d < 10.0
    print(f"\n[pinch] pads closer than 10 mm in {int(close.sum())} of {d.size} cases; "
          f"best facing among them: {facing[close].max() if close.any() else float('nan'):.2f}")
    print(f"[pinch] best facing overall: {facing.max():.2f}; closest pads overall: {d.min():.1f} mm")

    out_dir = os.path.join("logs", "hand_probe")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"pinch_{args_cli.finger}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.npz")
    np.savez(path, d_mm=d, facing=facing, aim=aim, q_hand=q_hand, joint_names=np.array(names),
             thumb_grid=np.array(thumb_grid), finger_grid=np.array(finger_grid), finger=args_cli.finger,
             pad_depth=PAD_DEPTH)
    print(f"[pinch] saved {path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
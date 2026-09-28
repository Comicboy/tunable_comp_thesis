"""Inspect the combined UR5 + Wuji Hand articulation.

Prints joint/body inventory, drive properties, and the joint-space inertia and
gravity torques in a few poses.

Run with:
    python scripts/inspect_ur5.py --headless
"""

import argparse
import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--arm-only", action="store_true", help="inspect the bare UR5 instead")
parser.add_argument("--left", action="store_true", help="use the left-hand build")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from tunable_comp_thesis.assets import UR5_CFG, UR5_WUJI_LEFT_CFG, UR5_WUJI_RIGHT_CFG

print("\n=== run at", datetime.datetime.now().strftime("%H:%M:%S"), "===")

if args.arm_only:
    cfg, cfg_name = UR5_CFG, "UR5_CFG"
elif args.left:
    cfg, cfg_name = UR5_WUJI_LEFT_CFG, "UR5_WUJI_LEFT_CFG"
else:
    cfg, cfg_name = UR5_WUJI_RIGHT_CFG, "UR5_WUJI_RIGHT_CFG"

print("config:", cfg_name)
print("usd:   ", cfg.spawn.usd_path)

DT = 0.005
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device))
robot = Articulation(cfg.replace(prim_path="/World/Robot"))
sim.reset()

names = robot.joint_names
idx = {n: i for i, n in enumerate(names)}

print("\n=== articulation ===")
print("num joints:", len(names))
print("joint names:", names)
print("num bodies:", len(robot.body_names))
print("body names:", robot.body_names)
print("fixed base:", robot.is_fixed_base)
print("total mass:", robot.data.default_mass[0].sum().item(), "kg")
print("  (UR5 alone measured 20.99 kg; the hand adds the rest)")

arm = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
       "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
missing = [n for n in arm if n not in idx]
if missing:
    print("\n!! arm joints not found:", missing)

print("\n=== drive properties (arm joints) ===")
for n in arm:
    if n in idx:
        i = idx[n]
        print(f"  {n:20s} Kp={robot.data.joint_stiffness[0, i]:10.1f}"
              f"  Kd={robot.data.joint_damping[0, i]:8.2f}"
              f"  tau_max={robot.data.joint_effort_limits[0, i]:7.1f}"
              f"  limits=[{robot.data.joint_pos_limits[0, i, 0]:+.3f},"
              f" {robot.data.joint_pos_limits[0, i, 1]:+.3f}] rad")

finger = [n for n in names if "finger" in n]
if finger:
    side = "left" if finger[0].startswith("left") else "right"
    print(f"\n=== drive properties, all {len(finger)} finger joints ({side}) ===")
    for f in range(1, 6):
        print(f"  finger{f}:")
        for j in range(1, 5):
            n = f"{side}_finger{f}_joint{j}"
            if n in idx:
                i = idx[n]
                print(f"    joint{j}  Kp={robot.data.joint_stiffness[0, i]:7.2f}"
                      f"  Kd={robot.data.joint_damping[0, i]:6.3f}"
                      f"  tau_max={robot.data.joint_effort_limits[0, i]:6.2f}"
                      f"  limits=[{robot.data.joint_pos_limits[0, i, 0]:+.3f},"
                      f" {robot.data.joint_pos_limits[0, i, 1]:+.3f}]")

# Arm poses in rad; every other joint stays at 0.
poses = {
    "folded":    {"shoulder_lift_joint": -1.712, "elbow_joint": 1.712},
    "stretched": {},                                    # arm horizontal
    "working":   {"shoulder_lift_joint": -1.0, "elbow_joint": 1.0,
                  "wrist_1_joint": -1.57, "wrist_2_joint": -1.57},
}

print("\n=== inertia and gravity per pose (arm joints) ===")
for label, angles in poses.items():
    q = torch.zeros((1, len(names)), device=sim.device)
    for name, value in angles.items():
        q[0, idx[name]] = value

    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    sim.step(render=False)
    robot.update(DT)

    M = robot.root_physx_view.get_generalized_mass_matrices()[0]
    tau_g = robot.root_physx_view.get_gravity_compensation_forces()[0]

    print(f"\n{label}")
    for n in arm:
        if n in idx:
            i = idx[n]
            print(f"  {n:20s} J={M[i, i]:8.4f} kg m^2   gravity={tau_g[i]:+8.2f} N m")

sim.stop()
app.close()

import os
os._exit(0)
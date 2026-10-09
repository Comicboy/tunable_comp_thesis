"""Check the fingertip admittance action inside the lift env (no cube grasp yet).

1. prints the index tables (which sensor row, Jacobian row and joints each finger uses)
2. holds the default pose for 1 s, then curls all fingers for 2 s; whatever the
   fingertips touch (each other, the palm) produces forces, and the admittance
   must make the soft envs yield more than the stiff ones, and inf not at all
3. checks the damped pseudo-inverse: J dq should reproduce a small tip offset e
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check the fingertip admittance action.")
parser.add_argument("--controller", choices=["dynamic", "static"], default="dynamic")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import tunable_comp_thesis.tasks  # noqa: F401  (registers the gym tasks)
from tunable_comp_thesis.controllers import AdmittanceControllerCfg, StaticAdmittanceControllerCfg

TASK = "Isaac-Lift-Cube-UR5Wuji-Play-v0"
STIFFNESS = [50.0, 150.0, 500.0, float("inf")]


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=len(STIFFNESS))
    hand = env_cfg.actions.hand_action
    hand.stiffness_values = STIFFNESS
    hand.controller = AdmittanceControllerCfg() if args_cli.controller == "dynamic" else StaticAdmittanceControllerCfg()
    env = gym.make(TASK, cfg=env_cfg)
    uw = env.unwrapped
    term = uw.action_manager.get_term("hand_action")
    robot = uw.scene["robot"]
    n, dev = uw.num_envs, uw.device

    # --- 1. index tables ------------------------------------------------------------------
    print(f"\ncontroller: {type(term._controller).__name__}, stiffness per env: {term.stiffness.squeeze(1).tolist()}")
    print(f"contact sensor bodies: {term._sensor.body_names}")
    for i in range(5):
        print(
            f"  finger{i + 1}: sensor row {term._sensor_rows[i]}, jacobian row {term._jac_rows[i]}, "
            f"joints {[robot.joint_names[j] for j in term._jac_cols[i]]}, action cols {term._action_cols[i]}"
        )

    # --- 2. hold, then curl the fingers ----------------------------------------------------------
    am = uw.action_manager
    start = sum(am.action_term_dim[: am.active_terms.index("hand_action")])  # where the hand's slice begins
    ids = robot.find_joints(term.joint_names, preserve_order=True)[0]
    q_default = robot.data.default_joint_pos[:, ids]
    upper = robot.data.soft_joint_pos_limits[:, ids, 1]
    flexion = torch.tensor([not name.endswith("joint2") for name in term.joint_names], device=dev)
    q_closed = torch.where(flexion, 0.9 * upper, q_default)  # joints 1, 3, 4 curl; joint2 (spread) stays

    actions = torch.zeros(n, am.total_action_dim, device=dev)
    max_f = torch.zeros(n)
    max_e = torch.zeros(n)
    max_dq = torch.zeros(n)
    env.reset()
    with torch.inference_mode():
        for k in range(150):  # 3 s at 50 Hz
            q_ref = q_default if k < 50 else q_closed
            actions[:, start : start + term.action_dim] = (q_ref - q_default) / hand.scale
            env.step(actions)
            max_f = torch.maximum(max_f, term.tip_forces.norm(dim=-1).amax(dim=1).cpu())
            max_e = torch.maximum(max_e, term.tip_offsets.norm(dim=-1).amax(dim=1).cpu())
            max_dq = torch.maximum(max_dq, term.joint_offsets.abs().amax(dim=1).cpu())
        finite = bool(torch.isfinite(term.joint_command).all())

    print("\ncurling the fingers (max over the run, over the 5 tips):")
    print(f"{'K [N/m]':>8} {'|f| [N]':>8} {'|e| [mm]':>9} {'|dq| [rad]':>11}")
    for i, k in enumerate(STIFFNESS):
        print(f"{k:8.0f} {max_f[i].item():8.2f} {max_e[i].item() * 1e3:9.2f} {max_dq[i].item():11.4f}")
    print(f"commands finite: {finite}, inf env yields exactly 0: {max_dq[-1].item() == 0.0}")

    # --- 3. pseudo-inverse check at the current pose ---------------------------------------------
    jac = robot.root_physx_view.get_jacobians()
    e = torch.zeros(n, 3, device=dev)
    e[:, 2] = 0.002  # 2 mm up (world z)
    print("\npseudo-inverse: residual |J dq - e| / |e| for a 2 mm offset (few % is normal; large = near-singular direction)")
    for i in range(5):
        J = jac[:, term._jac_rows[i], :3, :][:, :, term._jac_cols[i]]
        dq = term._damped_pinv_times(J, e)
        res = ((J @ dq.unsqueeze(-1)).squeeze(-1) - e).norm(dim=-1) / e.norm(dim=-1)
        print(f"  finger{i + 1}: {res[0].item() * 100:5.1f} %")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
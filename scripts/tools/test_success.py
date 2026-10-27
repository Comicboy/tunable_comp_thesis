"""Checks the success condition of the IL env (the "success" termination term, CubeHeldAboveHeight).

The robot holds its start pose; the script moves the cube itself (placed every step, gravity off on
the cube so it stays exactly where it is put, away from the hand at x = 0.70 m). Four envs, four
scenarios, heights of the cube's bottom face above the table:

  env 0: on the table until 0.5 s, then 0.160 m                  -> success at 3.5 s
  env 1: on the table until 0.5 s, then 0.147 m (within the 5 mm tolerance) -> success at 3.5 s
  env 2: on the table until 0.5 s, then 0.140 m (below the tolerance)       -> no success
  env 3: 0.160 m from 0.5 s, dips to 0.100 m from 2.5 to 3.0 s, back to 0.160 m
         (the dip restarts the timer)                            -> success at 6.0 s

and checks the time of each success (to the env step: 3.0 s must be exactly 150 steps) and that the episode ended there as
"terminated" with get_term("success") set. Runs in real time by default so you can watch env 0's
plane: light green, darker green from 0.5 s, then the episode ends at 3.5 s (cube back on the table,
plane light green again).

    python scripts/tools/test_success.py
    python scripts/tools/test_success.py --fast        # no real-time pacing
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check the IL env's success condition.")
parser.add_argument("--fast", action="store_true", help="run as fast as possible (no real-time pacing)")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import time

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import tunable_comp_thesis.tasks  # noqa: F401  (registers the gym tasks)
from tunable_comp_thesis.tasks.manager_based.lift_compliant.config.ur5_wuji.il_env_cfg import (
    CUBE_HALF_SIZE,
    SUCCESS_HOLD_S,
)

TASK = "Isaac-Lift-Cube-UR5Wuji-IL-v0"
RUN_S = 7.0
PLACE_XY = (0.70, 0.11)  # [m] base frame: under the plane, 15 cm in front of the hand
TABLE = None  # "leave the cube where it is" (on the table)


def bottom_height(env_id: int, t: float):
    """Scenario: height of the cube's bottom face [m] at time t, or None = leave it on the table."""
    if t < 0.5:
        return TABLE
    if env_id == 0:
        return 0.160
    if env_id == 1:
        return 0.147
    if env_id == 2:
        return 0.140
    return 0.100 if 2.5 <= t < 3.0 else 0.160


EXPECTED = {0: 3.5, 1: 3.5, 2: None, 3: 6.0}  # [s] end time of the step at which success is reported


def check(ok: bool, text: str) -> bool:
    print(f"  [{'OK' if ok else 'FAIL'}] {text}")
    return ok


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=4)
    env_cfg.scene.object.spawn.rigid_props.disable_gravity = True  # the cube stays where it is put
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (1.25, -0.55, 0.55)
    env_cfg.viewer.lookat = (0.60, 0.11, 0.12)
    env = gym.make(TASK, cfg=env_cfg)
    uw = env.unwrapped
    robot, cube = uw.scene["robot"], uw.scene["object"]
    arm, hand = uw.action_manager.get_term("arm_action"), uw.action_manager.get_term("hand_action")
    term = uw.termination_manager.get_term_cfg("success").func  # the CubeHeldAboveHeight instance
    dt, n = uw.step_dt, uw.num_envs

    arm_ids = robot.find_joints(list(arm._joint_names), preserve_order=True)[0]
    hand_ids = robot.find_joints(list(hand._joint_names), preserve_order=True)[0]
    q0 = robot.data.default_joint_pos
    hold = torch.cat([q0[:, arm_ids], q0[:, hand_ids]], dim=-1)

    print(f"\n[test] termination terms: {uw.termination_manager.active_terms}")
    all_ok = check("success" in uw.termination_manager.active_terms, "'success' term present")

    success_t = {i: None for i in range(n)}
    other_end = {i: None for i in range(n)}  # any other ending (should not happen)
    zero_vel = torch.zeros(1, 6, device=uw.device)
    env.reset()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for k in range(int(round(RUN_S / dt))):
            t = k * dt  # time at the start of this step
            for i in range(n):
                h = bottom_height(i, t)
                if h is None or success_t[i] is not None:  # after success: leave the reset cube alone
                    continue
                pose = torch.tensor([[PLACE_XY[0], PLACE_XY[1], h + CUBE_HALF_SIZE, 1.0, 0.0, 0.0, 0.0]],
                                    device=uw.device)
                pose[:, :3] += uw.scene.env_origins[i]
                ids = torch.tensor([i], device=uw.device)
                cube.write_root_pose_to_sim(pose, env_ids=ids)
                cube.write_root_velocity_to_sim(zero_vel, env_ids=ids)

            _, _, terminated, truncated, _ = env.step(hold)
            t_end = (k + 1) * dt
            ok_flag = uw.termination_manager.get_term("success")
            for i in range(n):
                if terminated[i] and ok_flag[i] and success_t[i] is None:
                    success_t[i] = t_end
                    print(f"  env {i}: success at {t_end:.2f} s")
                elif (terminated[i] or truncated[i]) and not ok_flag[i] and other_end[i] is None:
                    other_end[i] = t_end
                    print(f"  env {i}: episode ended WITHOUT success at {t_end:.2f} s")
            if k % 25 == 24:  # every 0.5 s
                print(f"  t = {t_end:4.2f} s  above {term.above.int().tolist()}  "
                      f"hold [s] {[round(v, 2) for v in term.hold_time.tolist()]}")
            if not args_cli.fast:
                wait = t0 + t_end - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)

    print(f"\n[test] results (hold {SUCCESS_HOLD_S} s):")
    for i in range(n):
        exp, got = EXPECTED[i], success_t[i]
        if exp is None:
            all_ok &= check(got is None, f"env {i}: no success (got {got})")
        else:
            all_ok &= check(got is not None and abs(got - exp) < 0.5 * dt, f"env {i}: success at {got if got is None else round(got, 3)} s, expected {exp} s")
        all_ok &= check(other_end[i] is None, f"env {i}: no other ending")

    print(f"\n[test] {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
"""Rolls out a trained policy in the IL env and reports its success rate.

Runs in the Isaac Lab env. The policy itself runs in scripts/il/policy_server.py (lerobot env); this
script sends it the state every env step and applies the action it returns. Start the server first:

    # terminal 1, conda env "lerobot"
    python scripts/il/policy_server.py --policy logs/il_train/<run>/checkpoints/last/pretrained_model

    # terminal 2, conda env "env_isaaclab", repo root
    python scripts/il/rollout_policy.py --stiffness 150 --episodes 20 --tag <run>
    python scripts/il/rollout_policy.py --stiffness 500 --episodes 20 --tag <run>
    python scripts/il/rollout_policy.py --stiffness 150 --episodes 20 --tag <run> --headless   # faster, no window

Same env and settings as the demos: Isaac-Lift-Cube-UR5Wuji-IL-v0, physics on the CPU, one env, one
stiffness K for the whole run (as in the recordings, one file per K). The policy performs the
fingertip admittance itself: the env's admittance law is off (its default), so the policy's finger
commands go to the drives unchanged; the env still smooths the contact forces for the state and
reports --stiffness as the K the policy is asked to behave with. The cube start positions are drawn by this script from --seed (uniform in the env's 10 x 10 cm
spawn square) and the cube is placed there at every episode start, so runs with the same seed see the
same position in episode i, whatever K or policy. (Relying on the env's own reset randomisation did
not give that: the env's random numbers are also used during an episode, e.g. by the lift task's
target command, so after the first episode the positions depended on how long the earlier episodes
had lasted.) --only runs selected episodes of that list, e.g. to watch failures again.

Every env step (50 Hz):
    state     = the IL state (51 numbers) split as in the dataset: cube position -> environment_state
                (3), everything else -> state (48), in the same order as scripts/il/convert_to_lerobot.py
    action    = the server's reply (26 absolute joint commands [arm targets | finger commands q_cmd]) -> env.step
Each episode starts with --settle_s (1 s) of holding the start pose (the cube settles; in the demos
the robot also stood still before the clutch was released), then the policy takes over.
An episode ends as in the demos: success (cube bottom >= 15 cm above the table for 3 s, the env's
"success" term), the cube off the table, or the time limit (--episode_s, default 30 s: the demos
took 8.5-12.7 s from the clutch). The server's action queue is reset at every episode start.

Results: one line per episode, a summary (success rate, time to success), and a JSON file
(default logs/il_eval/<tag>_<inference>_K<K>_seed<seed>.json, <inference> = n100 or te0.01 as reported
by the server, plus _only<episodes> for --only runs; an existing full-run file is not overwritten
without --overwrite, --only runs are) with the settings, the
server's info and every episode.

Videos (--video, usually with --only): the policy part of every episode (from the end of the settling
to the episode's end) from the palm-following camera (--cam_eye / --cam_lookat, by default the view of
the demo recordings), one mp4 per episode in <results name>_videos/ep<NNN>_K<K>_<outcome>.mp4, one frame
per env step (50 fps = real time). Rendering slows the run down; the physics is the same.

    python scripts/il/rollout_policy.py --stiffness 150 --tag act_qcmd_v1 --only 12 31 --video --headless

Trajectories: next to the JSON file, <same name>.hdf5 holds every episode's states and actions in the
layout of the demo files (/data/ep_NNN/obs (T, 51) = the state each step started from, /actions
(T, 26) = the action sent in it; attrs outcome, success, duration_s; file attrs stiffness, step_dt,
obs_terms, joint names, success condition, server info), so the rollouts can be compared with the
demos by the same code (scripts/il/plot_stiffness_behaviour.py). --no_traj switches this off.
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Roll out a policy (served by policy_server.py) in the IL env.")
parser.add_argument("--stiffness", type=float, required=True, help="K [N/m] for the whole run (150 or 500)")
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--episode_s", type=float, default=30.0, help="time limit of an episode [s]")
parser.add_argument("--seed", type=int, default=0, help="seed of the cube start positions (and of the env)")
parser.add_argument("--only", type=int, nargs="+", default=None,
                    help="run only these episode numbers of the seed's position list (e.g. --only 2 5 10)")
parser.add_argument("--server", default="127.0.0.1:9880", help="HOST:PORT of policy_server.py")
parser.add_argument("--settle_s", type=float, default=1.0,
                    help="hold the start pose this long after each reset before the policy takes over [s] "
                         "(in the demos the robot also stood still >= 1.4 s before the clutch)")
parser.add_argument("--tag", default="policy", help="name of the policy/run, used in the output file name")
parser.add_argument("--out", default=None,
                    help="JSON results file (default logs/il_eval/<tag>_<inference>_K<K>_seed<seed>.json, "
                         "<inference> = n<n_action_steps> or te<coeff>, from the server)")
parser.add_argument("--overwrite", action="store_true", help="replace an existing results file")
parser.add_argument("--realtime", action="store_true", help="pace the sim to real time (for watching)")
parser.add_argument("--no_traj", action="store_true", help="do not save the states and actions (.hdf5)")
parser.add_argument("--video", action="store_true",
                    help="save the policy part of every episode as an mp4 (camera as below; use with --only)")
parser.add_argument("--video_res", type=int, nargs=2, default=[1280, 720], help="video width height [px]")
parser.add_argument("--cam_eye", type=float, nargs=3, default=[-0.25, 0.12, 0.20],
                    help="camera position relative to the palm [m] (default: the view used for the recordings)")
parser.add_argument("--cam_lookat", type=float, nargs=3, default=[0.20, 0.0, -0.10],
                    help="point the camera looks at, relative to the palm [m]")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")  # as for the demos
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True  # off-screen rendering of the viewer camera
FRAME_LIMIT_OFF = "--/app/runLoops/main/rateLimitEnabled=false --/app/vsync=false"
args_cli.kit_args = f"{args_cli.kit_args or ''} {FRAME_LIMIT_OFF}".strip()
simulation_app = AppLauncher(args_cli).app

import datetime
import json
import os
import socket
import time

import gymnasium as gym
import h5py
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg

import tunable_comp_thesis.tasks  # noqa: F401  (registers the gym tasks)
from tunable_comp_thesis.controllers import AdmittanceControllerCfg
from tunable_comp_thesis.utils import VideoWriter
from tunable_comp_thesis.tasks.manager_based.lift_compliant.config.ur5_wuji.il_env_cfg import (
    ARM_JOINTS,
    CUBE_CENTER,
    CUBE_HALF_RANGE,
    CUBE_HALF_SIZE,
    CUBE_REST_Z,
)

TASK = "Isaac-Lift-Cube-UR5Wuji-IL-v0"
PALM = "right_palm_link"
ENV_TERMS = ("cube_position",)  # -> environment_state; must match scripts/il/convert_to_lerobot.py


class PolicyClient:
    """Talks to policy_server.py: one JSON line per request and per reply."""

    def __init__(self, address: str):
        host, _, port = address.rpartition(":")
        try:
            self._sock = socket.create_connection((host, int(port)), timeout=30.0)
        except OSError as exc:
            raise SystemExit(f"[rollout] cannot reach the policy server at {address} ({exc}). "
                             "Start scripts/il/policy_server.py first.") from exc
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._f = self._sock.makefile("rw", encoding="utf-8")

    def call(self, **req) -> dict:
        self._f.write(json.dumps(req) + "\n")
        self._f.flush()
        line = self._f.readline()
        if not line:
            raise SystemExit("[rollout] the policy server closed the connection")
        rep = json.loads(line)
        if "error" in rep:
            raise SystemExit(f"[rollout] policy server error: {rep['error']}")
        return rep

    def close(self):
        self._f.close()
        self._sock.close()


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    env_cfg.seed = args_cli.seed
    env_cfg.actions.hand_action.stiffness_values = [args_cli.stiffness]
    # admittance law off (the IL env's default): the policy's finger commands go to the drives as they are;
    # the controller cfg only sets the force smoothing for the state, as in the recordings
    env_cfg.actions.hand_action.controller = AdmittanceControllerCfg()
    assert not env_cfg.actions.hand_action.admittance_enabled, "the IL env should have the admittance law off"
    env_cfg.episode_length_s = args_cli.episode_s
    env_cfg.viewer.origin_type = "asset_body"  # the camera follows the palm
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.body_name = PALM
    env_cfg.viewer.eye = tuple(args_cli.cam_eye)
    env_cfg.viewer.lookat = tuple(args_cli.cam_lookat)
    env_cfg.viewer.resolution = tuple(args_cli.video_res)
    env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    uw = env.unwrapped
    dt = uw.step_dt
    success_cfg = uw.termination_manager.get_term_cfg("success")
    height = success_cfg.params["height"]

    # --- the dataset's split of the 51-number state ------------------------------------------------
    om = uw.observation_manager
    names, dims = om.active_terms["policy"], [d[0] for d in om.group_obs_term_dim["policy"]]
    state_idx, env_idx, start = [], [], 0
    for name, d in zip(names, dims):
        (env_idx if name in ENV_TERMS else state_idx).extend(range(start, start + d))
        start += d
    cube_z = env_idx[2]  # index of the cube's z in the 51 numbers

    # action that holds the start pose (default angles, arm in ARM_JOINTS order, hand in the action's order)
    robot, hand = uw.scene["robot"], uw.action_manager.get_term("hand_action")
    arm_ids = robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
    hand_ids = robot.find_joints(hand.joint_names, preserve_order=True)[0]
    q0 = robot.data.default_joint_pos
    hold = torch.cat([q0[:, arm_ids], q0[:, hand_ids]], dim=-1)
    settle_steps = int(round(args_cli.settle_s / dt))

    # cube start positions: drawn here from --seed, the same list for every run with this seed
    episodes = args_cli.only if args_cli.only is not None else list(range(args_cli.episodes))
    rng = np.random.default_rng(args_cli.seed)
    xy = np.asarray(CUBE_CENTER) + rng.uniform(-CUBE_HALF_RANGE, CUBE_HALF_RANGE, size=(max(episodes) + 1, 2))
    cube = uw.scene["object"]
    zero_vel = torch.zeros(1, 6, device=uw.device)

    def place_cube(i: int):
        """Cube at rest at position i of the list (base frame = env frame here), faces aligned as at spawn."""
        pose = torch.tensor([[xy[i, 0], xy[i, 1], CUBE_REST_Z, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float, device=uw.device)
        pose[:, :3] += uw.scene.env_origins[0]
        cube.write_root_pose_to_sim(pose)
        cube.write_root_velocity_to_sim(zero_vel)

    client = PolicyClient(args_cli.server)
    info = client.call(cmd="info")
    print(f"\n[rollout] server: {info}")
    # results file: the inference setting comes from the server, so two settings never share a name
    mode = (f"te{info['temporal_ensemble_coeff']:g}" if info.get("temporal_ensemble_coeff") is not None
            else f"n{info.get('n_action_steps')}")
    out = args_cli.out or os.path.join(
        "logs", "il_eval",
        f"{args_cli.tag}_{mode}_K{args_cli.stiffness:g}_seed{args_cli.seed}"
        f"{'_only' + '-'.join(map(str, args_cli.only)) if args_cli.only else ''}.json")
    # full runs are protected; --only runs (watching selected episodes) may simply be repeated
    if os.path.exists(out) and not args_cli.overwrite and not args_cli.only:
        raise SystemExit(f"[rollout] {out} exists; use another --tag, --out, or --overwrite")
    expected = (len(state_idx), len(env_idx), uw.action_manager.total_action_dim)
    got = (info["state_dim"], info["env_state_dim"], info["action_dim"])
    if expected != got:
        raise SystemExit(f"[rollout] sizes (state, env_state, action) differ: env {expected}, policy {got}")
    # trajectories (same layout as the demo files), written after every episode
    traj = None if args_cli.no_traj else os.path.splitext(out)[0] + ".hdf5"
    video_dir = os.path.splitext(out)[0] + "_videos" if args_cli.video else None
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)
    if traj is not None:
        os.makedirs(os.path.dirname(os.path.abspath(traj)), exist_ok=True)
        ends = np.cumsum(dims).tolist()
        with h5py.File(traj, "w") as f:
            f.create_group("data")
            f.attrs.update({
                "source": "policy rollout",
                "task": TASK,
                "stiffness": float(args_cli.stiffness),
                "step_dt": float(dt),
                "seed": int(args_cli.seed),
                "obs_dim": int(sum(dims)),
                "action_dim": int(uw.action_manager.total_action_dim),
                "obs_terms": json.dumps([[n, e - d, e] for n, d, e in zip(names, dims, ends)]),
                "arm_joints": json.dumps(ARM_JOINTS),
                "hand_joints": json.dumps(list(hand.joint_names)),
                "success": json.dumps({k: v for k, v in success_cfg.params.items()
                                       if k in ("height", "half_size", "hold_s", "tolerance")}),
                "server": json.dumps(info),
                "created": datetime.datetime.now().isoformat(timespec="seconds"),
            })

    print(f"[rollout] {len(episodes)} episodes at K = {args_cli.stiffness:g} N/m, seed {args_cli.seed}, "
          f"time limit {args_cli.episode_s:g} s (incl. {args_cli.settle_s:g} s settling), device {uw.device}\n")

    results = []
    obs, _ = env.reset()
    if video_dir:  # the renderer needs a few frames before the image is complete (first ones can be black)
        for _ in range(10):
            uw.render()
    with torch.inference_mode():
        for ep in episodes:
            place_cube(ep)
            for _ in range(settle_steps):  # the cube settles (it spawns 1 mm above the table), robot at rest
                obs, *_ = env.step(hold)
            client.call(cmd="reset")
            o = obs["policy"][0].cpu().numpy()
            cube0 = o[env_idx].tolist()
            max_lift, steps, replans, infer_ms = 0.0, 0, 0, []
            obs_log, act_log = [], []
            video = VideoWriter(os.path.join(video_dir, f"ep{ep:03d}.mp4"), fps=round(1.0 / dt)) if video_dir else None
            t_wall = time.perf_counter()
            while True:
                o = obs["policy"][0].cpu().numpy().astype(float)
                rep = client.call(cmd="act", state=o[state_idx].tolist(), env_state=o[env_idx].tolist())
                replans += int(rep["replanned"])
                infer_ms.append(rep["infer_ms"])
                action = torch.tensor([rep["action"]], dtype=torch.float, device=uw.device)
                if video is not None:  # the frame of s_t (the step that ends an episode already shows the next one)
                    video.add(uw.render())
                obs_log.append(o.astype(np.float32))  # s_t: the state this step starts from
                act_log.append(np.asarray(rep["action"], dtype=np.float32))  # a_t
                obs, _, terminated, truncated, _ = env.step(action)
                steps += 1
                if not (bool(terminated[0]) or bool(truncated[0])):  # after an ending, obs is already the next episode's
                    max_lift = max(max_lift, float(obs["policy"][0, cube_z]) - CUBE_HALF_SIZE)
                if args_cli.realtime:
                    wait = t_wall + steps * dt - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                if bool(terminated[0]) or bool(truncated[0]):
                    if bool(terminated[0]) and bool(uw.termination_manager.get_term("success")[0]):
                        outcome = "success"
                    elif bool(truncated[0]):
                        outcome = "time limit"
                    else:
                        outcome = "cube off the table"
                    break
            r = {
                "episode": ep,
                "outcome": outcome,
                "success": outcome == "success",
                "duration_s": steps * dt,  # policy time, without the settling
                "cube_start": [round(v, 4) for v in cube0],
                "max_lift_m": round(max_lift, 4),
                "replans": replans,
                "infer_ms_mean": float(np.mean(infer_ms)),
                "infer_ms_max": float(np.max(infer_ms)),
                "wall_s": time.perf_counter() - t_wall,
            }
            results.append(r)
            if traj is not None:
                with h5py.File(traj, "a") as f:
                    g = f["data"].create_group(f"ep_{ep:03d}")
                    g.create_dataset("obs", data=np.stack(obs_log))
                    g.create_dataset("actions", data=np.stack(act_log))
                    g.attrs.update({"episode": ep, "outcome": outcome, "success": outcome == "success",
                                    "duration_s": steps * dt})
            if video is not None:
                name = f"ep{ep:03d}_K{args_cli.stiffness:g}_{outcome.replace(' ', '_')}.mp4"
                path = video.close(rename_to=os.path.join(video_dir, name))
                print(f"[rollout] video -> {path} ({video.frames} frames, {video.seconds:.1f} s)")
            print(f"[rollout] episode {ep:3d}: {outcome:<18} {r['duration_s']:5.1f} s | cube start "
                  f"({cube0[0]:+.3f}, {cube0[1]:+.3f}) | max lift {100 * max_lift:5.1f} cm "
                  f"(target {100 * height:.0f}) | replans {replans}")

    client.close()

    # --- summary ----------------------------------------------------------------------------------
    n = len(results)
    ok = [r for r in results if r["success"]]
    rate = len(ok) / max(n, 1)
    # 95 % Wilson interval of the success rate (few episodes: the plain p +- 1.96 sigma is too optimistic)
    z = 1.96
    centre = (rate + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    outcomes = {o: sum(r["outcome"] == o for r in results) for o in ("success", "cube off the table", "time limit")}
    print(f"\n[rollout] K = {args_cli.stiffness:g}: success {len(ok)}/{n} = {100 * rate:.0f} % "
          f"(95 % interval {100 * max(centre - half, 0):.0f}-{100 * min(centre + half, 1):.0f} %) | {outcomes}")
    if ok:
        d = np.array([r["duration_s"] for r in ok])
        print(f"[rollout] time to success (incl. the 3 s hold): mean {d.mean():.1f} s, min {d.min():.1f} s, "
              f"max {d.max():.1f} s (demos: 8.5-12.7 s)")
    print(f"[rollout] inference: mean {np.mean([r['infer_ms_mean'] for r in results]):.2f} ms per step, "
          f"max {max(r['infer_ms_max'] for r in results):.1f} ms")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({
            "tag": args_cli.tag,
            "task": TASK,
            "stiffness": args_cli.stiffness,
            "seed": args_cli.seed,
            "episode_s": args_cli.episode_s,
            "device": str(uw.device),
            "server": info,
            "success_rate": rate,
            "outcomes": outcomes,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "episodes": results,
        }, fh, indent=1)
    print(f"[rollout] results -> {out}" + (f", trajectories -> {traj}" if traj else ""))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
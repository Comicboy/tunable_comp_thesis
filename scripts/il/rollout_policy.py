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
took 8.5-12.7 s from the clutch, the perturbation demos 9-22 s). The server's action queue is reset at
every episode start.

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
demos by the same code (scripts/plots/plot_il.py). --no_traj switches this off.

Push test (--push): does the policy behave compliantly when the held cube is pushed? Every selected
episode runs as above until the cube has been held at the success height for --push_after_s and has been
still (speed below --push_still_speed) for --push_still_s, i.e. the lift is over (the success termination
is switched off in this mode). That is the push moment: a constant horizontal force of --push_force N
then acts on the cube for --push_s, is released, and the run continues for --release_s and ends there.
The direction is taken from the grip at that moment, with the same rule and names as the perturbations
in the demos (teleop_lift.py --perturb): along the normal of the cube face the thumb presses on (of the
cube's two horizontal axes, the one closest to the force on the thumb), "push" = toward the thumb (the
cube pressed into the thumb), "pull" = away from it (toward the index and middle finger); the force acts
at the cube's centre and stays fixed in the world frame. The push moment also needs the grasp at the
fingertips: the thumb and at least one of fingers 2-5 in contact (> 0.5 N); otherwise it waits (up to
--push_max_wait_s). (Before 2026-11-25: "away" / "into" from the summed force of fingers 2-5, i.e. tilted
by the ring finger's side-face contact.) The scene is saved at the push moment, and from there one of three
conditions acts:

    policy   the policy keeps acting (what is being tested); it simply continues the run
    law      reference: the policy is frozen and the admittance law is switched on, with q_ref set so
             that its equilibrium is the grip at that moment (q_ref = q_cmd - J+ e*, e* = f/K clamped,
             the controller started at e*): the operator's intention held still, the admittance acting
    frozen   reference: the last command is held, only the drives act (a policy that ignores K and f)

law and frozen start from the saved scene (joint and cube states, the force smoothing and the last
command restored), so all three start from the same grip. (Re-running the episode from its start for
each condition does not work: runs that should be identical drift apart after a few resets.) Contacts
are recomputed after a restore; the printed first-step force change shows whether that changed the grip.

Ideal expectations per fingertip in contact: law -> stiffness K_r = (1/K + 1/K_drv)^-1 and the command
yields by df/K; frozen -> K_drv, the command does not move; both spring back after the release. The
policy's place between the two references is the result. Episodes: --only, or --push_from with result
JSON files of earlier full runs (the first --episodes episodes that succeeded in all of them, e.g. the
K = 150 and K = 500 runs of the same seed). Per episode and direction the policy runs once up to the push:

    python scripts/il/rollout_policy.py --stiffness 150 --tag act_qcmd_v1 --push --headless \
        --push_from logs/il_eval/act_qcmd_v1_te0.01_K150_seed0.json logs/il_eval/act_qcmd_v1_te0.01_K500_seed0.json

Output: logs/il_eval/push/<tag>_<inference>_K<K>_seed<seed>_F<force>.json (settings, one entry per run
with the summary numbers) and .hdf5 with one group per run, /data/ep_NNN_<direction>_<condition>: obs
(T, 51), actions (T, 26) = [arm targets | the finger command the drives actually got, i.e. q_cmd also in
the law condition], q_ref (T, 20; law condition, else NaN), push (T, 3) world force of the step, phase (T,)
0 before / 1 push / 2 after, tip_force (T, 5) |filtered fingertip force|, tip_pos (T, 5, 3) fingertip
(tip link COM) and palm_pose / cube_pose (T, 7) in the world frame, all at the start of each step; attrs
episode, direction, condition, outcome, push_start, push_end, push_dir. The file attributes are those of
the rollout files, so scripts/tools/fingertip_kinematics.py works on it as on any rollout. Per run the
script prints the cube's displacement relative to the palm along the push (end of the push and the end
of the run, against the 0.5 s before the push), the grasp stiffness F / displacement, the change of the
thumb and fingers 2-5 forces, and their change over the first step after the push moment.
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
parser.add_argument("--push", action="store_true", help="push test in the hold (see the module docstring)")
parser.add_argument("--push_from", nargs="+", default=None, metavar="JSON",
                    help="push test: take the first --episodes episodes that succeeded in all these result files")
parser.add_argument("--push_dirs", nargs="+", choices=["push", "pull"], default=["push", "pull"],
                    help="push = toward the thumb, pull = away from it (as the perturbations in the demos)")
parser.add_argument("--push_conditions", nargs="+", choices=["policy", "law", "frozen"],
                    default=["policy", "law", "frozen"], help="what acts from the push on")
parser.add_argument("--push_force", type=float, default=2.0, help="push force on the cube [N] (cube weight 2.1 N)")
parser.add_argument("--push_after_s", type=float, default=1.0, help="push after the cube has been held this long [s]")
parser.add_argument("--push_still_s", type=float, default=0.5,
                    help="...and the cube has been still this long (the lift is over) [s]")
parser.add_argument("--push_still_speed", type=float, default=0.01,
                    help="'still': the cube's mean speed over --push_still_s below this [m/s]")
parser.add_argument("--push_max_wait_s", type=float, default=3.0,
                    help="push anyway when the cube has been held this much longer than --push_after_s without "
                    "being still (the mean speed at the push is recorded) [s]")
parser.add_argument("--push_s", type=float, default=2.0,
                    help="push duration [s] (with temporal ensembling the policy averages ~2 s of predictions)")
parser.add_argument("--release_s", type=float, default=2.0, help="keep running this long after the release [s]")
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

from isaaclab.utils.math import matrix_from_quat
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
    if args_cli.push:  # the push comes after the hold has started: the success must not end the episode
        env_cfg.terminations.success.params["hold_s"] = 1.0e6
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
    if args_cli.only is not None:
        episodes = args_cli.only
    elif args_cli.push and args_cli.push_from:
        ok_sets = []
        for path in args_cli.push_from:
            with open(path) as fh:
                ok_sets.append({r["episode"] for r in json.load(fh)["episodes"] if r["success"]})
        episodes = sorted(set.intersection(*ok_sets))[:args_cli.episodes]
        if not episodes:
            raise SystemExit("[rollout] no episode succeeded in all --push_from files")
    else:
        episodes = list(range(args_cli.episodes))
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
        "logs", "il_eval", *(["push"] if args_cli.push else []),
        f"{args_cli.tag}_{mode}_K{args_cli.stiffness:g}_seed{args_cli.seed}"
        f"{f'_F{args_cli.push_force:g}' if args_cli.push else ''}"
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
    ends = np.cumsum(dims).tolist()
    file_attrs = {
        "source": "policy push test" if args_cli.push else "policy rollout",
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
    }
    if args_cli.push:
        file_attrs["push"] = json.dumps({k: getattr(args_cli, k) for k in (
            "push_force", "push_after_s", "push_still_s", "push_still_speed", "push_max_wait_s", "push_s", "release_s",
            "push_dirs",
            "push_conditions")})
    if traj is not None:
        os.makedirs(os.path.dirname(os.path.abspath(traj)), exist_ok=True)
        with h5py.File(traj, "w") as f:
            f.create_group("data")
            f.attrs.update(file_attrs)

    if args_cli.push:
        push_test(env, uw, client, episodes, place_cube, hold, settle_steps, state_idx, env_idx, cube_z, height,
                  success_cfg.params["tolerance"], traj, out, video_dir, info, file_attrs)
        client.close()
        env.close()
        return

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
              f"max {d.max():.1f} s (compare with the demos' durations: inspect_demos.py)")
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


# ======================================================================================================
# push test
# ======================================================================================================

FINGER_TIPS = [f"right_finger{i}_tip_link" for i in range(1, 6)]
CONTACT_N = 0.5  # [N] smoothed fingertip force that counts as contact (as kswap_test.py and teleop_lift.py)
WINDOW_S = 0.5  # averaging window of the printed summary [s]


def push_test(env, uw, client, episodes, place_cube, hold, settle_steps, state_idx, env_idx, cube_z, height,
              tolerance, traj, out, video_dir, info, file_attrs):
    """The push test (see the module docstring). Per episode and direction: the policy runs until the push
    moment, the scene is saved there; the policy condition simply continues, the law and frozen conditions
    start from the saved state (restored), so all three start from the same grip."""
    dt = uw.step_dt
    robot, hand, cube = uw.scene["robot"], uw.action_manager.get_term("hand_action"), uw.scene["object"]
    ctrl = hand._controller
    dev = uw.device
    n_arm = len(ARM_JOINTS)
    tip_ids = robot.find_bodies(FINGER_TIPS, preserve_order=True)[0]
    palm_id = robot.find_bodies(PALM)[0][0]
    after_steps = max(1, int(round(args_cli.push_after_s / dt)))
    still_steps = max(1, int(round(args_cli.push_still_s / dt)))
    wait_steps = int(round(args_cli.push_max_wait_s / dt))
    push_steps = int(round(args_cli.push_s / dt))
    release_steps = int(round(args_cli.release_s / dt))
    win = int(round(WINDOW_S / dt))
    composer = getattr(cube, "permanent_wrench_composer", None)  # Isaac Lab >= 2.3.2; older: set_external_force...
    log_keys = ("obs", "actions", "q_ref", "push", "phase", "tip_force", "tip_pos", "palm_pose", "cube_pose")

    def set_push(force_w):
        """Constant force on the cube's centre, world frame (None: release). Set again every step: the wrench
        composer stores it in the cube's frame, so re-setting keeps its direction fixed in the world."""
        zero = torch.zeros(1, 1, 3, device=dev)
        if force_w is None:
            if composer is not None:
                composer.reset()
            else:
                cube.set_external_force_and_torque(zero, zero)
            return
        F = force_w.reshape(1, 1, 3).to(device=dev, dtype=torch.float32).contiguous()
        if composer is not None:
            composer.reset()  # also makes it read the cube's current orientation
            composer.set_forces_and_torques(forces=F, torques=zero, is_global=True)
        else:
            cube.set_external_force_and_torque(F, zero, is_global=True)

    def fingertip_grip():
        """The grasp is at the fingertips: the thumb and at least one of fingers 2-5 touch the cube."""
        f = hand.tip_forces_filtered[0].norm(dim=-1)
        return bool(f[0] > CONTACT_N) and bool((f[1:] > CONTACT_N).any())

    def push_direction(which):
        """World-frame unit vector, as teleop_lift.py's Perturber: the outward normal of the cube face the thumb
        presses on (of the cube's two horizontal axes, the one closest to the horizontal part of the force on the
        thumb, which points from the cube to the thumb); "push" along it (toward the thumb), "pull" against it."""
        f_thumb = hand.tip_forces_filtered[0, 0].clone()  # world frame
        f_thumb[2] = 0.0
        axes = matrix_from_quat(cube.data.root_link_quat_w)[0]  # columns = cube axes in the world
        vertical = int(torch.argmax(axes[2].abs()))
        best, n = -1.0, None
        for i in range(3):
            if i == vertical:
                continue
            c = float(axes[:, i] @ f_thumb)
            if abs(c) > best:
                best, n = abs(c), axes[:, i] * (1.0 if c >= 0 else -1.0)
        return n if which == "push" else -n

    def admittance_offsets(e):
        """dq = J+ e per finger, exactly as the action term computes it (e: (1, 5, 3) -> (1, 20))."""
        jac = robot.root_physx_view.get_jacobians()
        dq = torch.zeros_like(hand.joint_command)
        for i in range(5):
            J = jac[:, hand._jac_rows[i], :3, :][:, :, hand._jac_cols[i]]
            dq[:, hand._action_cols[i]] = hand._damped_pinv_times(J, e[:, i])
        return dq

    def start_law():
        """Admittance on, with q_ref and the controller state chosen so that its equilibrium is the grip now."""
        e = ctrl.filtered_force / hand.stiffness.reshape(1, 1, 1)  # static equilibrium e* = f / K
        norm = e.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        e = e * torch.clamp(ctrl.cfg.max_offset / norm, max=1.0)  # the controller's offset clamp
        ctrl._e[:] = e
        ctrl._e_dot.zero_()
        hand.cfg.admittance_enabled = True
        return hand.joint_command - admittance_offsets(e)  # q_cmd = q_ref + J+ e*  ->  q_ref

    def record(log, o, action_pre, phase, push_w, q_ref):
        """Log the state a step starts from (the action is appended after the step)."""
        log["obs"].append(o.astype(np.float32))
        log["push"].append((push_w if phase == 1 else torch.zeros(3, device=dev)).cpu().numpy().astype(np.float32))
        log["phase"].append(phase)
        log["tip_force"].append(hand.tip_forces_filtered[0].norm(dim=-1).cpu().numpy().astype(np.float32))
        log["tip_pos"].append(robot.data.body_com_pos_w[0, tip_ids].cpu().numpy().astype(np.float32))
        log["palm_pose"].append(torch.cat([robot.data.body_link_pos_w[0, palm_id],
                                           robot.data.body_link_quat_w[0, palm_id]]).cpu().numpy().astype(np.float32))
        log["cube_pose"].append(torch.cat([cube.data.root_link_pos_w[0],
                                           cube.data.root_link_quat_w[0]]).cpu().numpy().astype(np.float32))
        nan = torch.full((hand.joint_command.shape[1],), float("nan"), device=dev)
        log["q_ref"].append((q_ref[0] if q_ref is not None else nan).cpu().numpy().astype(np.float32))

    def step(log, action):
        obs, _, terminated, truncated, _ = env.step(action)
        # the command the drives actually got (= the policy's own in the policy and frozen conditions)
        log["actions"].append(torch.cat([action[0, :n_arm], hand.joint_command[0]]).cpu().numpy().astype(np.float32))
        ended = None
        if bool(terminated[0]) or bool(truncated[0]):
            ended = "time limit" if bool(truncated[0]) else "cube off the table"
        return obs, ended

    def until_push(ep, direction, video):
        """The policy from the episode start until the cube has been held at the success height for
        --push_after_s and its mean speed over the last --push_still_s is below --push_still_speed (a mean over
        a window, not the instantaneous speed, which jitters with the contacts), or at the latest
        --push_max_wait_s after that."""
        hand.cfg.admittance_enabled = False
        obs, _ = env.reset()
        set_push(None)
        place_cube(ep)
        for _ in range(settle_steps):
            obs, *_ = env.step(hold)
        client.call(cmd="reset")
        log = {k: [] for k in log_keys}
        held = max_held = 0
        min_speed = float("inf")
        last_action = None
        cube_path = []  # cube positions, for its mean speed over the last --push_still_s
        while True:
            o = obs["policy"][0].cpu().numpy().astype(float)
            held = held + 1 if o[cube_z] - CUBE_HALF_SIZE >= height - tolerance else 0
            cube_path.append(cube.data.root_link_pos_w[0].cpu().numpy().astype(float))
            speed = (float(np.linalg.norm(cube_path[-1] - cube_path[-1 - still_steps])) / (still_steps * dt)
                     if len(cube_path) > still_steps else float("inf"))
            max_held = max(max_held, held)
            if held >= after_steps:
                min_speed = min(min_speed, speed)
            grip = fingertip_grip()
            if held >= after_steps + wait_steps and not grip:
                return (f"no fingertip grip to push against (thumb and one of fingers 2-5) within "
                        f"{args_cli.push_max_wait_s:g} s"), None
            if held >= after_steps and grip and (speed < args_cli.push_still_speed or held >= after_steps + wait_steps):
                d = push_direction(direction)
                return None, {"obs": obs, "o": o, "log": log, "last_action": last_action, "d": d, "speed": speed,
                              "state": copy_state(uw.scene.get_state(is_relative=False)),
                              "f_filt": ctrl._f_filt.clone(), "joint_cmd": hand.joint_command.clone()}
            rep = client.call(cmd="act", state=o[state_idx].tolist(), env_state=o[env_idx].tolist())
            action = torch.tensor([rep["action"]], dtype=torch.float, device=dev)
            last_action = action
            if video is not None:
                video.add(uw.render())
            record(log, o, action, 0, None, None)
            obs, ended = step(log, action)
            if ended:
                return (f"{ended} before the push (cube at the success height for {max_held * dt:.1f} s at most, "
                        f"lowest mean speed there {1000 * min_speed:.1f} mm/s)"), None

    def copy_state(state):
        return {k: {n: {q: v.clone() for q, v in d.items()} for n, d in grp.items()} for k, grp in state.items()}

    def after_push(cond, snap, video):
        """From the push moment on, under one condition. Returns outcome, push start/end index, logs."""
        log = {k: list(v) for k, v in snap["log"].items()}  # the shared part before the push
        obs, o = snap["obs"], snap["o"]
        if cond != "policy":  # back to the saved push moment
            uw.reset_to(snap["state"], env_ids=None, is_relative=False)
            ctrl._f_filt[:] = snap["f_filt"]  # the force smoothing (the state's force and the law act on it)
            hand._joint_cmd[:] = snap["joint_cmd"]
        hand.cfg.admittance_enabled = False
        push_w = args_cli.push_force * snap["d"]
        t0 = len(log["obs"])
        q_ref, frozen = None, None
        if cond != "policy":
            frozen = snap["last_action"].clone()
            frozen[:, n_arm:] = snap["joint_cmd"]  # the command the drives got last
            if cond == "law":
                q_ref = start_law()
                frozen[:, n_arm:] = q_ref
        t_wall = time.perf_counter()
        for k in range(push_steps + release_steps):
            phase = 1 if k < push_steps else 2
            set_push(push_w if phase == 1 else None)
            if cond == "policy":
                rep = client.call(cmd="act", state=o[state_idx].tolist(), env_state=o[env_idx].tolist())
                action = torch.tensor([rep["action"]], dtype=torch.float, device=dev)
            else:
                action = frozen
            if video is not None:
                video.add(uw.render())
            record(log, o, action, phase, push_w, q_ref)
            obs, ended = step(log, action)
            o = obs["policy"][0].cpu().numpy().astype(float)
            if args_cli.realtime:
                wait = t_wall + (k + 1) * dt - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
            if ended:
                set_push(None)
                hand.cfg.admittance_enabled = False
                return ended + (" during the push" if phase == 1 else " after the release"), t0, None, log
        set_push(None)
        hand.cfg.admittance_enabled = False
        return "done", t0, t0 + push_steps, log

    def summary(arr, push_start, push_end, d):
        """Cube displacement relative to the palm along the push and force changes, end of the push and
        the end of the run against the WINDOW_S before the push; first-step force change (restore check)."""
        rel = (arr["cube_pose"][:, :3] - arr["palm_pose"][:, :3]) @ d
        fth, ffi = arr["tip_force"][:, 0], arr["tip_force"][:, 1:].sum(axis=1)
        pre = slice(max(0, push_start - win), push_start)
        push, after = slice(push_end - win, push_end), slice(-win, None)
        dx = [1000 * (rel[w].mean() - rel[pre].mean()) for w in (push, after)]
        return {"dx_push_mm": float(dx[0]), "dx_after_mm": float(dx[1]),
                "k_grasp": float(args_cli.push_force / (dx[0] / 1000)) if abs(dx[0]) > 0.05 else float("nan"),
                "f_pre": [float(fth[pre].mean()), float(ffi[pre].mean())],
                "df_push": [float(fth[push].mean() - fth[pre].mean()), float(ffi[push].mean() - ffi[pre].mean())],
                "df_after": [float(fth[after].mean() - fth[pre].mean()), float(ffi[after].mean() - ffi[pre].mean())],
                # force change over the first step after the push moment: similar in all conditions if the
                # restore reproduced the grip (the push itself changes little in 20 ms)
                "df_first_step": [float(fth[push_start + 1] - fth[push_start]),
                                  float(ffi[push_start + 1] - ffi[push_start])]}

    print(f"[push] {len(episodes)} episodes {episodes} at K = {args_cli.stiffness:g} N/m: push "
          f"{args_cli.push_force:g} N for {args_cli.push_s:g} s once the cube has been held {args_cli.push_after_s:g} s "
          f"and still {args_cli.push_still_s:g} s, {args_cli.release_s:g} s after the release; directions "
          f"{args_cli.push_dirs}, conditions {args_cli.push_conditions}\n")
    conds = ["policy"] + [c for c in args_cli.push_conditions if c != "policy"]  # the policy must go first
    results = []
    with torch.inference_mode():
        for ep in episodes:
            for direction in args_cli.push_dirs:
                videos = {c: VideoWriter(os.path.join(video_dir, f"ep{ep:03d}_{direction}_{c}.mp4"), fps=round(1.0 / dt))
                          for c in conds} if video_dir else {}
                failed, snap = until_push(ep, direction, videos.get("policy"))
                if failed:
                    print(f"[push] ep {ep:3d} {direction:<5} {failed}")
                    results.append({"episode": ep, "direction": direction, "condition": None, "outcome": failed})
                    for v in videos.values():
                        v.close()
                    if "before the push" in failed:
                        break  # the same in the other direction (identical until the push)
                    continue
                for cond in conds:
                    outcome, ps, pe, log = after_push(cond, snap, videos.get(cond))
                    arr = {k: np.stack(v) for k, v in log.items()}
                    d = snap["d"].cpu().numpy()
                    r = {"episode": ep, "direction": direction, "condition": cond, "outcome": outcome,
                         "push_start": ps, "push_end": pe, "push_dir": d.tolist(), "speed_at_push": snap["speed"]}
                    if outcome == "done":
                        r.update(summary(arr, ps, pe, d))
                        print(f"[push] ep {ep:3d} {direction:<5} {cond:<7} cube {r['dx_push_mm']:+6.2f} mm "
                              f"(k {r['k_grasp']:6.0f} N/m), after release {r['dx_after_mm']:+6.2f} mm | thumb "
                              f"{r['f_pre'][0]:5.2f} N {r['df_push'][0]:+5.2f} / {r['df_after'][0]:+5.2f} | "
                              f"fingers 2-5 {r['f_pre'][1]:5.2f} N {r['df_push'][1]:+5.2f} / {r['df_after'][1]:+5.2f}"
                              f" | first step {r['df_first_step'][0]:+5.2f} / {r['df_first_step'][1]:+5.2f} N"
                              f" | cube speed at the push {1000 * snap['speed']:.1f} mm/s")
                    else:
                        print(f"[push] ep {ep:3d} {direction:<5} {cond:<7} {outcome}")
                    results.append(r)
                    if cond in args_cli.push_conditions and traj is not None:
                        with h5py.File(traj, "a") as f:
                            g = f["data"].create_group(f"ep_{ep:03d}_{direction}_{cond}")
                            for k, v in arr.items():
                                g.create_dataset(k, data=v)
                            g.attrs.update({"episode": ep, "direction": direction, "condition": cond,
                                            "outcome": outcome, "push_start": ps, "push_end": -1 if pe is None else pe,
                                            "push_dir": json.dumps(r["push_dir"]), "speed_at_push": snap["speed"]})
                for v in videos.values():
                    v.close()
    results = [r for r in results if r["condition"] is None or r["condition"] in args_cli.push_conditions]

    # --- summary: medians per direction and condition ----------------------------------------------
    def med(rs, key, i=None):
        v = np.array([r[key] if i is None else r[key][i] for r in rs], dtype=float)
        return float(np.median(v[np.isfinite(v)])) if np.isfinite(v).any() else float("nan")

    print(f"\n[push] K = {args_cli.stiffness:g} N/m, push {args_cli.push_force:g} N: medians over the episodes "
          "(cube displacement relative to the palm along the push; force change thumb / fingers 2-5)")
    for direction in args_cli.push_dirs:
        for cond in args_cli.push_conditions:
            rs = [r for r in results if r["direction"] == direction and r["condition"] == cond and r["outcome"] == "done"]
            if not rs:
                continue
            print(f"[push] {direction:<5} {cond:<7} n {len(rs):3d} | push {med(rs, 'dx_push_mm'):+6.2f} mm, k "
                  f"{med(rs, 'k_grasp'):6.0f} N/m | after release {med(rs, 'dx_after_mm'):+6.2f} mm | df push "
                  f"{med(rs, 'df_push', 0):+5.2f} / {med(rs, 'df_push', 1):+5.2f} N, after {med(rs, 'df_after', 0):+5.2f} / "
                  f"{med(rs, 'df_after', 1):+5.2f} N")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"tag": args_cli.tag, "task": TASK, "stiffness": args_cli.stiffness, "seed": args_cli.seed,
                   "server": info, "push": json.loads(file_attrs["push"]), "episodes_selected": episodes,
                   "created": datetime.datetime.now().isoformat(timespec="seconds"), "runs": results}, fh, indent=1)
    print(f"[push] results -> {out}" + (f", trajectories -> {traj}" if traj else ""))


if __name__ == "__main__":
    main()
    simulation_app.close()
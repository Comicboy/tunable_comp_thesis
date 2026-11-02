"""Converts the teleoperated demo files (HDF5, from teleop_lift.py --record) into one local LeRobot dataset.

Run in the "lerobot" conda env (LeRobot 0.6.1, Python 3.12), not in the Isaac Lab env:

    python scripts/il/convert_to_lerobot.py logs/il_demos/lift_K150.hdf5 logs/il_demos/lift_K500.hdf5

Result: a LeRobot dataset (format v3.0) in --root (default logs/il_datasets/ur5wuji_lift), used by
lerobot-train with --dataset.repo_id=local/ur5wuji_lift --dataset.root=<root>. Nothing is uploaded.

What goes where (decided 2026-10-31). Our 51-number state is split by its terms (the file's obs_terms):

    observation.state              (48)  arm q (6), arm q_dot (6), hand q (20), fingertip forces (15),
                                         log10 K (1), in this order
    observation.environment_state  (3)   cube position (base frame)
    action                         (26)  [arm joint targets (6) | finger joint commands q_cmd (20)], absolute [rad]
                                         (q_cmd = what the admittance sent to the drives while recording: the
                                         policy learns to perform the admittance itself)

LeRobot's ACT turns each of the two observation groups into one transformer token, and it needs an
environment state when there are no camera images. The stiffness is part of the robot's state: it
is a setting of the robot's own controller. LeRobot's dataset features are named by these keys:
"observation.environment_state" is its "environment" type, any other "observation.*" key is
"robot state".

Every demo becomes one episode, 50 fps (our env step), one task string for all. All demos of all
files go into the same dataset; which episode came from which file and demo is printed and written
to <root>/source_episodes.json. The normalisation statistics (mean, std, min, max per number) are
computed by LeRobot while saving.

At the end the dataset is loaded back and compared with the HDF5 files sample by sample.
"""

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

parser = argparse.ArgumentParser(description="Convert IL demo HDF5 files into a local LeRobot dataset.")
parser.add_argument("files", nargs="+", help="HDF5 demo files (teleop_lift.py --record)")
parser.add_argument("--root", default="logs/il_datasets/ur5wuji_lift", help="output folder of the dataset")
parser.add_argument("--repo_id", default="local/ur5wuji_lift", help="dataset name (only a label; nothing is uploaded)")
parser.add_argument("--task", default="Lift the cube 15 cm above the table and hold it there",
                    help="task string stored with every frame")
parser.add_argument("--overwrite", action="store_true", help="delete --root first if it exists")
args = parser.parse_args()

ENV_TERMS = ("cube_position",)  # -> observation.environment_state; every other term -> observation.state
MUST_MATCH = ("task", "obs_dim", "action_dim", "step_dt", "obs_terms", "arm_joints", "hand_joints", "format")


def term_names(term: str, size: int, attrs: dict) -> list[str]:
    """Readable names of the numbers of one state term (stored in the dataset's metadata)."""
    arm, hand = json.loads(attrs["arm_joints"]), json.loads(attrs["hand_joints"])
    if term == "arm_joint_pos":
        return [f"{j}.pos" for j in arm]
    if term == "arm_joint_vel":
        return [f"{j}.vel" for j in arm]
    if term == "hand_joint_pos":
        return [f"{j}.pos" for j in hand]
    if term == "fingertip_forces":  # thumb first: finger1..finger5, (fx, fy, fz) each, base frame
        return [f"finger{i}.force_{a}" for i in range(1, 6) for a in "xyz"]
    if term == "cube_position":
        return ["cube.x", "cube.y", "cube.z"]
    if term == "stiffness":
        return ["log10_stiffness"]
    return [f"{term}.{i}" for i in range(size)]


def main():
    # --- read the files' settings and check they belong together ----------------------------------
    metas = []
    for path in args.files:
        with h5py.File(path, "r") as f:
            metas.append(dict(f.attrs))
    for path, m in zip(args.files[1:], metas[1:]):
        for k in MUST_MATCH:
            mk, m0 = m.get(k), metas[0].get(k)
            same = np.isclose(mk, m0) if isinstance(mk, float) and isinstance(m0, float) else mk == m0
            if not same:
                raise SystemExit(f"{path}: {k} = {mk} differs from {args.files[0]} ({m0})")
    a0 = metas[0]
    fps = int(round(1.0 / float(a0["step_dt"])))

    # --- which numbers of our state go into which LeRobot feature --------------------------------
    terms = json.loads(a0["obs_terms"])  # [[name, start, end], ...]
    state_idx, state_names, env_idx, env_names = [], [], [], []
    for name, s, e in terms:
        idx, names = list(range(s, e)), term_names(name, e - s, a0)
        if name in ENV_TERMS:
            env_idx += idx
            env_names += names
        else:
            state_idx += idx
            state_names += names
    if "q_cmd" not in str(a0.get("format", "")):
        raise SystemExit(f"{args.files[0]}: not a demo file with q_cmd as the action (recorded before that change?)")
    action_names = [f"{j}.target" for j in json.loads(a0["arm_joints"])] + \
                   [f"{j}.q_cmd" for j in json.loads(a0["hand_joints"])]
    assert len(state_idx) + len(env_idx) == int(a0["obs_dim"]) and len(action_names) == int(a0["action_dim"])
    print(f"[convert] observation.state: {len(state_idx)} numbers ({', '.join(t[0] for t in terms if t[0] not in ENV_TERMS)})")
    print(f"[convert] observation.environment_state: {len(env_idx)} numbers ({', '.join(ENV_TERMS)})")
    print(f"[convert] action: {len(action_names)} numbers (arm targets | finger commands q_cmd), {fps} fps")

    features = {
        "observation.state": {"dtype": "float32", "shape": (len(state_idx),), "names": state_names},
        "observation.environment_state": {"dtype": "float32", "shape": (len(env_idx),), "names": env_names},
        "action": {"dtype": "float32", "shape": (len(action_names),), "names": action_names},
    }

    root = Path(args.root)
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} exists; use --overwrite to replace it")
        shutil.rmtree(root)
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=fps, features=features, root=root,
                               robot_type="ur5_wuji", use_videos=False)

    # --- one episode per demo ---------------------------------------------------------------------
    sources = []  # episode index -> (file, demo, stiffness, samples)
    for path in args.files:
        with h5py.File(path, "r") as f:
            k = float(f.attrs["stiffness"])
            for demo in sorted(f["data"].keys()):
                obs = f["data"][demo]["obs"][()].astype(np.float32)
                act = f["data"][demo]["actions"][()].astype(np.float32)
                for t in range(len(obs)):
                    ds.add_frame({
                        "observation.state": obs[t, state_idx],
                        "observation.environment_state": obs[t, env_idx],
                        "action": act[t],
                        "task": args.task,
                    })
                ds.save_episode()
                sources.append({"episode": len(sources), "file": path, "demo": demo, "stiffness": k,
                                "samples": len(obs)})
                print(f"[convert] episode {len(sources) - 1:3d} <- {path}:{demo} (K = {k:g}, {len(obs)} samples)")
    ds.finalize()
    with open(root / "source_episodes.json", "w") as fh:
        json.dump(sources, fh, indent=1)

    # --- load it back and compare with the HDF5 files ----------------------------------------------
    check = LeRobotDataset(args.repo_id, root=root)
    n_frames = sum(s["samples"] for s in sources)
    print(f"\n[check] loaded back: {check.num_episodes} episodes, {check.num_frames} frames "
          f"(expected {len(sources)}, {n_frames})")
    ok = check.num_episodes == len(sources) and check.num_frames == n_frames
    worst = 0.0
    i = 0
    for s in sources:
        with h5py.File(s["file"], "r") as f:
            obs = f["data"][s["demo"]]["obs"][()].astype(np.float32)
            act = f["data"][s["demo"]]["actions"][()].astype(np.float32)
        for t in (0, len(obs) // 2, len(obs) - 1):  # first, middle and last sample of every episode
            item = check[i + t]
            worst = max(worst,
                        np.abs(item["observation.state"].numpy() - obs[t, state_idx]).max(),
                        np.abs(item["observation.environment_state"].numpy() - obs[t, env_idx]).max(),
                        np.abs(item["action"].numpy() - act[t]).max())
            ok &= int(item["episode_index"]) == s["episode"] and int(item["frame_index"]) == t
        i += len(obs)
    ok &= worst == 0.0
    print(f"[check] largest difference to the HDF5 data: {worst:.2e}")
    st = check.meta.stats
    k_mean, k_std = float(st["observation.state"]["mean"][-1]), float(st["observation.state"]["std"][-1])
    print(f"[check] stats of log10 K (last state number): mean {k_mean:.3f}, std {k_std:.3f}")
    print(f"\n[convert] {'DATASET OK' if ok else 'CHECK FAILED'}: {root.resolve()}")


if __name__ == "__main__":
    main()
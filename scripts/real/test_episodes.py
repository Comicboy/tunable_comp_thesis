#!/usr/bin/env python3
"""Offline checks of tunable_comp_real.episodes (no robot, no hand, no glove, no Quest).

    python scripts/real/test_episodes.py

  1. SuccessDetector: no success without a grasp, with the thumb alone or the fingers alone, or
     below 15 cm; success exactly after 3 s grasped and >= 15 cm up; a short glove dropout does
     not break the hold, letting go resets the grasp reference;
  2. draw_stiffness: a K that has all its demos is never drawn; balanced otherwise;
  3. the recorders (the supervisor's EpisodeRecorder + ours, real h5py, files in a temp folder):
     one file per K, the total count shown instead of the per-file one, pending attributes
     applied, delete_last undoing saves in reverse order (and the next demo reusing the name),
     files of the other split or another K refused.
Prints PASS / FAIL per check; exit code 1 if any fails.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np

from quest_teleop2.config import TaskCfg
from quest_teleop2.data import schema
from tunable_comp_real.episodes import (FORMAT, DeletableRecorder, StiffnessRecorders, SuccessDetector,
                                        SuccessParams, demo_path, draw_stiffness)

failures = 0


def check(ok: bool, text: str) -> None:
    global failures
    failures += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}")


# ------------------------------------------------------------------------------------------------ #

def success() -> None:
    print("1. SuccessDetector (15 cm for 3 s, contact threshold 0.5)")
    dt = 0.008
    grip = np.array([1.0, 1.0, 1.0, 0.0, 0.0])

    def run(det, seconds, force, z_from, z_to=None, t0=0.0):
        """Ticks for `seconds`, the TCP moving linearly from z_from to z_to; returns the end time
        and the time of the first success (None if none)."""
        n = int(round(seconds / dt))
        z_to = z_from if z_to is None else z_to
        hit = None
        for i in range(n):
            t = t0 + (i + 1) * dt
            if det.update(force, z_from + (z_to - z_from) * (i + 1) / n, t) and hit is None:
                hit = t
        return t0 + n * dt, hit

    det = SuccessDetector()
    t, hit = run(det, 5.0, np.zeros(5), 0.30, 0.50)
    check(hit is None and not det.grasped, "no contact, TCP raised 20 cm: no grasp, no success")
    det.reset()
    t, hit = run(det, 5.0, np.array([1.0, 0, 0, 0, 0]), 0.30, 0.50)
    check(hit is None and not det.grasped, "thumb alone: no grasp")
    det.reset()
    t, hit = run(det, 5.0, np.array([0, 1.0, 1.0, 1.0, 1.0]), 0.30, 0.50)
    check(hit is None and not det.grasped, "four fingers without the thumb: no grasp")

    det.reset()
    t, _ = run(det, 0.5, grip, 0.30)                       # grasp forms on the table at z = 0.30
    check(det.grasped and abs(det.z0 - 0.30) < 1e-9, "grasp formed: reference height = TCP height then")
    t, _ = run(det, 1.0, grip, 0.30, 0.44, t)              # lift 14 cm
    t, hit = run(det, 5.0, grip, 0.44, t0=t)
    check(hit is None and abs(det.lift - 0.14) < 1e-9, f"held at 14 cm for 5 s: no success (lift {det.lift:.3f} m)")
    t_up = t                                               # 15 cm is crossed from here on:
    t, _ = run(det, dt, grip, 0.46, t0=t)                  # to 16 cm in one tick
    t, hit = run(det, 2.9, grip, 0.46, t0=t)
    check(hit is None, "16 cm for 2.9 s: not yet")
    t, hit = run(det, 0.2, grip, 0.46, t0=t)
    check(hit is not None and abs((hit - t_up) - 3.0) < 2 * dt, f"success after {hit - t_up:.3f} s at 16 cm")

    det.reset()
    t, _ = run(det, 0.5, grip, 0.30)
    t, _ = run(det, 0.5, grip, 0.30, 0.46, t)
    t, _ = run(det, 1.5, grip, 0.46, t0=t)
    t, _ = run(det, 0.1, np.zeros(5), 0.46, t0=t)           # glove drops contact for 0.1 s
    t, hit = run(det, 1.6, grip, 0.46, t0=t)
    check(hit is not None, "a 0.1 s contact dropout does not break the 3 s hold")
    det.reset()
    t, _ = run(det, 0.5, grip, 0.30)
    t, _ = run(det, 0.5, np.zeros(5), 0.30, t0=t)           # let go for 0.5 s
    check(not det.grasped, "let go for 0.5 s: grasp lost")
    t, _ = run(det, 0.5, grip, 0.40, t0=t)                  # grasp again 10 cm higher
    check(abs(det.z0 - 0.40) < 1e-9 and abs(det.lift) < 1e-9, "a new grasp sets a new reference height")
    check("K" not in det.status() and "cm" in det.status(), f"status line: {det.status()!r}")


def draw() -> None:
    print("2. draw_stiffness")
    rng = np.random.default_rng(0)
    ks = [150.0, 500.0]
    picks = [draw_stiffness(ks, {150.0: 0, 500.0: 4}, rng) for _ in range(200)]
    check(all(k == 500.0 for k in picks), "a K with all its demos is never drawn")
    picks = np.array([draw_stiffness(ks, {150.0: 10, 500.0: 10}, rng) for _ in range(2000)])
    share = float(np.mean(picks == 150.0))
    check(0.45 < share < 0.55, f"equal demand: K = 150 drawn {100 * share:.1f} % of the time")
    picks = np.array([draw_stiffness(ks, None, rng) for _ in range(2000)])
    check(0.45 < float(np.mean(picks == 150.0)) < 0.55, "no targets: uniform")


# ------------------------------------------------------------------------------------------------ #

def fake_tick(task) -> tuple[dict, np.ndarray, dict]:
    """One recorded tick with the schema's obs keys and one of ours."""
    obs = {}
    for key in schema.OBS_KEYS:
        if key.endswith("onehot"):
            obs[key] = task.one_hot(key.split("_")[0], 0)
        elif key == "gripper":
            obs[key] = np.zeros(schema.GRIPPER_DIM, np.float32)
        else:
            obs[key] = np.zeros(schema.N_JOINTS, np.float32)
    obs["fingertip_force"] = np.ones(5, np.float32)
    extra = {"hand_q_cmd_ticks": np.zeros((5, 20), np.float32)}
    return obs, np.zeros(schema.ACTION_DIM, np.float32), extra


def recorders() -> None:
    print("3. recorders: one file per K, delete last, refusing mixed files")
    task = TaskCfg(name="real_lift", starts=["home"], goals=["lift"])
    kw = dict(action_hz=25.0, more_obs_keys=("fingertip_force",), more_extra_keys=("hand_q_cmd_ticks",))
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "demos"

        def make(split="train", ks=(150.0, 500.0)):
            return StiffnessRecorders({k: DeletableRecorder(
                demo_path(folder, k), task, {"format": FORMAT, "split": split, "stiffness": k,
                                             "task": "real_lift"}, **kw) for k in ks}, target_per_k=2)

        def episode(rec, k, frames=12, attrs=None):
            rec.select(k)
            rec.pending = dict(attrs or {})
            rec.start_episode(0, 0)
            for _ in range(frames):
                rec.add(*fake_tick(task))
            return rec.save_episode(True)

        rec = make()
        n0 = episode(rec, 150.0, attrs={"cube_spot": "A"})
        n1 = episode(rec, 500.0)
        n2 = episode(rec, 150.0)
        check((n0, n1, n2) == (0, 1, 2), f"save returns a running number over all K: {(n0, n1, n2)}")
        check(rec.counts == {150.0: 2, 500.0: 1} and rec.n_saved == 3, f"counts per K {rec.counts}, total {rec.n_saved}")
        check(rec.missing == {150.0: 0, 500.0: 1} and not rec.targets_reached, f"missing {rec.missing}")
        with h5py.File(demo_path(folder, 150.0), "r") as f:
            names = sorted(f["data"])
            a = f["data"].attrs
            ok = (names == ["demo_0", "demo_1"] and float(a["stiffness"]) == 150.0
                  and str(a["split"]) == "train" and int(a["total"]) == 24)
            pending = f["data/demo_0"].attrs.get("cube_spot")
            shape = f["data/demo_0/extra/hand_q_cmd_ticks"].shape
        check(ok, f"K = 150 file: {names}, stiffness and split stored, total 24 frames")
        check(str(pending) == "A", "pending attributes applied at episode start")
        check(shape == (12, 5, 20), f"per-tick extra stored with its shape {shape}")

        check(rec.delete_last() == (150.0, 1), "d: the last save (K = 150, demo_1) deleted")
        check(rec.delete_last() == (500.0, 0), "d again: the one before (K = 500, demo_0)")
        check(rec.counts == {150.0: 1, 500.0: 0}, f"counts after deleting: {rec.counts}")
        with h5py.File(demo_path(folder, 150.0), "r") as f:
            check(sorted(f["data"]) == ["demo_0"] and int(f["data"].attrs["total"]) == 12,
                  "K = 150 file: demo_1 gone, total frames corrected")
        episode(rec, 150.0)
        with h5py.File(demo_path(folder, 150.0), "r") as f:
            check(sorted(f["data"]) == ["demo_0", "demo_1"], "the next demo reuses the freed name demo_1")
        rec.select(500.0)
        rec.start_episode(0, 0)
        rec.add(*fake_tick(task))
        rec.discard_episode()
        check(rec.counts[500.0] == 0 and not rec.is_recording, "discard writes nothing")
        check(rec.delete_last() == (150.0, 1) and rec.delete_last() == (150.0, 0) and rec.delete_last() is None,
              "deleting past the session's first save: nothing left to delete")

        episode(make(), 150.0)
        try:
            make(split="val")
            check(False, "a val run on a train file is refused")
        except ValueError as exc:
            check("split" in str(exc), f"a val run on a train file is refused ({exc})")
        try:
            DeletableRecorder(demo_path(folder, 150.0), task, {"format": FORMAT, "split": "train",
                                                               "stiffness": 500.0, "task": "real_lift"}, **kw)
            check(False, "another K on the same file is refused")
        except ValueError as exc:
            check("stiffness" in str(exc), "another K on the same file is refused")


def main() -> int:
    success()
    draw()
    recorders()
    print(f"\n{'all checks passed' if failures == 0 else f'{failures} check(s) FAILED'}")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
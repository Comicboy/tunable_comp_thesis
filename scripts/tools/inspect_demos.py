"""Summarises and checks a demo file written by scripts/teleop/teleop_lift.py --record.

No Isaac Sim needed (h5py + numpy only), so it also runs in a training environment.

For every demo it prints the length, the real-time factor, the idle time before the clutch, where
the cube started, how high it got, the largest fingertip force and how closely the arm followed
its targets, and it checks that:

  * obs and actions have the same number of samples, the sizes in the file's attributes, no NaN/inf;
  * the stiffness in the state (log10 K) is the file's K in every sample;
  * the demo ends with the success hold: in the last hold_s of samples the cube's bottom face is
    at or above height - tolerance (the samples are the states the steps started from, so the
    first sample of the hold is not included; the check uses hold_s - 1 step);
  * the cube starts at rest on the table.

    python scripts/tools/inspect_demos.py logs/il_demos/lift_K150.hdf5
    python scripts/tools/inspect_demos.py logs/il_demos/lift_K150.hdf5 logs/il_demos/lift_K500.hdf5
"""

import argparse
import json
import math

import h5py
import numpy as np

parser = argparse.ArgumentParser(description="Summarise and check IL demo files.")
parser.add_argument("files", nargs="+", help="HDF5 demo files")
args = parser.parse_args()


def inspect(path: str) -> bool:
    ok_all = True
    with h5py.File(path, "r") as f:
        a = dict(f.attrs)
        print(f"\n=== {path}")
        for k in ("task", "stiffness", "controller", "device", "physics_dt", "decimation", "step_dt", "lp_alpha",
                  "ik_point", "ik_damping", "max_joint_step", "no_rotation", "success", "created"):
            print(f"  {k:<15} {a.get(k)}")
        sl = {name: slice(s, e) for name, s, e in json.loads(a["obs_terms"])}
        succ = json.loads(a["success"])
        dt, k_file = float(a["step_dt"]), float(a["stiffness"])
        n_hold = int(round(succ["hold_s"] / dt)) - 1
        thresh = succ["height"] - succ["tolerance"]
        k_log = math.log10(min(k_file, 1.0e4))

        demos = sorted(f["data"].keys())
        print(f"  {len(demos)} demos\n")
        print(f"  {'demo':<9} {'T':>5} {'dur s':>6} {'rtf':>5} {'idle s':>6} {'cube start (x, y)':>18} "
              f"{'max lift cm':>11} {'max sum|f| N':>12} {'arm err mrad':>12}  checks")
        durations = []
        for name in demos:
            g = f["data"][name]
            obs, act = g["obs"][()], g["actions"][()]
            ga = dict(g.attrs)
            fails = []
            if obs.shape[0] != act.shape[0]:
                fails.append("T(obs) != T(actions)")
            if obs.shape[1] != a["obs_dim"] or act.shape[1] != a["action_dim"]:
                fails.append(f"sizes {obs.shape[1]}/{act.shape[1]}")
            if not (np.isfinite(obs).all() and np.isfinite(act).all()):
                fails.append("NaN/inf")
            if np.abs(obs[:, sl["stiffness"]] - k_log).max() > 1e-4:
                fails.append("log10 K != file K")
            cube = obs[:, sl["cube_position"]]
            bottom = cube[:, 2] - succ["half_size"]
            if len(bottom) < n_hold or (bottom[-n_hold:] < thresh).any():
                fails.append("no success hold at the end")
            if abs(bottom[0]) > 0.003:
                fails.append(f"cube not on the table at the start ({1000 * bottom[0]:.1f} mm)")
            forces = obs[:, sl["fingertip_forces"]].reshape(len(obs), 5, 3)
            f_max = np.linalg.norm(forces, axis=-1).sum(-1).max()
            # how closely the arm followed: target of step t vs angle after it (= state t+1)
            arm_q = obs[:, sl["arm_joint_pos"]]
            arm_err = np.abs(act[:-1, :6] - arm_q[1:]).mean() if len(obs) > 1 else float("nan")
            T = obs.shape[0]
            durations.append(T * dt)
            ok = not fails
            ok_all &= ok
            print(f"  {name:<9} {T:>5} {T * dt:>6.1f} {ga.get('rtf', float('nan')):>5.2f} {ga.get('idle_s', float('nan')):>6.1f} "
                  f"({cube[0, 0]:+.3f}, {cube[0, 1]:+.3f}) {100 * bottom.max():>11.1f} {f_max:>12.1f} "
                  f"{1000 * arm_err:>12.1f}  {'OK' if ok else 'FAIL: ' + '; '.join(fails)}")
        if durations:
            d = np.array(durations)
            print(f"\n  total {d.sum():.1f} s ({d.sum() / dt:.0f} samples); duration mean {d.mean():.1f} s, "
                  f"min {d.min():.1f} s, max {d.max():.1f} s")
    return ok_all


all_ok = True
for p in args.files:
    all_ok &= inspect(p)
print(f"\n{'ALL DEMOS OK' if all_ok else 'SOME DEMOS FAILED A CHECK'}")
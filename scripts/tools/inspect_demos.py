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

The action is [arm targets | finger commands q_cmd] (q_cmd: what the admittance sent to the drives
while recording, from the operator's references q_ref; also stored: q_ref, dq, tip_ref_offset,
tip_force_raw). Its admittance part is checked too:
  * q_cmd = q_ref + dq, except where q_cmd was clamped to a joint limit (the share of clamped
    entries is shown);
  * dq = 0 before the first contact (the admittance offset starts at zero);
  * the extra datasets have one row per sample;
and two numbers are shown: the largest |dq| (how far the admittance moved the fingers away from the
operator's references) and, in contact, |dq| per newton of fingertip force (median over fingers 2-5
and contact samples) [mrad/N]. At steady state e = f / K, so this should be clearly larger at
K = 150 than at K = 500 (ideally ~500/150 = 3.3 times, less in practice: the joint posture, the
admittance's dynamics and the clamp all enter). With two files the ratio is printed at the end.

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
CONTACT_N = 0.5  # [N] per fingertip: counts as contact for the |dq| / |f| number
dq_per_f = {}  # file -> median |dq| / |f| [rad/N]


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
        if "q_cmd" not in str(a.get("format", "")):
            print("  FAIL: not a demo file with q_cmd as the action (recorded before that change?)")
            return False
        v2 = True  # (kept as a switch for the extra columns)
        hand = json.loads(a["hand_joints"])
        n_arm = len(json.loads(a["arm_joints"]))
        finger_idx = [[i for i, n in enumerate(hand) if n.startswith(f"right_finger{k}_")] for k in range(1, 6)]
        print(f"  format          {a.get('format')}")
        print(f"  {len(demos)} demos\n")
        print(f"  {'demo':<9} {'T':>5} {'dur s':>6} {'rtf':>5} {'idle s':>6} {'cube start (x, y)':>18} "
              f"{'max lift cm':>11} {'max sum|f| N':>12} {'arm err mrad':>12}"
              + (f" {'max|dq| mrad':>12} {'dq/f mrad/N':>11} {'clamped %':>9}" if v2 else "") + "  checks")
        durations, ratios = [], []
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
            extra = ""
            if v2:
                q_cmd, q_ref, dq = act[:, n_arm:], g["q_ref"][()], g["dq"][()]
                for key in ("q_ref", "dq", "tip_ref_offset", "tip_force_raw"):
                    if key not in g or g[key][()].shape[0] != T:
                        fails.append(f"{key} missing or wrong length")
                if not fails:
                    diff = np.abs(q_cmd - (q_ref + dq))
                    clamped = 100.0 * (diff > 1e-4).mean()
                    f_tip = np.linalg.norm(forces, axis=-1)  # (T, 5) smoothed, as the admittance sees it
                    # first contact: the raw force is sampled only at the last physics step of each env step,
                    # so a brief touch in the step's first physics step is missed there; the smoothed force in
                    # the next state (obs row t + 1 = after step t) still shows it
                    raw = np.linalg.norm(g["tip_force_raw"][()], axis=-1).sum(-1)  # (T,) same rows as dq
                    smooth_next = np.append(f_tip.sum(-1)[1:], 0.0)  # (T,) smoothed force after step t
                    touch = np.nonzero((raw > 0.0) | (smooth_next > 0.0))[0]
                    first = int(touch[0]) if touch.size else T
                    if first > 0 and np.abs(dq[:first]).max() > 1e-6:
                        fails.append(f"dq != 0 before the first contact ({np.abs(dq[:first]).max():.1e} rad)")
                    r = []
                    for k_ in range(1, 5):  # fingers 2-5
                        c = f_tip[:, k_] > CONTACT_N
                        if c.sum() >= 5:
                            r.append(np.median(np.linalg.norm(dq[c][:, finger_idx[k_]], axis=-1) / f_tip[c, k_]))
                    ratio = float(np.median(r)) if r else float("nan")
                    if r:
                        ratios.append(ratio)
                    extra = f" {1000 * np.abs(dq).max():>12.0f} {1000 * ratio:>11.1f} {clamped:>9.1f}"
            ok = not fails
            ok_all &= ok
            print(f"  {name:<9} {T:>5} {T * dt:>6.1f} {ga.get('rtf', float('nan')):>5.2f} {ga.get('idle_s', float('nan')):>6.1f} "
                  f"({cube[0, 0]:+.3f}, {cube[0, 1]:+.3f}) {100 * bottom.max():>11.1f} {f_max:>12.1f} "
                  f"{1000 * arm_err:>12.1f}{extra}  {'OK' if ok else 'FAIL: ' + '; '.join(fails)}")
        if durations:
            d = np.array(durations)
            print(f"\n  total {d.sum():.1f} s ({d.sum() / dt:.0f} samples); duration mean {d.mean():.1f} s, "
                  f"min {d.min():.1f} s, max {d.max():.1f} s")
        if ratios:
            dq_per_f[(path, k_file)] = float(np.median(ratios))
            print(f"  admittance in the action: median |dq| / |f| = {1000 * dq_per_f[(path, k_file)]:.1f} mrad/N "
                  f"(fingers 2-5, in contact)")
    return ok_all


all_ok = True
for p in args.files:
    all_ok &= inspect(p)
ks = sorted({k for _, k in dq_per_f})
if len(ks) == 2:
    lo = np.median([v for (p_, k), v in dq_per_f.items() if k == ks[0]])
    hi = np.median([v for (p_, k), v in dq_per_f.items() if k == ks[1]])
    print(f"\n|dq| / |f| at K = {ks[0]:g} vs K = {ks[1]:g}: {1000 * lo:.1f} vs {1000 * hi:.1f} mrad/N, ratio "
          f"{lo / hi:.2f} (the admittance alone would give {ks[1] / ks[0]:.2f} at steady state)")
print(f"\n{'ALL DEMOS OK' if all_ok else 'SOME DEMOS FAILED A CHECK'}")
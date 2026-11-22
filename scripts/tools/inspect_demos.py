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

Perturbation demos (teleop_lift.py --perturb; the file has push_force): additionally, for every
perturbation (a run of samples with a force on the cube), how the grasp responded. Windows (sample t
of push_force = the force acting during step t; force and fingertip position after step t = obs[t + 1]
and tip_ref_offset[t]): "before" = after the step before the onset, "end" = after the last step with
the force, "after" = --settle_s after the release (if the demo lasts that long and no new one has
started). Per finger:

  df     change of the fingertip force |f| (the smoothed force in the state) from before to end [N]
  dx     fingertip displacement along the contact normal n (direction of the force on the fingertip
         before the perturbation; + = away from the cube, i.e. yielding) [mm], from
         tip_ref_offset = J (q_ref - q): with q_ref unchanged, dx = -(change of tip_ref_offset) . n
  dx/df  the fingertip's compliance during the perturbation [mm/N], for fingers in contact with
         |df| > 0.3 N; a steady-state admittance in series with the drives gives 1/K + 1/K_drv
         (K_drv ~ 215 N/m on fingers 2-5: 11.3 mm/N at K = 150, 6.7 mm/N at K = 500; admittance alone
         6.7 / 2.0), less if the response has not settled within the 0.2-0.3 s
  back   share of dx still there --settle_s after the release (0 = back to the grip, 1 = stayed)

and for the whole perturbation: the largest change of the operator's q_ref and of the arm targets
from the onset to the end of the "after" window [mrad], with the same measure over an equally long
window just before the onset ("before": tremor and drift of the teleoperation); a reaction of the
operator shows as a change clearly above that (more than --react_mrad above it is flagged); push / pull
are told apart by the force on the thumb at the onset (push = toward the thumb); and the share of finger joints of fingers 2-5 whose drive torque
k_p (q_cmd - q) reached its effort limit in that window (approximate: the drive gains and limits
below, damping ignored). --perturb_detail prints every perturbation; the summary per file gives the
medians, push (toward the thumb) and pull (away) separately.

    python scripts/tools/inspect_demos.py logs/il_demos_perturb_test/lift_K150.hdf5 logs/il_demos_perturb_test/lift_K500.hdf5 --perturb_detail
"""

import argparse
import json
import math

import h5py
import numpy as np

parser = argparse.ArgumentParser(description="Summarise and check IL demo files.")
parser.add_argument("files", nargs="+", help="HDF5 demo files")
parser.add_argument("--perturb_detail", action="store_true", help="print every perturbation (perturbation demos)")
parser.add_argument("--settle_s", type=float, default=0.4, help="the 'after' window: this long after a release [s]")
parser.add_argument("--react_mrad", type=float, default=50.0,
                    help="a q_ref change this much above the one before the onset is flagged as a reaction [mrad]")
args = parser.parse_args()
CONTACT_N = 0.5  # [N] per fingertip: counts as contact for the |dq| / |f| number
DF_MIN = 0.3  # [N] force change below this: no compliance number for that finger
FINGERS = ("thumb", "index", "middle", "ring", "little")
# hand drives (status doc section 4, read with fingertip_kinematics.py): k_p per joint 1..4 [Nm/rad] and
# effort limits [Nm] of the thumb and of fingers 2-5 (approximate: 0.62-0.65 / 0.18 / 0.21-0.24 / 0.19-0.22)
KP = {1: 2.0, 2: 2.0, 3: 1.0, 4: 0.8}
TAU_MAX = {"thumb": {1: 0.45, 2: 0.43, 3: 0.19, 4: 0.15}, "finger": {1: 0.63, 2: 0.18, 3: 0.22, 4: 0.20}}
K_DRV = 215.0  # [N/m] median drive stiffness at the fingertips of fingers 2-5 along the contact force (IL tables)
dq_per_f = {}  # file -> median |dq| / |f| [rad/N]
compliance = {}  # file K -> list of dx/df of fingers 2-5 [m/N]


def perturbations(push: np.ndarray) -> list[tuple[int, int]]:
    """(first row, first row after) of every run of rows with a force on the cube."""
    on = np.linalg.norm(push, axis=-1) > 1e-6
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def perturbation_report(name, obs, act, g, sl, hand, n_arm, k_file, dt, rows):
    """Response of the grasp to each perturbation of one demo; appends one dict per perturbation to rows."""
    push = g["push_force"][()]
    tro = g["tip_ref_offset"][()]  # (T, 5, 3) J (q_ref - q), logged in step t
    q_ref = g["q_ref"][()]
    T = len(obs)
    f_vec = obs[:, sl["fingertip_forces"]].reshape(T, 5, 3)  # state t: smoothed, before step t
    q_hand = obs[:, sl["hand_joint_pos"]]
    q_cmd, arm = act[:, n_arm:], act[:, :n_arm]
    settle = int(round(args.settle_s / dt))
    runs = perturbations(push)
    jinfo = [(int(n.split("finger")[1][0]), int(n[-1])) for n in hand]  # (finger, joint) per hand column
    kp = np.array([KP[j] for _, j in jinfo])
    tmax = np.array([TAU_MAX["thumb" if fi == 1 else "finger"][j] for fi, j in jinfo])
    f25 = np.array([fi >= 2 for fi, _ in jinfo])
    for i, (s, e) in enumerate(runs):
        if s < 1 or e >= T:
            continue  # starts at the very first sample or runs into the end: no clean windows
        b, en = s - 1, e - 1  # rows of the steps before the onset and the last one with the force
        a = en + settle if en + settle + 1 < T and (i + 1 == len(runs) or runs[i + 1][0] > en + settle) else None
        F = push[s]
        # push = toward the thumb: along the force on the thumb at the onset (it points from the cube to the thumb)
        kind = "push" if float(F @ f_vec[s, 0]) > 0 else "pull"
        r = {"demo": name, "i": i, "kind": kind, "F": float(np.linalg.norm(F)), "dur": (e - s) * dt,
             "df": np.full(5, np.nan), "dx": np.full(5, np.nan), "c": np.full(5, np.nan), "back": np.full(5, np.nan),
             "contact": np.zeros(5, bool)}
        for k in range(5):
            f0, f1 = f_vec[b + 1, k], f_vec[en + 1, k]
            n0 = f0 if np.linalg.norm(f0) > CONTACT_N else f1
            if np.linalg.norm(n0) <= CONTACT_N:
                continue  # not in contact before or at the end
            n = n0 / np.linalg.norm(n0)
            r["contact"][k] = np.linalg.norm(f0) > CONTACT_N
            r["df"][k] = np.linalg.norm(f1) - np.linalg.norm(f0)
            r["dx"][k] = -float((tro[en, k] - tro[b, k]) @ n)
            if r["contact"][k] and abs(r["df"][k]) > DF_MIN:
                r["c"][k] = r["dx"][k] / r["df"][k]
            if a is not None and abs(r["dx"][k]) > 1e-4:
                r["back"][k] = -float((tro[a, k] - tro[b, k]) @ n) / r["dx"][k]
        last = a if a is not None else en
        r["react"] = 1000 * np.abs(q_ref[s:last + 1] - q_ref[b]).max()
        # the same measure over an equally long quiet window just before the onset: drift and tremor
        L = last + 1 - s
        r["drift"] = 1000 * np.abs(q_ref[max(b - L, 0):b + 1] - q_ref[max(b - L, 0)]).max() if b > 0 else float("nan")
        r["arm"] = 1000 * np.abs(arm[s:last + 1] - arm[b]).max()
        tau = kp * np.abs(q_cmd[s:last + 1] - q_hand[s:last + 1])
        r["sat"] = 100.0 * (tau[:, f25] >= tmax[f25]).mean()
        r["clamp"] = [FINGERS[k] for k in range(5) if np.linalg.norm(f_vec[en + 1, k]) / k_file >= 0.03]
        rows.append(r)


def print_perturbations(rows, k_file):
    if args.perturb_detail:
        print(f"\n  perturbations: df [N] / dx [mm] per finger (thumb, index, middle, ring, little; '.' = no contact), "
              f"q_ref and arm change [mrad], fingers 2-5 joints saturated [%]")
        for r in rows:
            cells = " ".join("    .     " if np.isnan(r["df"][k]) else f"{r['df'][k]:+5.2f}/{1000 * r['dx'][k]:+5.1f}"
                             for k in range(5))
            flag = "  REACTION?" if r["react"] - np.nan_to_num(r["drift"]) > args.react_mrad else ""
            clamp = f"  at e_max (est.): {','.join(r['clamp'])}" if r["clamp"] else ""
            print(f"  {r['demo']:<9} #{r['i']:<2} {r['kind']:<4} {r['F']:4.2f} N {r['dur']:4.2f} s | {cells} | "
                  f"q_ref {r['react']:5.1f} (before {r['drift']:5.1f}) arm {r['arm']:5.1f} | sat {r['sat']:4.0f} %{flag}{clamp}")
    print(f"\n  perturbations: {len(rows)} in {len({r['demo'] for r in rows})} demos "
          f"({sum(r['kind'] == 'push' for r in rows)} push, {sum(r['kind'] == 'pull' for r in rows)} pull); "
          f"medians, fingers in contact before the onset:")
    print(f"  {'':<5} {'finger':<7} {'n':>4} {'df N':>7} {'dx mm':>7} {'dx/df mm/N':>10} {'back':>6}")
    for kind in ("push", "pull"):
        rk = [r for r in rows if r["kind"] == kind]
        for k in range(5):
            m = np.array([r["contact"][k] for r in rk], bool)
            if not m.any():
                continue
            df = np.array([r["df"][k] for r in rk])[m]
            dx = np.array([r["dx"][k] for r in rk])[m]
            c = np.array([r["c"][k] for r in rk])[m]
            bk = np.array([r["back"][k] for r in rk])[m]
            med = lambda v: np.nanmedian(v) if np.isfinite(v).any() else float("nan")  # noqa: E731
            print(f"  {kind:<5} {FINGERS[k]:<7} {m.sum():>4} {med(df):>+7.2f} {1000 * med(dx):>+7.1f} "
                  f"{1000 * med(c):>10.1f} {med(bk):>6.2f}")
            if k >= 1:
                compliance.setdefault(k_file, []).extend(c[np.isfinite(c)].tolist())
    react = np.array([r["react"] for r in rows])
    excess = react - np.nan_to_num(np.array([r["drift"] for r in rows]))
    sat = np.array([r["sat"] for r in rows])
    if len(rows):
        print(f"  q_ref change during a perturbation: median {np.median(react):.1f} mrad, max {react.max():.1f} mrad; "
              f"minus the change in an equally long window before it: median {np.median(excess):.1f} mrad, "
              f"{(excess > args.react_mrad).sum()} above {args.react_mrad:g} (reaction?); fingers 2-5 joints saturated: "
              f"median {np.median(sat):.0f} %; at the offset limit (est. |f|/K >= 30 mm) at the end: "
              f"{sum(bool(r['clamp']) for r in rows)} perturbations")
        print(f"  expected dx/df at K = {k_file:g} (steady state): admittance in series with the drives "
              f"{1000 * (1 / k_file + 1 / K_DRV):.1f} mm/N, admittance alone {1000 / k_file:.1f} mm/N")


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
        durations, ratios, prows = [], [], []
        perturbed = "push_force" in str(a.get("format", ""))
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
                for key in ("q_ref", "dq", "tip_ref_offset", "tip_force_raw") + (("push_force",) if perturbed else ()):
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
                    if perturbed:
                        n0 = len(prows)
                        perturbation_report(name, obs, act, g, sl, hand, n_arm, k_file, dt, prows)
                        extra += f"  perturbations {len(perturbations(g['push_force'][()]))} ({len(prows) - n0} analysed)"
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
        if perturbed:
            print_perturbations(prows, k_file)
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
kc = sorted(compliance)
if len(kc) == 2 and all(compliance[k] for k in kc):
    lo, hi = np.median(compliance[kc[0]]), np.median(compliance[kc[1]])
    law = (1 / kc[0] + 1 / K_DRV) / (1 / kc[1] + 1 / K_DRV)
    print(f"\nperturbations, fingers 2-5: dx/df at K = {kc[0]:g} vs K = {kc[1]:g}: {1000 * lo:.1f} vs {1000 * hi:.1f} mm/N, "
          f"ratio {lo / hi:.2f} (steady state, admittance in series with the drives: {law:.2f}; admittance alone "
          f"{kc[1] / kc[0]:.2f})")
print(f"\n{'ALL DEMOS OK' if all_ok else 'SOME DEMOS FAILED A CHECK'}")
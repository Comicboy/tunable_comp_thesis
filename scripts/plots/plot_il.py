"""Figures, tables and captions of the simulation IL stage (paper and slides).

What they show: how the contact force, the stiffness K and the penetration are related at the fingertips in
the scripted lift, the teleoperated demonstrations and the policy rollouts, and how well the policy does the
task. Only the two stiffnesses of the demonstrations (K = 150 and 500 N/m) and the dynamic admittance
controller (the one the demonstrations were recorded with) are compared. No Isaac Sim, no LeRobot (h5py,
numpy, matplotlib); everything is recomputed from the files:

    python scripts/plots/plot_il.py \\
        --demos logs/il_demos/lift_K150.hdf5 logs/il_demos/lift_K500.hdf5 \\
        --rollouts logs/il_eval/act_qcmd_v1_te0.01_K150_seed0.hdf5 logs/il_eval/act_qcmd_v1_te0.01_K500_seed0.hdf5 \\
        --scripted logs/scripted_lift/dynamic_2026-11-03_16-33-51.npz \\
        --val_demos logs/il_demos_val/lift_K150.hdf5 logs/il_demos_val/lift_K500.hdf5 \
        --kswap logs/il_eval/kswap/act_qcmd_v1_val --kswap_train logs/il_eval/kswap/act_qcmd_v1 \
        --pair_episode 23 [--out DIR]

Inputs: the demo files (teleop_lift.py), the rollout trajectory files (rollout_policy.py; the results
JSON next to each gives the outcomes), optionally a scripted-lift log of the dynamic controller
(scripted_lift.py; its K = 150 and 500 envs are used) and the K-swap folder (kswap_test.py: kswap.h5,
summary.json). Every input except the K-swap folders needs its fingertip Jacobians and drive properties
next to it, <name>_kin.h5, from scripts/tools/fingertip_kinematics.py.

Training vs validation demonstrations: everything that evaluates the policy on demonstration states (kswap,
law_capture, command_error) uses the validation demos (--val_demos, recorded with teleop_lift.py --split val,
never trained on) and their K-swap folder (--kswap, from kswap_test.py on those files), so it measures the
policy and not its memory of the training states. --kswap_train (kswap_test.py on the training demos) only
adds the training numbers to the table, as a reference for the gap. The closed-loop figures use rollouts
(new trajectories anyway) and keep the training demos (--demos) as the reference: the behaviour taught.
Without --val_demos the K-swap figures fall back to the training demos (--demos).

Quantities (per fingertip, base frame; n = f / |f| is the direction of the contact force, out of the cube):
    contact force       |f|                                      [N]
    yield               e . n        (the admittance offset)      [mm]   e = (I + lambda^2 (J J^T)^-1) J dq in the demos
    reference penetr.   delta_ref = -J (q_ref - q) . n            [mm]   how far the reference reaches into the cube
    command penetr.     delta_cmd = -J (q_cmd - q) . n            [mm]   what the drives turn into force
                        (in steady contact delta_cmd = delta_ref - e: the admittance gives away the yield)
    drive stiffness     K_drv = 1 / (n^T J K_p^-1 J^T n)            [N/m]  the joint drives at the fingertip, along n
    joint side          tau_d = clip(k_p (q_cmd - q), +-tau_max) against -(J^T f)_j [Nm], per joint
Demos and rollouts: sample t pairs the command of step t with the state at its end (obs[t + 1]: angles,
force); scripted lift: everything logged after the step.

Output (PNG, --out, default logs/il_eval/figures/<rollout name>/), each with a short title; the
explanation is in captions.md:
    outcome              task outcome per K
    hold_force           grip force and command penetration in the hold per finger, demonstrations vs policy (one
                         marker per episode; the demonstrations' median reference penetration as a dashed bar with
                         an arrow down to the median command: the yield)
    hold_share           the policy's hold force as a share of the demonstrations' per finger and K (ratio of the means)
    grip_timeline        force and cube height over time, all successful episodes aligned at the lift-off (or at the
                         end, --timeline_align end), median and interquartile range per K, demos vs policy
    penetration_timeline the same for the command penetration (samples in contact)
    admittance_law       yield vs contact force (scripted lift, demonstrations), the law e = f / K
    penetration          contact force vs reference and command penetration, one marker per episode (scripted
                         lift, demonstrations, policy); the gap between the two is the yield
    kswap                Delta^pi = delta^pi_cmd(s, 500) - delta^pi_cmd(s, 150): change of the policy's command
                         penetration when only K is swapped, against the force, with the admittance's change
                         Delta^adm = delta^adm_cmd(s, 500) - delta^adm_cmd(s, 150) at the same state
    law_capture          gain g of Delta^pi on Delta^adm per finger and phase (closing, lift, hold, whole episode):
                         how much of the admittance law's K-dependence the policy captures
    command_error        the policy's command against the admittance's at the same state, at the demonstration's
                         own K and at the swapped K, per finger and phase (approach, closing, lift, hold, whole)
    force_ratio          grip-force ratio K = 500 / 150 per source, with the law and the series prediction
    precontact           the policy's output before contact, K = 150 vs 500 at the same cube positions
    single_demos         one demonstration per K (time series per finger)
    single_rollouts      one rollout per K at the same cube position (--pair_episode)
    drive_relation       validity check: drive torque vs the torque of the measured fingertip force, per joint
    tables.md / tables.tex / tables.json, captions.md
"""

import argparse
import json
import math
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_style import (  # noqa: F401  (shared look and statistics, scripts/plots/plot_style.py)
    AXIS, FINGER_COLOR, FINGERS, GRID, INK, INK_2, MUTED, PHASE_BAND, RIGID, SURFACE, aligned, band, k_shade, layout,
    mcnemar, quiet, suptitle, wilson,
)
from plot_style import ratio_ci as _ratio_ci
from plot_style import save as _save

parser = argparse.ArgumentParser(description="Figures, tables and captions of the sim IL stage.")
parser.add_argument("--demos", nargs=2, required=True, help="the two demo files (one per K)")
parser.add_argument("--rollouts", nargs=2, required=True, help="rollout trajectory files (.hdf5, one per K, same seed)")
parser.add_argument("--scripted", default=None,
                    help="scripted-lift log (.npz) of the dynamic controller; its envs at the demos' K are used")
parser.add_argument("--val_demos", nargs=2, default=None,
                    help="the two validation demo files (teleop_lift.py --split val), for the K-swap figures")
parser.add_argument("--kswap", default=None, help="output folder of kswap_test.py (summary.json, kswap.h5) on the "
                    "validation demos (on --demos without --val_demos)")
parser.add_argument("--kswap_train", default=None, help="kswap_test.py folder of the training demos: their numbers in "
                    "the comparison table only")
parser.add_argument("--pair_episode", type=int, default=None, help="rollout episode for single_rollouts (default: "
                    "both K successful, cube closest to the spawn centre)")
parser.add_argument("--hold_s", type=float, default=3.0, help="hold window: the last hold_s of a successful episode "
                    "[s] (3 s = the success hold)")
parser.add_argument("--timeline_align", choices=["liftoff", "end"], default="liftoff",
                    help="grip_timeline: align the episodes at the cube's lift-off or at their end (success)")
parser.add_argument("--timeline_window", type=float, nargs=2, default=None, metavar=("FROM", "TO"),
                    help="grip_timeline: time window around the alignment point [s] (default -1.5 6 at the lift-off, "
                    "-6 0 at the end)")
parser.add_argument("--timeline_share", type=float, default=0.5, help="grip_timeline: a time is shown only while at "
                    "least this share of the episodes is still running (and, for the penetration, half of those in contact)")
parser.add_argument("--window_s", type=float, default=0.5, help="scripted lift: window at the end of the squeeze and the hold [s]")
parser.add_argument("--contact_n", type=float, default=0.5, help="a fingertip counts as in contact above this force [N]")
parser.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
parser.add_argument("--out", default=None, help="output folder (default logs/il_eval/figures/<rollout name>)")
args = parser.parse_args()

LAMBDA = 0.01        # [m] damping of the admittance's pseudo-inverse (FingertipAdmittanceActionCfg.ik_damping)
E_MAX = 0.03         # [m] the admittance's offset limit per fingertip (AdmittanceControllerCfg.max_offset)
LIMIT_TOL = 0.01     # [rad] a joint this close to its (soft) position limit is held by the limit
CENTRE = (0.60, 0.11)  # [m] centre of the cube spawn square (base frame)
TOUCH_N = 1e-3       # [N] summed fingertip force: first contact
LIFTOFF_M = 0.005    # [m] cube bottom above the table: lift-off
LIFTED_M = 0.03      # [m] max lift below this: "never lifted"
TITLES = ["Thumb", "Index", "Middle", "Ring", "Little"]
SRC_MARKER = {"scripted": "D", "demo": "o", "policy": "^"}
SRC_LABEL = {"scripted": "Scripted lift", "demo": "Demonstrations", "policy": "Policy"}
OUTCOME = {  # outcome -> (label, colour)
    "never": ("Never lifted", "#e34948"),
    "partial": ("Lifted < 15 cm", "#eda100"),
    "nohold": ("Lifted, 3 s hold not kept", "#eb6834"),
}
PHASES = ["approach", "closing", "lift", "hold"]  # per sample of a demonstration, see phase_labels()
CAPTIONS = {}
OUT = None
KS_SET = "demonstrations"  # which demonstrations the K-swap figures use (set in main)
IL_KS = ()  # the two stiffnesses of the demonstrations (set in main)


ROLES = {  # what each figure answers (written into captions.md)
    "outcome": "Task performance of the policy",
    "hold_force": "(a) Does the policy reproduce the stiffness-dependent grip of the demonstrations, in force and in "
                  "command depth?",
    "hold_share": "(a) How much of the demonstrations' grip force the policy applies",
    "grip_timeline": "(a) How the grip force develops over time, demonstrations vs policy",
    "penetration_timeline": "(a) How the command depth develops over time, demonstrations vs policy",
    "force_ratio": "(a) Does the policy reproduce the stiffness-dependent grip of the demonstrations?",
    "single_rollouts": "(a), one example",
    "penetration": "(a) How the force arises: the command depth on the drives' characteristic",
    "kswap": "(b) Does the policy change its command with K as the admittance law does?",
    "law_capture": "(b) How much of the admittance law's K-dependence the policy captures, per phase",
    "command_error": "(b) How close the policy's command is to the admittance's, at the own and the swapped K",
    "precontact": "(b), negative control: before contact the law asks for no change",
    "admittance_law": "Background: the behaviour the policy has to learn",
    "single_demos": "Background: the behaviour the policy has to learn, one example",
    "drive_relation": "Check: is the measured fingertip force what the drives produce?",
}


def save(fig, name, caption):
    _save(fig, OUT, name)
    CAPTIONS[name] = caption


def ratio_ci(lo, hi, min_median=0.0):
    """median(hi) / median(lo) with a 95 % bootstrap, own generator per call (same groups, same interval)."""
    return _ratio_ci(np.asarray(lo), np.asarray(hi), np.random.default_rng(0), boot=args.boot, min_median=min_median)


def k_text(k):
    return "∞" if np.isinf(k) else f"{k:g}"


# ---------------------------------------------------------------------------------------------------
# loading: every source becomes a list of episodes with the same per-sample quantities
# ---------------------------------------------------------------------------------------------------
def load_kin(path):
    p = os.path.splitext(path)[0] + "_kin.h5"
    if not os.path.exists(p):
        raise SystemExit(f"{p} is missing: run scripts/tools/fingertip_kinematics.py on {path} first")
    with h5py.File(p, "r") as f:
        cols = np.asarray(f.attrs["hand_cols"]).astype(int)
        if "effort_limit" not in f.attrs:
            raise SystemExit(f"{p} has no drive properties: re-run scripts/tools/fingertip_kinematics.py --overwrite on {path}")
        drive = {k: np.asarray(f.attrs[k], float) for k in ("kp", "kd", "effort_limit", "pos_limits")}
        J = {g: f["data"][g]["J"][()].astype(np.float64) for g in f["data"]}
    return cols, J, drive


def tip(J, v):
    """J (T, 5, 3, 4) times per-finger joint vectors v (T, 5, 4) -> (T, 5, 3)."""
    return np.einsum("tfij,tfj->tfi", J, v)


def recover_offset(J, dq_f):
    """The admittance offset e from its joint offset dq = J^T (J J^T + l^2 I)^-1 e: e = (I + l^2 (J J^T)^-1) J dq."""
    Jd = tip(J, dq_f)
    A = np.einsum("tfij,tfkj->tfik", J, J)
    return Jd + LAMBDA**2 * np.linalg.solve(A, Jd[..., None])[..., 0]


def episode(src, K, name, dt, success, q, q_cmd, f, J, cols, drive, h=None, d_ref=None, e=None, raw=None, **extra):
    """Per-sample quantities of one episode (T samples), see the module docstring."""
    fn = np.linalg.norm(f, axis=-1)  # (T, 5)
    n = f / np.maximum(fn, 1e-9)[..., None]
    x = (q_cmd - q)[:, cols]  # (T, 5, 4) joint command offset, joint1..joint4 of each finger
    kp, eff, lim = drive["kp"][cols], drive["effort_limit"][cols], drive["pos_limits"][cols]  # (5, 4), (5, 4), (5, 4, 2)
    qf = q[:, cols]
    at_limit = (qf - lim[..., 0] < LIMIT_TOL) | (lim[..., 1] - qf < LIMIT_TOL)  # (T, 5, 4)
    JTn = np.einsum("tfij,tfi->tfj", J, n)
    # drive stiffness at the fingertip along the force; a joint held by its limit gives no compliance
    compliance = np.where(at_limit, 0.0, JTn**2 / kp).sum(-1)
    ep = {"src": src, "K": float(K), "name": name, "dt": dt, "success": success, "T": len(q), "fn": fn,
          "x": x, "tau": -np.einsum("tfij,tfi->tfj", J, f), "p_cmd": -(tip(J, x) * n).sum(-1),
          "tau_drive": np.clip(kp * x, -eff, eff), "saturated": np.abs(kp * x) >= eff, "at_limit": at_limit,
          "kd": 1.0 / np.maximum(compliance, 1e-12), "h": h, "raw": raw or {}, **extra}
    if d_ref is not None:
        ep["p_ref"] = -(d_ref * n).sum(-1)
    if e is not None:
        ep["e"] = (e * n).sum(-1)
    touch = np.nonzero(fn.sum(-1) > TOUCH_N)[0]
    ep["first"] = int(touch[0]) if touch.size else ep["T"]
    if h is not None:
        up = np.nonzero(h > LIFTOFF_M)[0]
        ep["liftoff"] = int(up[0]) if up.size else None
    ep["last"] = ep["T"] - 1  # alignment point "end" of grip_timeline
    return ep


def load_il(path):
    """Demo or rollout file -> {"K", "dt", "rollout", "eps": {name: episode}, "results", "hand"}."""
    cols, Jk, drive = load_kin(path)
    with h5py.File(path, "r") as f:
        a = f.attrs
        sl = {nm: slice(s, e) for nm, s, e in json.loads(a["obs_terms"])}
        hand = json.loads(a["hand_joints"])
        n_arm = len(json.loads(a["arm_joints"]))
        half = json.loads(a["success"])["half_size"]
        K, dt = float(a["stiffness"]), float(a["step_dt"])
        rollout = a.get("source") == "policy rollout"
        out = {"path": path, "K": K, "dt": dt, "rollout": rollout, "hand": hand, "cols": cols, "drive": drive, "eps": {}}
        for name in sorted(f["data"]):
            g = f["data"][name]
            obs, act = g["obs"][()].astype(np.float64), g["actions"][()].astype(np.float64)
            J = Jk[name]
            T = len(obs)
            # sample t: the command of step t with the state at its end (obs[t + 1]) and J there
            o, ac, Jt = obs[1:], act[:-1], J[1:]
            q, q_cmd = o[:, sl["hand_joint_pos"]], ac[:, n_arm:]
            frc = o[:, sl["fingertip_forces"]].reshape(-1, 5, 3)
            h = o[:, sl["cube_position"]][:, 2] - half
            kw = {}
            if "q_ref" in g:  # demos: the admittance's own logs of step t
                dq = g["dq"][()][:-1].astype(np.float64)
                kw["d_ref"] = g["tip_ref_offset"][()][:-1].astype(np.float64)  # world = base orientation here
                kw["e"] = recover_offset(Jt, dq[:, cols])
            raw = {"q_cmd": act[:, n_arm:], "arm": act[:, :n_arm], "J": J, "q": obs[:, sl["hand_joint_pos"]],
                   "f": obs[:, sl["fingertip_forces"]].reshape(T, 5, 3),
                   "fn": np.linalg.norm(obs[:, sl["fingertip_forces"]].reshape(T, 5, 3), axis=-1),
                   "h": obs[:, sl["cube_position"]][:, 2] - half}
            ep = episode("policy" if rollout else "demo", K, name, dt, bool(g.attrs.get("success", True)), q, q_cmd,
                         frc, Jt, cols, drive, h=h, raw=raw, cube0=obs[0, sl["cube_position"]][:2], **kw)
            touch = np.nonzero(raw["fn"].sum(-1) > TOUCH_N)[0]
            raw["first"] = int(touch[0]) if touch.size else T
            out["eps"][name] = ep
    js = os.path.splitext(path)[0] + ".json"
    out["results"] = json.load(open(js)) if rollout and os.path.exists(js) else None
    return out


def load_scripted(path):
    """Scripted-lift log -> {"controller", "eps": [episode per env]}."""
    cols, Jk, drive = load_kin(path)
    d = np.load(path, allow_pickle=True)
    dt = float(d["dt"])
    names = [str(n) for n in d["phase_names"]]
    phase = np.asarray(d["phase"])
    n_w = int(round(args.window_s / dt))
    win = {}
    for ph in ("squeeze", "hold"):
        idx = np.nonzero(phase == names.index(ph))[0]
        win[ph] = idx[-n_w:]
    eps = []
    for i, K in enumerate(np.asarray(d["stiffness"], float)):
        ep = episode("scripted", K, f"env_{i:03d}", dt, True, d["q"][:, i].astype(float), d["q_cmd"][:, i].astype(float),
                     d["tip_force"][:, i].astype(float), Jk[f"env_{i:03d}"], cols, drive,
                     d_ref=d["tip_ref_offset"][:, i].astype(float), e=d["tip_offset"][:, i].astype(float))
        ep["windows"] = win
        eps.append(ep)
    return {"path": path, "controller": str(d["controller"]), "eps": eps}


# ---------------------------------------------------------------------------------------------------
# selections
# ---------------------------------------------------------------------------------------------------
def hold_idx(ep):
    """The hold window of a successful demo or rollout (its last hold_s), or the scripted lift's squeeze end."""
    if ep["src"] == "scripted":
        return ep["windows"]["squeeze"]
    n = int(round(args.hold_s / ep["dt"]))
    return np.arange(max(ep["T"] - n, 0), ep["T"])


def scripted_eps(sc):
    """The scripted-lift envs at the stiffnesses of the demonstrations (nothing else has a counterpart)."""
    return [e for e in sc["eps"] if e["K"] in IL_KS] if sc else []


def successful(s):
    return [e for e in s["eps"].values() if e["success"]]


def contact_samples(eps, key, finger):
    """All samples of these episodes with the finger in contact: (force, eps[key]) and K per sample."""
    fs, ys, ks = [], [], []
    for e in eps:
        if key not in e:
            continue
        on = e["fn"][:, finger] > args.contact_n
        fs.append(e["fn"][on, finger])
        ys.append(e[key][on, finger])
        ks.append(np.full(on.sum(), e["K"]))
    if not fs:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    return np.concatenate(fs), np.concatenate(ys), np.concatenate(ks)


def outcome_class(r):
    if r["success"]:
        return "success"
    lift = r.get("max_lift_m", 0.0)
    if lift < LIFTED_M:
        return "never"
    return "nohold" if lift >= 0.145 else "partial"  # 0.145: the success height minus its 5 mm tolerance


def success_stats(results, K):
    eps = results["episodes"]
    n, k = len(eps), sum(e["success"] for e in eps)
    cls = [outcome_class(e) for e in eps]
    t = [e["duration_s"] for e in eps if e["success"]]
    lo, hi = wilson(k, n)
    return {"K": K, "n": n, "k": k, "rate": k / n, "lo": lo, "hi": hi, "never": cls.count("never"),
            "partial": cls.count("partial"), "nohold": cls.count("nohold"),
            "t_mean": float(np.mean(t)) if t else float("nan"),
            "success": np.array([e["success"] for e in eps]), "episode": [e["episode"] for e in eps]}


def finger_axes(nrows, height, sharey="row", extra_cols=0, width=15.0):
    fig, axes = plt.subplots(nrows, 5 + extra_cols, figsize=(width, height), sharey=sharey, squeeze=False)
    return fig, axes


def k_legend(ks, extra=()):
    hs = [Line2D([], [], marker="o", linestyle="", color=k_shade(k), markersize=7,
                 label="K = ∞ (rigid)" if np.isinf(k) else f"K = {k:g} N/m") for k in ks]
    return hs + list(extra)


# ---------------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------------
def fig_outcome(stats, mc):
    fig, ax = plt.subplots(figsize=(9, 3.0))
    parts = ["success", "nohold", "partial", "never"]
    for row, s in enumerate(stats):
        left = 0.0
        for key in parts:
            n = s["k"] if key == "success" else s[key]
            w = 100 * n / s["n"]
            if w <= 0:
                continue
            ax.barh(row, w, left=left, height=0.6, color=k_shade(s["K"]) if key == "success" else OUTCOME[key][1],
                    edgecolor=SURFACE, linewidth=2)
            if w >= 2.5:
                ax.text(left + w / 2, row, f"{n}", ha="center", va="center", fontsize=10,
                        color="#ffffff" if key != "partial" else INK)
            left += w
        ax.text(101.5, row, f"{100 * s['rate']:.0f} % [{100 * s['lo']:.0f}, {100 * s['hi']:.0f}]", va="center",
                fontsize=10.5, color=INK)
    ax.set_yticks(range(len(stats)), [f"K = {s['K']:g} N/m" for s in stats])
    ax.invert_yaxis()
    ax.set_xlim(0, 122)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Share of the rollouts [%]")
    ax.grid(axis="y", visible=False)
    handles = [(Patch(color=k_shade(stats[0]["K"])), Patch(color=k_shade(stats[-1]["K"])))]
    labels = ["Success"]
    for key in ("nohold", "partial", "never"):
        if any(s[key] for s in stats):
            handles.append(Patch(color=OUTCOME[key][1]))
            labels.append(OUTCOME[key][0])
    from matplotlib.legend_handler import HandlerTuple
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02),
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})
    suptitle(fig, "Task outcome per stiffness")
    layout(fig, 0.12)
    n = stats[0]["n"]
    save(fig, "outcome",
         f"Outcome of {n} policy rollouts per stiffness K at the same {n} cube start positions. Right: success "
         f"rate with its 95 % Wilson interval. Success: cube bottom held at least 15 cm above the table "
         f"(5 mm tolerance) for 3 s. Paired comparison K = {stats[0]['K']:g} vs {stats[1]['K']:g} N/m: "
         f"{mc['n10']} positions succeed only at the lower K, {mc['n01']} only at the higher K, "
         f"exact McNemar p = {mc['p']:.2f}.")


def hold_values(demos, rolls):
    """Per group (source, K index): the hold force of every successful episode (n, 5) [N], and per finger the command
    and reference penetration [mm] of the episodes with that finger in contact in the hold (episode_medians)."""
    out = {}
    for src, sets in (("demo", demos), ("policy", rolls)):
        for i in (0, 1):
            eps = successful(sets[i])
            force = np.array([np.median(e["fn"][hold_idx(e)], axis=0) for e in eps]).reshape(-1, 5)
            cmd, ref = [], []
            for f in range(5):
                ms = [m for m in (episode_medians(e, f) for e in eps) if m is not None]
                cmd.append(1000 * np.array([m["p_cmd"] for m in ms]))
                ref.append(1000 * np.array([m["p_ref"] for m in ms if "p_ref" in m]))
            out[(src, i)] = {"force": force, "cmd": cmd, "ref": ref}
    return out


def group_stats(lo, hi, min_median):
    """Median and IQR of two groups (K low, K high) and the ratio of the medians hi / lo [95 % bootstrap]."""
    nan = float("nan")
    med = lambda a: float(np.median(a)) if len(a) else nan  # noqa: E731
    iqr = lambda a: np.percentile(a, [25, 75]).tolist() if len(a) else [nan, nan]  # noqa: E731
    ok = len(lo) >= 3 and len(hi) >= 3
    return {"median": [med(lo), med(hi)], "iqr": [iqr(lo), iqr(hi)], "n": [len(lo), len(hi)],
            "ratio": ratio_ci(lo, hi, min_median=min_median) if ok else (nan, nan, nan)}


def share_ci(d, p, stat, min_den):
    """stat(p) / stat(d), the policy as a share of the demonstrations, with a 95 % bootstrap (episodes resampled
    within each group). NaN with fewer than 3 episodes per group or stat(d) < min_den; a resample whose denominator
    falls below min_den counts as infinite, so an unstable ratio shows an open upper bound instead of a huge number."""
    nan = float("nan")
    d, p = np.asarray(d, float), np.asarray(p, float)
    if len(d) < 3 or len(p) < 3 or stat(d) < min_den:
        return [nan, nan, nan]
    rng = np.random.default_rng(0)
    b = np.empty(args.boot)
    for i in range(args.boot):
        den = stat(rng.choice(d, len(d)))
        b[i] = stat(rng.choice(p, len(p))) / den if den >= min_den else np.inf
    # nearest-rank percentiles: linear interpolation between two infinite resamples gives inf - inf = NaN
    # (and a RuntimeWarning); "lower" / "higher" pick a resample, so an open bound stays inf
    return [float(stat(p) / stat(d)), float(np.percentile(b, 2.5, method="lower")),
            float(np.percentile(b, 97.5, method="higher"))]


def policy_vs_demo(d, p, min_median):
    """The policy against the demonstrations at one K: difference of the medians (policy - demos) and their ratio
    policy / demos [95 % bootstrap]."""
    nan = float("nan")
    diff = float(np.median(p) - np.median(d)) if len(d) >= 3 and len(p) >= 3 else nan
    return {"diff": diff, "ratio": share_ci(d, p, np.median, min_median), "n": [len(d), len(p)]}


def fig_hold_force(demos, rolls, tables):
    """Hold force (top) and command penetration (bottom) per finger, demonstrations vs policy, one marker per episode;
    the demonstrations' reference penetration as an arrow down to their command (the yield)."""
    ks = [demos[0]["K"], demos[1]["K"]]
    groups = [("demo", 0), ("demo", 1), ("policy", 0), ("policy", 1)]
    xpos = {g: xi + (0.5 if g[0] == "policy" else 0.0) for xi, g in enumerate(groups)}
    v = hold_values(demos, rolls)
    fig, axes = plt.subplots(2, 6, figsize=(16, 7.6))
    jit = np.random.default_rng(1)
    rec, rec_p, vs = {}, {}, {}

    def column(ax, data):
        for g in groups:
            y = data[g]
            if not len(y):
                continue
            x = xpos[g]
            ax.scatter(x + jit.uniform(-0.15, 0.15, len(y)), y, s=14, marker=SRC_MARKER[g[0]], color=k_shade(ks[g[1]]),
                       alpha=0.6, linewidths=0, zorder=3)
            ax.plot([x - 0.28, x + 0.28], [np.median(y)] * 2, color=INK, linewidth=2.0, zorder=4)
        ax.set_xticks([0.5, 3.0], ["Demos", "Policy"])
        ax.set_xlim(-0.6, 4.1)
        ax.grid(axis="x", visible=False)

    # top: contact force, per finger and fingers 2-5 together
    for c in range(6):
        ax = axes[0, c]
        get = (lambda g, c=c: v[g]["force"][:, c]) if c < 5 else (lambda g: v[g]["force"][:, 1:].sum(-1))
        column(ax, {g: get(g) for g in groups})
        ax.set_ylim(bottom=0)
        ax.set_title((TITLES + ["Fingers 2–5 (sum)"])[c])
        name = (FINGERS + ["fingers 2-5"])[c]
        for src in ("demo", "policy"):
            rec[f"{src} | {name}"] = group_stats(get((src, 0)), get((src, 1)), 0.5)
    # bottom: command penetration per finger (no sum: it is a per-fingertip depth)
    for f in range(5):
        ax = axes[1, f]
        column(ax, {g: v[g]["cmd"][f] for g in groups})
        for i in (0, 1):
            r, cm = v[("demo", i)]["ref"][f], v[("demo", i)]["cmd"][f]
            if len(r) and len(cm):
                # a different quantity from the dots: the operator's reference, not the command
                x, col, rm = xpos[("demo", i)], k_shade(ks[i]), float(np.median(r))
                ax.plot([x - 0.28, x + 0.28], [rm, rm], color=col, linewidth=1.8, linestyle=(0, (2.0, 1.4)), zorder=6)
                ax.text(x + 0.31, rm, "ref", ha="left", va="center", fontsize=8.5, color=INK_2)
                ax.annotate("", xy=(x, np.median(cm)), xytext=(x, rm), zorder=5,
                            arrowprops=dict(arrowstyle="-|>", color=col, linewidth=1.3, shrinkA=1, shrinkB=2))
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(min(0.0, lo_), hi_)
        for src in ("demo", "policy"):
            st = group_stats(v[(src, 0)]["cmd"][f], v[(src, 1)]["cmd"][f], 1.0)
            if src == "demo":
                st["ref_median"] = [float(np.median(v[(src, i)]["ref"][f])) if len(v[(src, i)]["ref"][f])
                                    else float("nan") for i in (0, 1)]
            rec_p[f"{src} | {FINGERS[f]}"] = st
        vs[f"delta_cmd | {FINGERS[f]}"] = [policy_vs_demo(v[("demo", i)]["cmd"][f], v[("policy", i)]["cmd"][f], 1.0)
                                           for i in (0, 1)]
    axes[0, 0].set_ylabel(r"Contact force $\Vert f\Vert$ [N]")
    axes[1, 0].set_ylabel(r"Command penetration $\delta_{\mathrm{cmd}}$ [mm]")
    axes[1, 5].axis("off")
    hs = [Line2D([], [], marker=SRC_MARKER[s], linestyle="", color=MUTED, label=SRC_LABEL[s]) for s in ("demo", "policy")]
    hs += [Patch(color=k_shade(k), label=f"K = {k:g} N/m") for k in ks]
    hs += [Line2D([], [], color=INK, linewidth=2, label="Median over the episodes"),
           Line2D([], [], color=MUTED, linewidth=1.8, linestyle=(0, (2.0, 1.4)),
                  label="Demos: median reference\n" + r"penetration $\delta_{\mathrm{ref}}$ ('ref');" + "\narrow: the yield")]
    axes[1, 5].legend(handles=hs, loc="center left", frameon=False)
    suptitle(fig, "Grip force and command penetration in the hold")
    layout(fig)
    tables["hold_force"] = rec
    tables["hold_penetration"] = rec_p
    tables["hold_policy_vs_demo"] = vs
    n_d = [len(v[("demo", i)]["force"]) for i in (0, 1)]
    n_p = [len(v[("policy", i)]["force"]) for i in (0, 1)]
    save(fig, "hold_force",
         f"The grip while the cube is held: one marker per successful episode (median over its last {args.hold_s:g} s), "
         f"demonstrations (n = {n_d[0]} / {n_d[1]}) and policy rollouts (n = {n_p[0]} / {n_p[1]}) at K = {ks[0]:g} and "
         f"{ks[1]:g} N/m; bars: medians over the episodes. Top: contact force between the cube and each fingertip "
         "(only the normal force from the cube on the fingertip links is measured: not friction, not contacts of "
         "other finger segments, the palm or the table). Bottom: command penetration delta_cmd, the component of "
         "J (q_cmd - q) along -f, i.e. how deep the commanded fingertip lies inside the cube (what the drives turn "
         "into force); per episode over the hold samples with the finger in contact, only episodes with the finger "
         "in contact in the hold. Dashed bar marked 'ref' (demonstrations only): the median reference "
         "penetration delta_ref, a different quantity from the markers (the operator's hand, before the admittance); "
         "the arrow down to the median command is the yield the admittance takes off it. The "
         "policy has no reference; it outputs the command directly. Where a joint drive is saturated, a deeper "
         "command adds no force (see drive_relation).")


def band_running(rows, share):
    """Median, 25th and 75th percentile over the episodes (rows) at each time; NaN where fewer than
    max(3, share * n) episodes have data (so the curve does not end on the few longest episodes)."""
    m = np.array(rows, float)
    enough = (~np.isnan(m)).sum(0) >= max(3, int(math.ceil(share * len(m))))
    return tuple(np.where(enough, quiet(np.nanpercentile, m, p, 0), np.nan) for p in (50, 25, 75))


def timeline_window():
    """Alignment key, time window [s], lags [samples] and times [s] of the timeline figures."""
    align = args.timeline_align
    w = args.timeline_window or ((-1.5, 6.0) if align == "liftoff" else (-6.0, 0.0))
    return align, w, "liftoff" if align == "liftoff" else "last"


def timeline_groups(demos, rolls, key):
    """(source, K, successful episodes with an alignment point) for the four groups."""
    for src, sets in (("demo", demos), ("policy", rolls)):
        for s_ in sets:
            yield src, s_["K"], [e for e in successful(s_) if e.get(key) is not None]


TL_STYLE = {"demo": dict(linestyle="-", linewidth=1.8), "policy": dict(linestyle="--", linewidth=1.8)}


def timeline_finish(fig, axes, align, w, legend_ax, ks, title, name, caption, ylabels):
    xlabel = "Time from the lift-off [s]" if align == "liftoff" else "Time to the end (success) [s]"
    for ax in axes:
        if align == "end":
            ax.axvspan(-args.hold_s, 0, color=PHASE_BAND, zorder=0, lw=0)
        ax.axvline(0, color=AXIS, linewidth=1.0, zorder=1)
        ax.set_xlim(*w)
        ax.set_xlabel(xlabel)
    for ax, lab in ylabels:
        ax.set_ylabel(lab)
    hs = [Line2D([], [], color=MUTED, label=SRC_LABEL["demo"], **TL_STYLE["demo"]),
          Line2D([], [], color=MUTED, label=SRC_LABEL["policy"], **TL_STYLE["policy"])]
    hs += [Patch(color=k_shade(k), label=f"K = {k:g} N/m") for k in ks]
    hs += [Patch(color=MUTED, alpha=0.25, label="Interquartile range")]
    if align == "end":
        hs.append(Patch(color=PHASE_BAND, label=f"Hold (last {args.hold_s:g} s)"))
    legend_ax.axis("off")
    legend_ax.legend(handles=hs, loc="center left", frameon=False)
    suptitle(fig, title + (", aligned at the lift-off" if align == "liftoff" else ", aligned at the end"))
    layout(fig)
    save(fig, name, caption)


def timeline_caption_head(align, counts, ks, share):
    ref = ("the cube's lift-off (its bottom first more than 5 mm above the table)" if align == "liftoff"
           else "their end (the success: the cube held for 3 s; shaded: the hold window of hold_force)")
    n = lambda src, k: counts.get(f"{src} | {k:g}", 0)  # noqa: E731
    return (f"All successful episodes aligned at {ref}: median (line) and interquartile range (band) over the "
            f"episodes, demonstrations (solid; n = {n('demo', ks[0])} / {n('demo', ks[1])}) and policy rollouts (dashed; "
            f"n = {n('policy', ks[0])} / {n('policy', ks[1])}) at K = {ks[0]:g} and {ks[1]:g} N/m. A time is shown while "
            f"at least {100 * share:.0f} % of the episodes are still running (episodes have different lengths).")


def fig_grip_timeline(demos, rolls, tables):
    """Contact force over time (per finger, fingers 2-5 together) and the cube height: all successful episodes
    aligned at the lift-off (or at their end), median and IQR per source and K."""
    align, w, key = timeline_window()
    dt = demos[0]["dt"]
    lag = np.arange(int(round(w[0] / dt)), int(round(w[1] / dt)) + 1)
    t = lag * dt
    ks = [demos[0]["K"], demos[1]["K"]]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.6))
    panels = list(axes[0]) + list(axes[1, :3])  # thumb, index, middle, ring | little, fingers 2-5, cube height
    counts = {}
    for src, k, eps in timeline_groups(demos, rolls, key):
        counts[f"{src} | {k:g}"] = len(eps)
        if len(eps) < 3:
            continue
        fns = [lambda e, c=c: e["fn"][:, c] for c in range(5)] + [lambda e: e["fn"][:, 1:].sum(-1),
                                                                  lambda e: 100 * e["h"]]
        for ax, fn in zip(panels, fns):
            med, q1, q3 = band_running(aligned(eps, fn, key, lag), args.timeline_share)
            ax.fill_between(t, q1, q3, color=k_shade(k), alpha=0.10, lw=0)
            ax.plot(t, med, color=k_shade(k), **TL_STYLE[src])
    for ax, title in zip(panels, TITLES + ["Fingers 2–5 (sum)", "Cube height"]):
        ax.set_title(title)
        ax.set_ylim(bottom=0)
    tables["grip_timeline"] = {"align": align, "window_s": list(w), "share": args.timeline_share, "episodes": counts}
    timeline_finish(fig, panels, align, w, axes[1, 3], ks, "Grip force over time", "grip_timeline",
                    timeline_caption_head(align, counts, ks, args.timeline_share)
                    + " Panels: contact force per fingertip, of fingers 2–5 together, and the cube height (bottom above "
                    "the table).",
                    [(axes[0, 0], r"Contact force $\Vert f\Vert$ [N]"), (axes[1, 0], r"Contact force $\Vert f\Vert$ [N]"),
                     (axes[1, 2], "Cube bottom above the table [cm]")])


def fig_penetration_timeline(demos, rolls, tables):
    """Command penetration over time per finger, from the samples in contact, aligned as grip_timeline."""
    align, w, key = timeline_window()
    dt = demos[0]["dt"]
    lag = np.arange(int(round(w[0] / dt)), int(round(w[1] / dt)) + 1)
    t = lag * dt
    ks = [demos[0]["K"], demos[1]["K"]]
    share = args.timeline_share
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6))
    panels = list(axes[0]) + list(axes[1, :2])
    counts = {}
    for src, k, eps in timeline_groups(demos, rolls, key):
        counts[f"{src} | {k:g}"] = len(eps)
        if len(eps) < 3:
            continue
        # shown while at least half of the running episodes have the finger in contact
        running = (~np.isnan(np.array(aligned(eps, lambda e: np.zeros(e["T"]), key, lag)))).sum(0)
        for f, ax in enumerate(panels):
            m = np.array(aligned(eps, lambda e, f=f: np.where(e["fn"][:, f] > args.contact_n, 1000 * e["p_cmd"][:, f],
                                                             np.nan), key, lag))
            on = (~np.isnan(m)).sum(0)
            ok = (running >= max(3, int(math.ceil(share * len(eps))))) & (on >= 3) & (on >= 0.5 * running)
            med, q1, q3 = (np.where(ok, quiet(np.nanpercentile, m, p, 0), np.nan) for p in (50, 25, 75))
            ax.fill_between(t, q1, q3, color=k_shade(k), alpha=0.10, lw=0)
            ax.plot(t, med, color=k_shade(k), **TL_STYLE[src])
    for f, ax in enumerate(panels):
        ax.set_title(TITLES[f])
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(min(0.0, lo_), hi_)
    lab = r"Command penetration $\delta_{\mathrm{cmd}}$ [mm]"
    timeline_finish(fig, panels, align, w, axes[1, 2], ks, "Command penetration over time", "penetration_timeline",
                    timeline_caption_head(align, counts, ks, share)
                    + " Command penetration delta_cmd per fingertip (component of J (q_cmd - q) along -f), from the "
                    "samples with the finger in contact only, shown while at least half of the running episodes have "
                    "the finger in contact.",
                    [(axes[0, 0], lab), (axes[1, 0], lab)])


def fig_hold_share(demos, rolls, tables):
    """Hold force of the policy as a share of the demonstrations', per finger and K (ratio of the means)."""
    ks = [demos[0]["K"], demos[1]["K"]]
    v = hold_values(demos, rolls)
    names = FINGERS + ["fingers 2-5"]
    get = lambda g, c: v[g]["force"][:, c] if c < 5 else v[g]["force"][:, 1:].sum(-1)  # noqa: E731
    rec = {}
    for c, name in enumerate(names):
        d = [get(("demo", i), c) for i in (0, 1)]
        p = [get(("policy", i), c) for i in (0, 1)]
        rec[name] = {"mean": [share_ci(d[i], p[i], np.mean, 0.2) for i in (0, 1)],
                     "median": [share_ci(d[i], p[i], np.median, 0.5) for i in (0, 1)],
                     "demo_mean": [float(np.mean(x)) for x in d], "policy_mean": [float(np.mean(x)) for x in p],
                     "demo_in_contact": [float(np.mean(x > args.contact_n)) for x in d],
                     "policy_in_contact": [float(np.mean(x > args.contact_n)) for x in p], "n": [len(d[0]), len(p[0])]}
    fin = [r for name in names for r in rec[name]["mean"] if np.isfinite(r[0])]
    lo = min([0.25] + [r[1] for r in fin if np.isfinite(r[1]) and r[1] > 0]) * 0.5  # room for the value row
    hi = max([4.0] + [r[2] for r in fin if np.isfinite(r[2])]) * 1.25
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for i, k in enumerate(ks):
        for c, name in enumerate(names):
            r = rec[name]["mean"][i]
            x = c + (i - 0.5) * 0.36
            # the value in a row along the bottom, under its marker (labels next to the markers collide)
            lab_kw = dict(transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9)
            if not np.isfinite(r[0]):
                ax.text(x, 0.03, "n/a", color=MUTED, **lab_kw)
                continue
            ax.text(x, 0.03, f"{100 * r[0]:.0f} %", color=INK, **lab_kw)
            top = min(r[2], hi)
            ax.errorbar(x, r[0], yerr=[[r[0] - max(r[1], lo)], [top - r[0]]], fmt="o", color=k_shade(k), markersize=7,
                        capsize=3, linewidth=1.4, zorder=3)
            if not np.isfinite(r[2]):
                ax.plot(x, hi, "^", color=k_shade(k), markersize=6, zorder=3)
    ax.axhline(1.0, color=INK, linestyle="--", linewidth=1.2)
    ax.axvline(4.5, color=GRID, linewidth=1.0)
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ticks = [t_ for t_ in (0.125, 0.25, 0.5, 1, 2, 4, 8, 16) if lo <= t_ <= hi]
    ax.set_yticks(ticks, [f"{100 * t_:g} %" for t_ in ticks])
    ax.minorticks_off()
    ax.set_xticks(range(6), TITLES + ["Fingers 2–5\n(sum)"])
    ax.set_xlim(-0.75, 5.75)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Policy hold force / demonstrations'")
    hs = [Line2D([], [], marker="o", linestyle="", color=k_shade(k), markersize=7, label=f"K = {k:g} N/m") for k in ks]
    hs += [Line2D([], [], color=INK, linestyle="--", linewidth=1.2, label="Same force as the demonstrations"),
           Line2D([], [], marker="^", linestyle="", color=MUTED, label="Upper bound open")]
    fig.legend(handles=hs, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    suptitle(fig, "Policy hold force relative to the demonstrations")
    layout(fig, 0.1)
    tables["hold_share"] = rec
    save(fig, "hold_share",
         f"Mean hold force of the policy's successful rollouts divided by that of the demonstrations at the same K, "
         f"per finger and for fingers 2–5 together (per episode the median force over its last {args.hold_s:g} s, "
         "episodes without contact counted with their zero force, so the ratio is the share of the average grip "
         "force). Bars: 95 % bootstrap interval (episodes resampled within each group); log scale, 100 % = the same "
         "force as the demonstrations. Ratio of the means rather than of the medians because several fingers are out "
         "of contact in more than half of the episodes (median 0); the ratio of the medians is in the table. n/a: the "
         "demonstrations' mean below 0.2 N. Triangle: the interval has no upper bound (the demonstrations' resampled "
         "mean falls below 0.2 N).")


def fig_admittance_law(demos):
    """Yield vs contact force in the demonstrations, with the law e = min(|f| / K, e_max)."""
    eps = [e for s_ in demos for e in s_["eps"].values()]
    fig, axes = finger_axes(1, 4.0, sharey=True)
    for f in range(5):
        ax = axes[0, f]
        x, y, kk = contact_samples(eps, "e", f)
        ax.set_title(TITLES[f])
        ax.set_xlabel(r"Contact force $\Vert f\Vert$ [N]")
        if not len(x):
            ax.text(0.5, 0.5, "no contact", transform=ax.transAxes, ha="center", color=MUTED)
            continue
        order = np.argsort(-kk, kind="stable")
        ax.scatter(x[order], 1000 * y[order], s=3, c=[k_shade(k) for k in kk[order]], alpha=0.3, linewidths=0,
                   rasterized=True)
        xm = max(float(np.percentile(x, 99)), 1.0)
        g = np.linspace(0, xm, 200)
        for k in IL_KS:
            ax.plot(g, 1000 * np.minimum(g / k, E_MAX), color=k_shade(k), linestyle="--", linewidth=1.4)
        ax.axhline(1000 * E_MAX, color=MUTED, linestyle=":", linewidth=1.2)
        ax.set_xlim(0, xm)
        ax.set_ylim(bottom=0, top=1000 * E_MAX * 1.12)
    axes[0, 0].set_ylabel(r"Yield $e$ [mm]")
    extra = [Line2D([], [], color=MUTED, linestyle="--", label=r"Admittance law $e = \Vert f\Vert / K$"),
             Line2D([], [], color=MUTED, linestyle=":", label=r"Offset limit $e_{\max}$ = 30 mm")]
    fig.legend(handles=k_legend(IL_KS, extra), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    suptitle(fig, "Admittance law in the demonstrations")
    layout(fig, 0.14)
    save(fig, "admittance_law",
         "Yield e of each fingertip against its contact force in the demonstrations: every sample with the "
         f"fingertip in contact (> {args.contact_n:g} N); colour: K. The yield is the admittance controller's offset, "
         "how far it pulls the fingertip back from the operator's reference because of the force (its component "
         "along the force; recovered exactly from the recorded joint offset dq = J^+ e). Dashed: the steady-state "
         f"law e = |f| / K, limited to e_max = 30 mm, reached at {IL_KS[0] * E_MAX:g} N for K = {IL_KS[0]:g} and at "
         f"{IL_KS[1] * E_MAX:g} N for K = {IL_KS[1]:g} N/m. Points off the line are the controller's dynamics "
         "(mass-spring-damper, force filter), e.g. the offset decaying more slowly than the force when the grip "
         "loosens.")


def model_force(p, K, kd):
    """Series model with the offset limit: f = K_r p while f / K < e_max, beyond it f = K_drv (p - e_max)."""
    if np.isinf(K):
        return kd * p
    kr = 1.0 / (1.0 / K + 1.0 / kd)
    f = kr * p
    return np.where(f / K < E_MAX, f, np.maximum(kd * (p - E_MAX), 0.0))


def episode_medians(e, f):
    """Hold medians of one episode for finger f (scripted lift: end of the squeeze), None if not in contact."""
    idx = hold_idx(e)
    fn = float(np.median(e["fn"][idx, f]))
    if fn <= args.contact_n:
        return None
    on = idx[e["fn"][idx, f] > args.contact_n]
    out = {"f": fn, "p_cmd": float(np.median(e["p_cmd"][on, f])), "kd": e["kd"][on, f]}
    if "p_ref" in e:
        out["p_ref"] = float(np.median(e["p_ref"][on, f]))
    return out


def fig_penetration(sc, demos, rolls, tables):
    """Group medians (source, K) of force vs penetration per finger: arrow reference -> command, drive line."""
    sources = ([("scripted", scripted_eps(sc))] if sc else []) + [("demo", [e for s_ in demos for e in successful(s_)]),
                                                                ("policy", [e for s_ in rolls for e in successful(s_)])]
    fig, axes = finger_axes(1, 4.4, sharey=False)
    rec, pen = {}, {}
    for f in range(5):
        ax = axes[0, f]
        kd_all, xmax, ymax = [], 0.0, 0.0
        for src, eps in sources:
            for k in IL_KS:
                ms = [m for m in (episode_medians(e, f) for e in eps if e["K"] == k) if m is not None]
                if not ms:
                    continue
                F = np.array([m["f"] for m in ms])
                C = 1000 * np.array([m["p_cmd"] for m in ms])
                R = 1000 * np.array([m["p_ref"] for m in ms]) if "p_ref" in ms[0] else None
                kd_all += [m["kd"] for m in ms]
                fm, cm = float(np.median(F)), float(np.median(C))
                col = k_shade(k)
                big = 9 if src == "scripted" else 8
                if len(ms) > 1:  # interquartile range as a cross
                    ax.plot(np.percentile(C, [25, 75]), [fm, fm], color=col, linewidth=1.0, alpha=0.7, zorder=2)
                    ax.plot([cm, cm], np.percentile(F, [25, 75]), color=col, linewidth=1.0, alpha=0.7, zorder=2)
                if R is not None:
                    rm = float(np.median(R))
                    ax.annotate("", xy=(cm, fm), xytext=(rm, fm), zorder=3,
                                arrowprops=dict(arrowstyle="-|>", color=col, linewidth=1.3, shrinkA=4, shrinkB=5))
                    ax.plot(rm, fm, SRC_MARKER[src], markerfacecolor=SURFACE, markeredgecolor=col, markeredgewidth=1.4,
                            markersize=big, zorder=4)
                    xmax = max(xmax, rm)
                ax.plot(cm, fm, SRC_MARKER[src], color=col, markersize=big, zorder=5)
                xmax = max(xmax, float(np.percentile(C, 75)))
                ymax = max(ymax, float(np.percentile(F, 75)))
                pen[f"{src} | {k:g} | {FINGERS[f]}"] = {
                    "n": len(ms), "force_median": fm, "cmd_mm_median": cm,
                    "ref_mm_median": None if R is None else float(np.median(R))}
        ax.set_title(TITLES[f])
        ax.set_xlabel(r"Penetration $\delta$ [mm]")
        if not kd_all:
            ax.text(0.5, 0.5, "no contact", transform=ax.transAxes, ha="center", color=MUTED)
            continue
        kd = float(np.median(np.concatenate(kd_all)))
        rec[FINGERS[f]] = {"kd_median": kd, "kd_iqr": np.percentile(np.concatenate(kd_all), [25, 75]).tolist()}
        g = np.linspace(0, xmax * 1.1, 50)
        ax.plot(g, kd * g / 1000, color=INK, linewidth=1.2, zorder=1)
        ax.set_xlim(0, xmax * 1.1)
        ax.set_ylim(0, ymax * 1.25)
        ax.text(0.04, 0.95, rf"$K_{{\mathrm{{drv}}}}$ ≈ {kd:.0f} N/m", transform=ax.transAxes, va="top", fontsize=9.5, color=INK_2)
    axes[0, 0].set_ylabel(r"Contact force $\Vert f\Vert$ [N]")
    hs = [Line2D([], [], marker=SRC_MARKER[s_], linestyle="", color=MUTED, markersize=8, label=SRC_LABEL[s_])
          for s_, _ in sources]
    hs += [Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE, markeredgecolor=MUTED, markersize=8,
                  label=r"Reference penetration $\delta_{\mathrm{ref}}$"),
           Line2D([], [], marker="o", linestyle="", color=MUTED, markersize=8,
                  label=r"Command penetration $\delta_{\mathrm{cmd}}$"),
           Line2D([], [], color=INK, linewidth=1.2, label=r"Drives $f = K_{\mathrm{drv}}\,\delta_{\mathrm{cmd}}$")]
    fig.legend(handles=k_legend(IL_KS, hs), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    suptitle(fig, "Contact force vs penetration")
    layout(fig, 0.22)
    tables["drive_stiffness"] = rec
    tables["penetration"] = dict(sorted(pen.items()))
    sc_txt = f"; scripted lift: its one run per K, median over the last {args.window_s:g} s of the squeeze" if sc else ""
    save(fig, "penetration",
         "Contact force of each fingertip against its penetration into the cube, along the contact force: median "
         f"over the successful episodes of each source and K (per episode the median over its last {args.hold_s:g} s"
         f"{sc_txt}), cross: interquartile range over the episodes; only fingers in contact. Filled: the command "
         "penetration delta_cmd (component of J (q_cmd - q) along -f: what the drives turn into force). Hollow: "
         "the reference penetration delta_ref of the operator or the script; the arrow from it to the command is "
         "the admittance's yield e = min(|f| / K, 30 mm), longer at the lower K. The policy has no reference: it "
         r"outputs the command directly. Line: the drives f = K_drv delta_cmd, the same at both K, with "
         r"K_drv = 1 / (n^T J K_p^-1 J^T n) (each joint's own gain, joints at a position limit rigid; median over the "
         "panel's samples). K only decides where on this line a grasp ends up. Points below the line: saturated "
         "drives and contacts the sensor does not see (see drive_relation).")


def fig_drive_relation(sc, demos, rolls, tables):
    """Joint space: contact torque -(J^T f)_j vs the drive torque clip(k_p,j (q_cmd - q)_j, +-tau_max,j), per finger."""
    sources = ([("scripted", scripted_eps(sc))] if sc else []) + [("demo", [e for s in demos for e in successful(s)]),
                                                                ("policy", [e for s in rolls for e in successful(s)])]
    fig, axes = finger_axes(1, 4.3, sharey=False)
    rec, all_k = {}, set()
    for f in range(5):
        ax = axes[0, f]
        lo_, hi_ = 0.0, 0.0
        for src, eps in sources:
            xs, ys, lims, sats = [], [], [], []
            for e in eps:
                idx = hold_idx(e)
                if np.median(e["fn"][idx, f]) <= args.contact_n:
                    continue
                x = np.median(e["tau_drive"][idx, f], axis=0)  # (4,) per joint
                y = np.median(e["tau"][idx, f], axis=0)
                lim = e["at_limit"][idx, f].mean(0) > 0.5
                sat = e["saturated"][idx, f].mean(0) > 0.5
                col = k_shade(e["K"])
                kw = dict(s=16 if src != "scripted" else 30, marker=SRC_MARKER[src], zorder=3 if src != "scripted" else 4)
                if (~lim).any():
                    ax.scatter(x[~lim], y[~lim], color=col, alpha=0.55 if src != "scripted" else 0.95, linewidths=0, **kw)
                if lim.any():
                    ax.scatter(x[lim], y[lim], facecolors="none", edgecolors=col, linewidths=1.0, alpha=0.8, **kw)
                xs.append(x); ys.append(y); lims.append(lim); sats.append(sat)
                all_k.add(e["K"])
            if xs:
                xs, ys, lims, sats = map(np.concatenate, (xs, ys, lims, sats))
                free = ~lims
                rec[f"{src} | {FINGERS[f]}"] = {
                    "slope_free_joints": float((xs[free] * ys[free]).sum() / max((xs[free] ** 2).sum(), 1e-12)),
                    "points": int(len(xs)), "share_at_limit": float(lims.mean()), "share_saturated": float(sats.mean())}
                lo_ = min(lo_, float(np.percentile(np.r_[xs, ys], 1)))
                hi_ = max(hi_, float(np.percentile(np.r_[xs, ys], 99)))
        g = np.array([lo_, hi_]) * 1.05
        ax.plot(g, g, color=INK, linewidth=1.2, zorder=2)
        ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
        ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
        ax.set_title(TITLES[f])
        ax.set_xlabel(r"Drive torque $\tau_{d,j}$ [Nm]")
    axes[0, 0].set_ylabel(r"Contact torque $-(J^\top f)_j$ [Nm]")
    hs = [Line2D([], [], marker=SRC_MARKER[s], linestyle="", color=MUTED, markersize=7, label=SRC_LABEL[s])
          for s, _ in sources]
    hs += [Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=MUTED, markersize=7,
                  label="Joint at its position limit"),
           Line2D([], [], color=INK, linewidth=1.2, label="Drive torque balanced by the fingertip force")]
    fig.legend(handles=k_legend(sorted(all_k), hs), loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01))
    suptitle(fig, "Joint drive relation")
    layout(fig, 0.2)
    tables["drive_relation"] = rec
    kp, eff = demos[0]["drive"]["kp"], demos[0]["drive"]["effort_limit"]
    sc_txt = (f"; scripted lift: median over the last {args.window_s:g} s of the squeeze, one marker per joint and K"
              if sc else "")
    save(fig, "drive_relation",
         "Validity check of the force measurement. Torque the fingertip's contact force puts on each joint of its finger, -(J^T f)_j, against the torque the "
         "joint drive applies, tau_d = clip(k_p (q_cmd - q), -tau_max, tau_max) with the joint's own gain "
         f"(k_p = {kp.min():.2g}–{kp.max():.2g} Nm/rad) and effort limit (tau_max = {eff.min():.2g}–{eff.max():.2g} Nm), "
         f"all four joints of the finger. Demonstrations and policy: median over the last {args.hold_s:g} s of each "
         f"successful episode, one marker per joint and episode{sc_txt}; colour: K; only fingers in contact; hollow: the "
         "joint is at its position limit (the limit, not the drive, takes the torque). On the line the drive torque is "
         "balanced by the measured fingertip force alone. Below it, part of the drive torque goes into contacts the "
         "sensor does not see: other finger segments on the cube, the table, the neighbouring fingers, and friction.")


def scripted_ratio(sc, kl, kh, window):
    vals = {}
    for e in sc["eps"]:
        if e["K"] in (kl, kh):
            fn = e["fn"][e["windows"][window]].mean(0)
            vals[e["K"]] = (fn[1:].sum(), fn[0])
    if kl in vals and kh in vals:
        return [vals[kh][0] / vals[kl][0], vals[kh][1] / max(vals[kl][1], 1e-9)]
    return None


def fig_force_ratio(sc, demos, rolls, tables):
    kl, kh = demos[0]["K"], demos[1]["K"]
    entries = []  # (label, [ratio fingers 2-5, ratio thumb], [ci_lo, ci_hi] per group or None)
    r = scripted_ratio(sc, kl, kh, "squeeze") if sc else None
    if r:
        entries.append(("Scripted lift", r, None))
    for src, sets in (("demo", demos), ("policy", rolls)):
        lo = np.array([np.median(e["fn"][hold_idx(e)], 0) for e in successful(sets[0])])
        hi = np.array([np.median(e["fn"][hold_idx(e)], 0) for e in successful(sets[1])])
        rf = ratio_ci(lo[:, 1:].sum(-1), hi[:, 1:].sum(-1))
        rt = ratio_ci(lo[:, 0], hi[:, 0])
        entries.append((SRC_LABEL[src], [rf[0], rt[0]], [rf[1:], rt[1:]]))
    # series prediction from the demonstrations' drive stiffness in the hold
    kd = {}
    for c, sel in ((0, slice(1, 5)), (1, slice(0, 1))):
        v = np.concatenate([e["kd"][hold_idx(e)][:, sel][e["fn"][hold_idx(e)][:, sel] > args.contact_n]
                            for s in demos for e in successful(s)])
        kd[c] = float(np.median(v))
    series = {c: (1 / kl + 1 / kd[c]) / (1 / kh + 1 / kd[c]) for c in kd}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
    x = np.arange(len(entries))
    for c, (ax, title) in enumerate(zip(axes, ("Fingers 2–5 (sum)", "Thumb"))):
        for i, (label, r, ci) in enumerate(entries):
            col = INK_2 if label.startswith("Scripted") else k_shade(kh)
            mk = SRC_MARKER["scripted"] if label.startswith("Scripted") else SRC_MARKER["demo" if label == SRC_LABEL["demo"] else "policy"]
            if ci is not None:
                ax.errorbar(i, r[c], yerr=[[r[c] - ci[c][0]], [ci[c][1] - r[c]]], fmt=mk, color=col, markersize=8,
                            capsize=4, linewidth=1.4)
            else:
                ax.plot(i, r[c], mk, color=col, markersize=8)
            ax.text(i - 0.14, r[c], f"{r[c]:.2f}", ha="right", va="center", fontsize=9.5, color=INK)
        ax.axhline(kh / kl, color=INK, linestyle="--", linewidth=1.2)
        ax.axhline(series[c], color=INK, linestyle=":", linewidth=1.4)
        # line labels in a free strip right of the last marker
        ax.text(len(entries) + 0.75, kh / kl + 0.04, f"Law alone\n{kh / kl:.2f}", ha="right", va="bottom",
                fontsize=9, color=INK_2, linespacing=1.1)
        ax.text(len(entries) + 0.75, series[c] + 0.04, f"Series model\n{series[c]:.2f}", ha="right", va="bottom",
                fontsize=9, color=INK_2, linespacing=1.1)
        ax.axhline(1.0, color=AXIS, linewidth=0.8)
        ax.set_xticks(x, [e[0] for e in entries], fontsize=9.5)
        ax.set_xlim(-0.5, len(entries) + 0.8)
        ax.grid(axis="x", visible=False)
        ax.set_title(title)
    axes[0].set_ylabel(rf"Force ratio $f_{{{kh:g}}} / f_{{{kl:g}}}$")
    axes[0].set_ylim(0, max(kh / kl, max(max(e[1]) for e in entries)) * 1.15)
    suptitle(fig, f"Grip-force ratio K = {kh:g} / {kl:g} N/m")
    layout(fig)
    tables["force_ratio"] = {"entries": [{"source": e[0].replace("\n", " "), "fingers_2_5": e[1][0], "thumb": e[1][1],
                                          "ci": e[2]} for e in entries],
                             "law": kh / kl, "series_model": {"fingers_2_5": series[0], "thumb": series[1]},
                             "kd_demos_hold": {"fingers_2_5": kd[0], "thumb": kd[1]}}
    save(fig, "force_ratio",
         f"Ratio of the contact force at K = {kh:g} to that at K = {kl:g} N/m. Scripted lift: one run per K, force "
         f"averaged over the last {args.window_s:g} s of the squeeze; demonstrations and policy: ratio of the median "
         f"hold force over the successful episodes, with its 95 % bootstrap interval. Both lines are predictions for a "
         f"fingertip pressed equally deep into the cube (same reference penetration) at both K. Dashed: the virtual "
         f"spring alone, with ideally stiff drives, f = K delta_ref, so the ratio is {kh:g} / {kl:g} = {kh / kl:.2f}. "
         f"Dotted: the drives as a spring K_drv in series with it, f = K_r delta_ref, K_r = (1/K + 1/K_drv)^-1, so the ratio "
         f"is (1/{kl:g} + 1/K_drv) / (1/{kh:g} + 1/K_drv), with the "
         f"drive stiffness of the demonstrations in the hold (median K_drv = {kd[0]:.0f} N/m for fingers 2–5, "
         f"{kd[1]:.0f} N/m for the thumb). The thumb's offset reaches its 30 mm limit at both K in the hold, "
         "so its force does not follow either line.")


def law_change(fn, kl, kh):
    """Delta^adm = delta^adm_cmd(s, kh) - delta^adm_cmd(s, kl) at the force fn [m].

    delta^adm_cmd(s, K) = delta_ref(s) - min(|f| / K, e_max) is the admittance's command penetration at the
    state s (q, q_ref, f, J as recorded) with only K changed, at steady state; delta_ref cancels, leaving the
    change of the yield from kh to kl."""
    return np.minimum(fn / kl, E_MAX) - np.minimum(fn / kh, E_MAX)


def fig_kswap(ks_dir, demos, tables, res=None):
    """Delta^pi (change of the policy's command penetration when only K is swapped) against the force, with Delta^adm."""
    path = os.path.join(ks_dir, "kswap.h5")
    if not os.path.exists(path):
        print(f"[fig] kswap skipped: {path} missing")
        return
    kl, kh = IL_KS
    by_k = {s["K"]: s for s in demos}
    with h5py.File(path, "r") as fh:
        groups = {n: {k: fh["demos"][n][k][()] for k in ("pred",)} | dict(fh["demos"][n].attrs) for n in fh["demos"]}
    xs = {fi: [] for fi in range(5)}
    ys = {fi: [] for fi in range(5)}
    kk = {fi: [] for fi in range(5)}
    for n, g in groups.items():
        K = float(g["K"])
        s = by_k[K]
        raw = s["eps"][n.split("_", 1)[1]]["raw"]  # the states the K swap was evaluated on: obs[t]
        fidx = [[c for c, nm in enumerate(s["hand"]) if nm.startswith(f"right_finger{k}_")] for k in range(1, 6)]
        pred_h = np.zeros((len(raw["J"]), 20))
        for fi in range(5):
            pred_h[:, fidx[fi]] = g["pred"][:, fi]
        P = tip(raw["J"], pred_h[:, s["cols"]])  # J (q_cmd(kl) - q_cmd(kh)), (T, 5, 3) [m]
        fn = np.linalg.norm(raw["f"], axis=-1)
        nvec = raw["f"] / np.maximum(fn, 1e-9)[..., None]
        # Delta^pi = delta^pi_cmd(kh) - delta^pi_cmd(kl) = -n . J (q_cmd(kh) - q_cmd(kl)) = n . J (q_cmd(kl) - q_cmd(kh))
        d = (P * nvec).sum(-1)
        for fi in range(5):
            on = fn[:, fi] > args.contact_n
            xs[fi].append(fn[on, fi]); ys[fi].append(d[on, fi]); kk[fi].append(np.full(on.sum(), K))
    fig, axes = finger_axes(1, 4.2, sharey=False)
    rec = {}
    for fi in range(5):
        ax = axes[0, fi]
        x, y, k_ = (np.concatenate(v[fi]) for v in (xs, ys, kk))
        if not len(x):
            ax.text(0.5, 0.5, "no contact", transform=ax.transAxes, ha="center", color=MUTED)
            ax.set_title(TITLES[fi])
            continue
        xm = max(float(np.percentile(x, 99)), 1.0)
        # the policy's median and interquartile range in force bins (at least 30 samples per bin)
        edges = np.linspace(args.contact_n, xm, 13)
        mid, mv, q1, q3 = [], [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (x >= a) & (x < b)
            if sel.sum() >= 30:
                v = 1000 * y[sel]
                mid.append(0.5 * (a + b)); mv.append(np.median(v))
                q1.append(np.percentile(v, 25)); q3.append(np.percentile(v, 75))
        ax.fill_between(mid, q1, q3, color=k_shade(IL_KS[0]), alpha=0.25, linewidth=0, zorder=2)
        ax.plot(mid, mv, color=k_shade(IL_KS[1]), linewidth=2.0, marker="o", markersize=3.5, zorder=4)
        gg = np.linspace(0, xm, 300)
        ax.plot(gg, 1000 * law_change(gg, kl, kh), color=INK, linestyle="--", linewidth=1.3, zorder=3)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        L = law_change(x, kl, kh)
        g_tri = float((y * L).sum() / max((L**2).sum(), 1e-12))
        # the gain shown: as in law_capture (the admittance's change with the joint limits applied), when available
        r_whole = (res or {}).get(f"{FINGERS[fi]} | whole", {})
        gain = float(r_whole["gain"][0]) if "gain" in r_whole else g_tri
        rec[FINGERS[fi]] = {"gain_tip": gain, "gain_tip_triangle": g_tri, "samples": int(len(x))}
        ax.set_xlim(0, xm)
        top = 1000 * law_change(np.array([kl * E_MAX]), kl, kh)[0]
        ax.set_ylim(min(min(q1) if q1 else 0.0, 0.0) - 1.0, max(max(q3) if q3 else 0.0, top) * 1.15)
        ax.set_title(f"{TITLES[fi]}   g = {gain:.2f}")
        ax.set_xlabel(r"Contact force $\Vert f\Vert$ [N]")
    axes[0, 0].set_ylabel(r"Change of command penetration $\Delta$ [mm]")
    hs = [Line2D([], [], color=INK, linestyle="--", linewidth=1.3, label=r"Admittance $\Delta^{\mathrm{adm}}$"),
          Line2D([], [], color=k_shade(IL_KS[1]), linewidth=2.0, marker="o", markersize=3.5,
                 label=r"Policy $\Delta^{\pi}$, median"),
          Patch(color=k_shade(IL_KS[0]), alpha=0.25, label=r"Policy $\Delta^{\pi}$, interquartile range")]
    fig.legend(handles=hs, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    suptitle(fig, "Change of the command penetration with K")
    layout(fig, 0.18)
    tables["kswap_tip"] = rec
    save(fig, "kswap",
         f"Offline K swap: every state s of the {KS_SET} is given to the policy twice, with K = {kl:g} and with "
         f"K = {kh:g} N/m. Solid: the change of its command penetration along the contact force, Delta^pi = "
         f"delta^pi_cmd(s, {kh:g}) - delta^pi_cmd(s, {kl:g}) (first action of the chunk), against the contact force of "
         f"that state: median and interquartile range over all samples with the finger in contact (> {args.contact_n:g} N), "
         "in force bins, demonstrations of both K together. Dashed: the admittance's change at the same state, Delta^adm "
         f"= delta^adm_cmd(s, {kh:g}) - delta^adm_cmd(s, {kl:g}), where delta^adm_cmd(s, K) = delta_ref(s) - min(|f| / K, "
         f"30 mm) is the command penetration the admittance would give with the recorded reference and force and only "
         f"K changed (steady state); delta_ref cancels, so Delta^adm = min(|f| / {kl:g}, 30 mm) - min(|f| / {kh:g}, "
         f"30 mm): it grows up to {kl * E_MAX:g} N, where K = {kl:g} reaches the offset limit, and returns to 0 at "
         f"{kh * E_MAX:g} N, where K = {kh:g} reaches it too. Only one of the two admittance values was recorded (the "
         "demonstration's own K); the other is computed. Delta^pi = 0 means the policy ignores K at that state. "
         "g = sum(Delta^pi Delta^adm) / sum((Delta^adm)^2), the least-squares gain of Delta^pi on Delta^adm through "
         "the origin (1 = as the admittance, 0 = no response), over all contact samples; samples with Delta^adm = 0 do "
         "not enter it. The g in the titles is the whole-episode value of law_capture, where the admittance's change "
         "is built with the joint limits applied (a joint held at its limit cannot change); the dashed triangle "
         "ignores the limits, which matters mostly for the thumb (its gain against the triangle is in tables.json, "
         "gain_tip_triangle).")


def phase_labels(raw, T):
    """Phase of every sample t (state obs[t]) of a demonstration: 0 approach (before the first contact),
    1 closing (first contact to lift-off: cube bottom more than 5 mm above the table), 2 lift (lift-off to the
    start of the hold), 3 hold (the last hold_s; a demonstration ends with the 3 s success hold)."""
    hs = max(T - int(round(args.hold_s / raw.get("dt", 0.02))), 0)
    first = min(raw["first"], T)
    up = np.nonzero((raw["h"] > LIFTOFF_M) & (np.arange(T) >= first))[0]
    lift = int(up[0]) if up.size else hs
    lift = min(max(lift, first), hs)
    ph = np.full(T, 3)
    ph[:hs] = 2
    ph[:lift] = 1
    ph[:first] = 0
    return ph


def compare_with_admittance(ks_dir, dset):
    """Per demonstration state: the policy's command against the admittance's, at the fingertip along the force.

    From kswap.h5 (kswap_test.py): the policy's first chunk action at both K, the recorded action (the
    admittance's q_cmd), q_ref and the clamped mask; from the demo file: J, the force and the cube height at
    the same state obs[t]. Policy commands are clipped to the joint limits (as the env does). The admittance's
    command at the other K is built in joint space, as the controller would build it: the recorded offset
    dq = J^+ e scaled to the other K and clipped to the joint limits like the recorded command,
        q^adm_cmd(s, K') = clip(q_ref + dq * e*(K') / e*(K)),   e*(K) = min(|f| / K, e_max)
    (where no force: the unclamped ratio K / K'), so a joint held at its limit cannot change. Per sample and
    finger, along the force [m]:
        own K:     err_own = delta^pi_cmd(s, K) - delta^adm_cmd(s, K)      (the recorded admittance command)
        swapped K: err_oth = delta^pi_cmd(s, K') - delta^adm_cmd(s, K')
        Delta^pi = delta^pi_cmd(s, 500) - delta^pi_cmd(s, 150), Delta^adm = delta^adm_cmd(s, 500) - delta^adm_cmd(s, 150)
    so err_oth = err_own - (Delta^pi - Delta^adm) taken from K towards K'. Without clipping Delta^adm is the
    triangle min(|f|/150, e_max) - min(|f|/500, e_max) of the kswap figure. Before contact (no force
    direction): the fingertip distance |J (q^pi_cmd - q^adm_cmd)| at the own and the swapped K (the admittance
    changes nothing there). clamped: a joint of the finger was clipped to its limit in the recording."""
    path = os.path.join(ks_dir, "kswap.h5")
    if not os.path.exists(path):
        print(f"[fig] comparison with the admittance skipped: {path} missing")
        return None
    kl, kh = IL_KS
    cols, lim = dset["by_k"][kl]["cols"], dset["by_k"][kl]["drive"]["pos_limits"]
    out = []
    with h5py.File(path, "r") as fh:
        n_arm = len(json.loads(fh.attrs["arm_joints"])) if "arm_joints" in fh.attrs else 6
        if "act_first" not in fh["demos"][next(iter(fh["demos"]))]:
            raise SystemExit(f"{path} has no absolute commands: re-run kswap_test.py (version of 2026-11-08 or later)")
        for gname in fh["demos"]:
            g = fh["demos"][gname]
            K = float(g.attrs["K"])
            ep = dset["by_k"][K]["eps"][gname.split("_", 1)[1]]
            raw = dict(ep["raw"], dt=ep["dt"])
            J, T = raw["J"], len(raw["J"])
            fvec = raw["f"]
            fn = np.linalg.norm(fvec, axis=-1)
            nvec = fvec / np.maximum(fn, 1e-9)[..., None]
            q = g["q_hand"][()].astype(np.float64)
            rec = g["act_rec"][()].astype(np.float64)[:, n_arm:]
            pi = np.clip(g["act_first"][()].astype(np.float64)[:, :, n_arm:], lim[:, 0], lim[:, 1])  # (T, 2, 20)
            own = 0 if K == kl else 1
            K_oth = kh if own == 0 else kl
            pen = lambda v: -(tip(J, v[:, cols]) * nvec).sum(-1)  # noqa: E731  (T, 5) [m], into the cube
            dist = lambda v: np.linalg.norm(tip(J, v[:, cols]), axis=-1)  # noqa: E731
            estar = lambda k: np.minimum(fn / k, E_MAX)  # noqa: E731
            q_ref, dq = g["q_ref"][()].astype(np.float64), g["dq"][()].astype(np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(fn > 1e-6, estar(K_oth) / estar(K), K / K_oth)  # (T, 5) e*(K') / e*(K)
            r20 = np.ones((T, 20))
            for fi in range(5):
                r20[:, cols[fi]] = ratio[:, fi:fi + 1]
            q_adm_oth = np.clip(q_ref + dq * r20, lim[:, 0], lim[:, 1])
            a_own, a_oth = pen(rec - q), pen(q_adm_oth - q)
            p_own, p_oth = pen(pi[:, own] - q), pen(pi[:, 1 - own] - q)
            p_lo, p_hi = (p_own, p_oth) if own == 0 else (p_oth, p_own)
            a_lo, a_hi = (a_own, a_oth) if own == 0 else (a_oth, a_own)
            clamped = g["clamped"][()].astype(bool)[:, cols].any(-1)  # (T, 5)
            out.append({"K": K, "name": gname, "phase": phase_labels(raw, T), "fn": fn, "clamped": clamped,
                        "err_own": p_own - a_own, "err_oth": p_oth - a_oth,
                        "yield_own": pen(q_ref - q) - a_own, "yield_oth": pen(q_ref - q) - a_oth,
                        "ypi_own": pen(q_ref - q) - p_own, "ypi_oth": pen(q_ref - q) - p_oth,
                        "d_pi": p_hi - p_lo, "d_adm": a_hi - a_lo,
                        "dist_own": dist(pi[:, own] - rec), "dist_oth": dist(pi[:, 1 - own] - rec)})
    return out


def summarise_comparison(cmp):
    """Per finger and phase (approach, closing, lift, hold, whole episode): gain g with a 95 % bootstrap over the
    demonstrations, and median absolute / signed errors and yields [mm]."""
    rng = np.random.default_rng(0)
    res = {}
    for fi in range(5):
        for pname in PHASES + ["whole"]:
            m = []
            for d in cmp:
                if pname == "approach":
                    sel = (d["phase"] == 0)
                else:
                    sel = d["fn"][:, fi] > args.contact_n
                    if pname != "whole":
                        sel &= d["phase"] == PHASES.index(pname)
                m.append(sel)
            n = int(sum(x.sum() for x in m))
            n_demos = int(sum(x.any() for x in m))
            key = f"{FINGERS[fi]} | {pname}"
            if n < MIN_CELL_SAMPLES or n_demos < MIN_CELL_DEMOS:
                res[key] = {"samples": n, "demos": n_demos}
                continue
            if pname == "approach":
                do = 1000 * np.concatenate([d["dist_own"][x, fi] for d, x in zip(cmp, m)])
                dx = 1000 * np.concatenate([d["dist_oth"][x, fi] for d, x in zip(cmp, m)])
                res[key] = {"samples": n, "abs_own": float(np.median(do)), "abs_oth": float(np.median(dx)),
                            "iqr_own": np.percentile(do, [25, 75]).tolist(), "iqr_oth": np.percentile(dx, [25, 75]).tolist()}
                continue
            cat = lambda k: 1000 * np.concatenate([d[k][x, fi] for d, x in zip(cmp, m)])  # noqa: E731
            eo, ex, yo, yx = cat("err_own"), cat("err_oth"), cat("yield_own"), cat("yield_oth")
            num = np.array([(d["d_pi"][x, fi] * d["d_adm"][x, fi]).sum() for d, x in zip(cmp, m)])
            den = np.array([(d["d_adm"][x, fi] ** 2).sum() for d, x in zip(cmp, m)])
            g = float(num.sum() / den.sum()) if den.sum() > 0 else float("nan")
            bi = rng.integers(0, len(cmp), (args.boot, len(cmp)))
            with np.errstate(invalid="ignore", divide="ignore"):
                gb = num[bi].sum(1) / den[bi].sum(1)
            lo, hi = (np.nanpercentile(gb, [2.5, 97.5]).tolist() if np.isfinite(gb).any() else [np.nan, np.nan])
            yr = {}
            for which in ("own", "oth"):  # the policy's yield on the admittance's: least squares through the origin
                nu = np.array([(d[f"ypi_{which}"][x, fi] * d[f"yield_{which}"][x, fi]).sum() for d, x in zip(cmp, m)])
                de = np.array([(d[f"yield_{which}"][x, fi] ** 2).sum() for d, x in zip(cmp, m)])
                with np.errstate(invalid="ignore", divide="ignore"):
                    rb = nu[bi].sum(1) / de[bi].sum(1)
                yr[which] = [float(nu.sum() / de.sum()) if de.sum() > 0 else float("nan")] + (
                    np.nanpercentile(rb, [2.5, 97.5]).tolist() if np.isfinite(rb).any() else [np.nan, np.nan])
            ae, ax_ = np.abs(eo), np.abs(ex)
            res[key] = {"samples": n, "gain": [g, lo, hi],
                        "abs_own": float(np.median(ae)), "abs_oth": float(np.median(ax_)),
                        "iqr_own": np.percentile(ae, [25, 75]).tolist(), "iqr_oth": np.percentile(ax_, [25, 75]).tolist(),
                        "signed_own": float(np.median(eo)), "signed_oth": float(np.median(ex)),
                        "yield_own": float(np.median(np.abs(yo))), "yield_oth": float(np.median(np.abs(yx))),
                        "rel_own": float(np.median(ae) / max(np.median(np.abs(yo)), 1e-9)),
                        "rel_oth": float(np.median(ax_) / max(np.median(np.abs(yx)), 1e-9)),
                        "yield_ratio_own": yr["own"], "yield_ratio_oth": yr["oth"],
                        "share_clamped": float(np.concatenate([d["clamped"][x, fi] for d, x in zip(cmp, m)]).mean())}
    return res


CMP_PHASES = ["closing", "lift", "hold", "whole"]
MIN_CELL_SAMPLES, MIN_CELL_DEMOS = 100, 3  # a finger x phase cell is shown only with this much data
CMP_LABELS = {"approach": "Approach", "closing": "Closing", "lift": "Lift", "hold": "Hold", "whole": "Whole\nepisode"}
OWN_COL, OTH_COL = "#2a78d6", "#eb6834"


def fig_law_capture(res, n_demos):
    """Gain g per finger and phase, validation demonstrations."""
    fig, axes = finger_axes(1, 3.9, sharey=True)
    x = np.arange(len(CMP_PHASES))
    for fi in range(5):
        ax = axes[0, fi]
        for xi, pname in enumerate(CMP_PHASES):
            r = res.get(f"{FINGERS[fi]} | {pname}", {})
            if "gain" not in r or not np.isfinite(r["gain"][0]):
                ax.text(xi, 0.03, "n/a", ha="center", va="bottom", fontsize=8.5, color=MUTED,
                        transform=ax.get_xaxis_transform())  # too little data in this phase
                continue
            g, lo, hi = r["gain"]
            mk = "s" if pname == "whole" else "o"
            ax.errorbar(xi, g, yerr=[[max(g - lo, 0)], [max(hi - g, 0)]] if np.isfinite(lo) else None, fmt=mk,
                        color=INK_2, markersize=7, capsize=3, linewidth=1.3, zorder=3)
            ax.text(xi + 0.12, g, f"{g:.2f}", va="center", ha="left", fontsize=8.5, color=INK)
        ax.axhline(1.0, color=INK, linestyle="--", linewidth=1.1, zorder=1)
        ax.axhline(0.0, color=AXIS, linewidth=0.9, zorder=1)
        ax.axvline(len(CMP_PHASES) - 1.5, color=GRID, linewidth=0.8, zorder=0)
        ax.set_xticks(x, [CMP_LABELS[p] for p in CMP_PHASES], fontsize=9)
        ax.set_xlim(-0.5, len(CMP_PHASES) - 0.3)
        ax.grid(axis="x", visible=False)
        ax.set_title(TITLES[fi])
    axes[0, 0].set_ylabel(r"Gain $g$ of $\Delta^{\pi}$ on $\Delta^{\mathrm{adm}}$")
    lo_all = min([0.0] + [r["gain"][1] for r in res.values() if "gain" in r and np.isfinite(r["gain"][1])])
    hi_all = max([1.0] + [r["gain"][2] for r in res.values() if "gain" in r and np.isfinite(r["gain"][2])])
    axes[0, 0].set_ylim(max(lo_all, -1.0) - 0.1, min(hi_all, 2.0) + 0.1)
    hs = [Line2D([], [], color=INK, linestyle="--", linewidth=1.1, label="Admittance law (g = 1)"),
          Line2D([], [], marker="o", color=INK_2, linestyle="", markersize=7, label="Policy, per phase [95 % CI]"),
          Line2D([], [], marker="s", color=INK_2, linestyle="", markersize=7, label="Policy, whole episode")]
    fig.legend(handles=hs, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    suptitle(fig, "Share of the admittance law captured by the policy")
    layout(fig, 0.14)
    kl, kh = IL_KS
    save(fig, "law_capture",
         f"How much of the admittance law's K-dependence the policy has captured, on the {KS_SET} ({n_demos} per K). "
         f"Every state s is given to the policy with K = {kl:g} and {kh:g} N/m (first action of the chunk); the change "
         f"of its command penetration along the contact force, Delta^pi = delta^pi_cmd(s, {kh:g}) - delta^pi_cmd(s, {kl:g}), "
         "is compared with the admittance's change at the same state, Delta^adm = delta^adm_cmd(s, "
         f"{kh:g}) - delta^adm_cmd(s, {kl:g}): the recorded command at the demonstration's own K, and at the other K the "
         "recorded offset dq = J^+ e scaled by e*(K') / e*(K), e* = min(|f| / K, 30 mm), with the joint limits applied as "
         "in the recording (without a joint at its limit this is min(|f| / 150, 30 mm) - min(|f| / 500, 30 mm), the "
         "kswap figure's triangle). g = sum(Delta^pi Delta^adm) / sum((Delta^adm)^2), the least-squares gain through the "
         "origin: 1 = the policy changes its command as the admittance would, 0 = it ignores K. Bars: 95 % bootstrap "
         "interval over the demonstrations. Phases (from each demonstration): closing = first contact to lift-off "
         f"(cube bottom 5 mm above the table), lift = lift-off to the hold, hold = the last {args.hold_s:g} s (the success "
         "hold); whole episode = all of them. Only samples with the finger in contact (> "
         f"{args.contact_n:g} N); samples where the admittance changes nothing (both K at the 30 mm offset limit, or "
         "the joints held at their limits) do not enter g. Before contact the law asks "
         "for no change, so g is not defined there (see command_error and precontact). n/a: fewer than "
         f"{MIN_CELL_SAMPLES} samples or fewer than {MIN_CELL_DEMOS} demonstrations with the finger in contact in that phase.")


def fig_command_error(res, n_demos):
    """Median |error| of the policy's command against the admittance's, own and swapped K, per finger and phase."""
    phases = ["approach"] + CMP_PHASES
    x = np.arange(len(phases))
    fig, axes = finger_axes(2, 7.4, sharey="row")
    top, bottom = [0.0, 0.0], 0.0
    for fi in range(5):
        for row in (0, 1):
            ax = axes[row, fi]
            for xi, pname in enumerate(phases):
                r = res.get(f"{FINGERS[fi]} | {pname}", {})
                if "abs_own" not in r:
                    ax.text(xi, 0.03, "n/a", ha="center", va="bottom", fontsize=8, color=MUTED,
                            transform=ax.get_xaxis_transform())
                    continue
                if row == 1 and pname == "approach":
                    continue  # no yield before contact
                pairs = (("own", OWN_COL, -0.13), ("oth", OTH_COL, 0.13)) if row == 0 else (("own", OWN_COL, 0.0),)
                for which, col, dx in pairs:
                    hollow = pname == "approach"
                    if row == 0:
                        v = r[f"abs_{which}"]
                        q1, q3 = r[f"iqr_{which}"]
                        top[0] = max(top[0], q3)
                    else:
                        v, q1, q3 = (100 * c for c in r[f"yield_ratio_{which}"])
                        if not np.isfinite(v):
                            continue
                        q1, q3 = (v, v) if not (np.isfinite(q1) and np.isfinite(q3)) else (q1, q3)
                        top[1], bottom = max(top[1], q3), min(bottom, q1)
                    ax.plot([xi + dx] * 2, [q1, q3], color=col, linewidth=1.2, alpha=0.7, zorder=2)
                    ax.plot(xi + dx, v, "s" if pname == "whole" else "o", color=col, markersize=6.5, zorder=3,
                            markerfacecolor=SURFACE if hollow else col, markeredgewidth=1.4)
            ax.axvline(len(phases) - 1.5, color=GRID, linewidth=0.8, zorder=0)
            ax.set_xticks(x, [CMP_LABELS[p] for p in phases], fontsize=8.5, rotation=35, ha="right",
                          rotation_mode="anchor")
            ax.set_xlim(-0.5, len(phases) - 0.5)
            ax.grid(axis="x", visible=False)
            if row == 0:
                ax.set_title(TITLES[fi])
                ax.tick_params(labelbottom=False)
    # shared rows: one limit from all panels (setting it per panel would freeze the first)
    axes[0, 0].set_ylim(0, max(top[0], 1.0) * 1.08)
    span = max(top[1], 110.0) - min(bottom, 0.0)
    axes[1, 0].set_ylim(min(bottom, 0.0) - 0.05 * span, max(top[1], 110.0) + 0.05 * span)
    for fi in range(5):
        axes[1, fi].axhline(100, color=INK, linestyle="--", linewidth=1.1, zorder=1)
        axes[1, fi].axhline(0, color=AXIS, linewidth=0.9, zorder=1)
    axes[0, 0].set_ylabel(r"Command error $|\delta^{\pi}_{\mathrm{cmd}} - \delta^{\mathrm{adm}}_{\mathrm{cmd}}|$ [mm]")
    axes[1, 0].set_ylabel("Policy yield / admittance yield\nat the own K [%]")
    kl, kh = IL_KS
    hs = [Line2D([], [], marker="o", color=OWN_COL, linestyle="", markersize=7, label="Demonstration's own K"),
          Line2D([], [], marker="o", color=OTH_COL, linestyle="", markersize=7, label="Swapped K"),
          Line2D([], [], color=MUTED, linewidth=1.2, label="Top: interquartile range; bottom: 95 % CI"),
          Line2D([], [], color=INK, linestyle="--", linewidth=1.1, label="Same yield as the admittance"),
          Line2D([], [], marker="o", markerfacecolor=SURFACE, markeredgecolor=MUTED, linestyle="", markersize=7,
                 label="Approach: fingertip distance")]
    fig.legend(handles=hs, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    suptitle(fig, "Policy command against the admittance's")
    layout(fig, 0.08)
    save(fig, "command_error",
         f"The policy's finger command against the admittance controller's at the same state, on the {KS_SET} "
         f"({n_demos} per K; first action of the chunk, clipped to the joint limits). Blue: at the demonstration's own K, "
         "against the recorded admittance command, err = delta^pi_cmd(s, K) - delta^adm_cmd(s, K). Orange: at the "
         "swapped K, against the command the admittance would give there with the same reference and force "
         "(the recorded offset scaled by e*(K') / e*(K), e* = min(|f| / K, 30 mm), joint limits applied); the "
         "difference between the two is the K-swap shortfall. Top: median absolute error at the fingertip along "
         "the contact force [mm] with its interquartile range; approach (hollow, no force yet): fingertip distance "
         "|J (q^pi_cmd - q^adm_cmd)| (the admittance changes nothing there, so at the swapped K any difference is the "
         "policy's own use of K). Bottom, at the demonstration's own K only: how far the policy's command stays "
         "behind the operator's reference compared with the admittance's, the policy's yield e_pi = delta_ref - "
         "delta^pi_cmd against the admittance's e_adm = delta_ref - delta^adm_cmd, least-squares slope through the "
         "origin, sum(e_pi e_adm) / sum(e_adm^2), with a 95 % bootstrap interval over the demonstrations; 100 % = the "
         "same yield as the admittance, 0 % = none (the command at the operator's reference). This is the level of "
         "the pull-back, i.e. imitation accuracy on unseen states: in steady contact the drive relation and the "
         "admittance law predict the same command, so it does not show which of the two the policy uses (that is g "
         "in law_capture). The thumb is close to 100 % because in the hold its offset sits at the 30 mm limit at "
         "both K and its command barely leaves the actual thumb position (joints at their limits), not because of "
         "compliance. Not shown at the swapped K: pooled over both swap directions the ratio of a policy that "
         "ignores K is not fixed (about 333 % for 150 -> 500, 30 % for 500 -> 150). Phases and sample selection as "
         "in law_capture.")


def comparison_table_rows(val, train):
    rows = []
    fmt_g = lambda r: ("–" if "gain" not in r or not np.isfinite(r["gain"][0])  # noqa: E731
                       else f"{r['gain'][0]:.2f} [{r['gain'][1]:.2f}, {r['gain'][2]:.2f}]")
    for fi in range(5):
        for pname in PHASES + ["whole"]:
            key = f"{FINGERS[fi]} | {pname}"
            v, t = val.get(key, {}), (train or {}).get(key, {})
            if "abs_own" not in v:
                rows.append([TITLES[fi], pname, str(v.get("samples", 0))] + ["–"] * 7)
                continue
            tr_g = fmt_g(t).split(" [")[0] if t else "–"
            tr_e = (f"{t['abs_own']:.1f} / {t['abs_oth']:.1f}" if "abs_own" in t else "–")
            sg = "–" if pname == "approach" else f"{v['signed_own']:+.1f} / {v['signed_oth']:+.1f}"
            yl = "–" if pname == "approach" else f"{v['yield_own']:.1f} / {v['yield_oth']:.1f}"
            yr = ("–" if pname == "approach" else
                  f"{100 * v['yield_ratio_own'][0]:.0f} % [{100 * v['yield_ratio_own'][1]:.0f}, "
                  f"{100 * v['yield_ratio_own'][2]:.0f}]")
            rows.append([TITLES[fi], pname, str(v["samples"]), fmt_g(v), tr_g,
                         f"{v['abs_own']:.1f} / {v['abs_oth']:.1f}", tr_e, sg, yl, yr])
    return rows


def fig_precontact(rolls, tables):
    lo, hi = rolls
    dt = lo["dt"]
    lag = np.arange(int(round(-3.0 / dt)), int(round(1.5 / dt)) + 1)
    rows_f, rows_a, starts = {fi: [] for fi in range(5)}, [], []
    for n in sorted(set(lo["eps"]) & set(hi["eps"])):
        a, b = lo["eps"][n]["raw"], hi["eps"][n]["raw"]
        T = min(len(a["q_cmd"]), len(b["q_cmd"]))
        tc = min(a["first"], b["first"], T)
        if tc < 2:
            continue
        dtip = np.linalg.norm(tip(a["J"][:T], (a["q_cmd"][:T] - b["q_cmd"][:T])[:, lo["cols"]]), axis=-1)  # (T, 5)
        darm = np.abs(a["arm"][:T] - b["arm"][:T]).max(-1)
        idx = tc + lag
        ok = (idx >= 0) & (idx < T)
        for fi in range(5):
            w = np.full(len(lag), np.nan)
            w[ok] = 1000 * dtip[idx[ok], fi]
            rows_f[fi].append(w)
        w = np.full(len(lag), np.nan)
        w[ok] = 1000 * darm[idx[ok]]
        rows_a.append(w)
        starts.append((1000 * dtip[0], 1000 * darm[0], 1000 * dtip[tc - 1], 1000 * darm[tc - 1], tc * dt))
    t = lag * dt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), gridspec_kw={"width_ratios": [1.4, 1]})
    for fi in range(5):
        med, q1, q3 = band(rows_f[fi])
        axes[0].fill_between(t, q1, q3, color=FINGER_COLOR[fi], alpha=0.10, lw=0)
        axes[0].plot(t, med, color=FINGER_COLOR[fi], label=TITLES[fi])
    med, q1, q3 = band(rows_a)
    axes[1].fill_between(t, q1, q3, color=INK_2, alpha=0.12, lw=0)
    axes[1].plot(t, med, color=INK_2)
    for ax in axes:
        ax.axvline(0, color=AXIS, linewidth=1.0)
        ax.set_xlabel("Time from the first contact [s]")
    axes[0].set_title("Finger commands at the fingertip")
    axes[0].set_ylabel(r"$\Vert J\,(q_{\mathrm{cmd}}^{150} - q_{\mathrm{cmd}}^{500})\Vert$ [mm]")
    axes[0].legend(loc="upper left", ncol=1)
    axes[1].set_title("Arm targets (largest joint)")
    axes[1].set_ylabel("Target difference [mrad]")
    suptitle(fig, "Effect of K on the policy before contact")
    layout(fig)
    st = np.array([np.r_[s[0], s[1], s[2], s[3], s[4]] for s in starts])
    tables["precontact"] = {"pairs": len(starts), "first_contact_s_median": float(np.median(st[:, -1])),
                            "first_step_tip_mm_median": np.median(st[:, 0:5], 0).tolist(),
                            "first_step_arm_mrad_median": float(np.median(st[:, 5])),
                            "last_before_contact_tip_mm_median": np.median(st[:, 6:11], 0).tolist(),
                            "last_before_contact_arm_mrad_median": float(np.median(st[:, 11]))}
    save(fig, "precontact",
         f"The policy at the same {len(starts)} cube start positions with K = {lo['K']:g} and {hi['K']:g} N/m: "
         "difference of its outputs, aligned to the first fingertip contact in either run (median and interquartile "
         "range over the pairs). Left: finger commands, mapped to the fingertip with J; right: arm joint targets. "
         "The start states are identical, so every difference is caused by the K input. Before contact there is no "
         "force, so an admittance would make no difference there.")


def fig_single(sets, names, title, fname, caption, show_yield):
    """One episode per K: cube height, and per finger the force, (the yield) and the command penetration."""
    rows = [("fn", r"Contact force $\Vert f\Vert$ [N]", 1.0)]
    if show_yield:
        rows.append(("e", r"Yield $e$ [mm]", 1000.0))
    rows.append(("p_cmd", r"Command penetration $\delta_{\mathrm{cmd}}$ [mm]", 1000.0))
    fig = plt.figure(figsize=(15, 2.2 + 2.5 * len(rows)))
    gs = GridSpec(1 + len(rows), 5, figure=fig, height_ratios=[0.8] + [1] * len(rows))
    axh = fig.add_subplot(gs[0, :])
    axes = [[fig.add_subplot(gs[1 + r, f], sharex=axh) for f in range(5)] for r in range(len(rows))]
    for s, n in zip(sets, names):
        e = s["eps"][n]
        t = np.arange(e["T"]) * e["dt"]
        col = k_shade(s["K"])
        axh.plot(t, 100 * e["h"], color=col, label=f"K = {s['K']:g} N/m")
        for r, (key, _, sc) in enumerate(rows):
            for f in range(5):
                y = sc * e[key][:, f]
                if key != "fn":
                    y = np.where(e["fn"][:, f] > args.contact_n, y, np.nan)  # only meaningful in contact
                axes[r][f].plot(t, y, color=col, linewidth=1.5)
    axh.set_ylabel("Cube height [cm]")
    axh.legend(loc="upper left")
    axh.set_ylim(bottom=0)
    axh.tick_params(labelbottom=False)
    for r, (_, label, _) in enumerate(rows):
        axes[r][0].set_ylabel(label)
        for f in range(5):
            ax = axes[r][f]
            ax.set_ylim(bottom=0, top=max(ax.get_ylim()[1], 1.0))
            if r == 0:
                ax.set_title(TITLES[f])
            if r == len(rows) - 1:
                ax.set_xlabel("Time [s]")
            else:
                ax.tick_params(labelbottom=False)
    suptitle(fig, title)
    layout(fig)
    save(fig, fname, caption)


def pick_demo(s):
    return min(s["eps"], key=lambda n: math.hypot(*(s["eps"][n]["cube0"] - np.array(CENTRE))))


def pick_pair(rolls):
    lo, hi = rolls
    if args.pair_episode is not None:
        name = f"ep_{args.pair_episode:03d}"
        if name not in lo["eps"] or name not in hi["eps"]:
            raise SystemExit(f"episode {args.pair_episode} is not in both rollout files")
        return name
    both = [n for n in lo["eps"] if n in hi["eps"] and lo["eps"][n]["success"] and hi["eps"][n]["success"]]
    return min(both, key=lambda n: math.hypot(*(lo["eps"][n]["cube0"] - np.array(CENTRE))))


# ---------------------------------------------------------------------------------------------------
# tables and captions
# ---------------------------------------------------------------------------------------------------
def write_tables(t, ks):
    md, tex = [], []

    def table(title, head, rows, align):
        md.extend([f"### {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                  + ["| " + " | ".join(r) + " |" for r in rows] + [""])
        esc = lambda s: s.replace("%", r"\%").replace("–", "--").replace("∞", r"$\infty$")  # noqa: E731
        tex.extend([r"\begin{table}[t]\centering\small", r"\caption{" + esc(title) + "}",
                    r"\begin{tabular}{" + align + r"}\toprule", " & ".join(esc(h) for h in head) + r"\\\midrule"]
                   + [" & ".join(esc(c) for c in r) + r"\\" for r in rows] + [r"\bottomrule\end{tabular}\end{table}", ""])

    m = t["mcnemar"]
    table(f"Task outcome ({t['outcome'][0]['n']} rollouts per K, same cube positions; exact McNemar p = {m['p']:.2f})",
          ["K [N/m]", "success", "95 % CI", "lifted, no 3 s hold", "lifted < 15 cm", "never lifted", "mean time to success [s]"],
          [[f"{r['K']:g}", f"{r['k']}/{r['n']}", f"{100 * r['lo']:.0f}–{100 * r['hi']:.0f} %", str(r["nohold"]),
            str(r["partial"]), str(r["never"]), f"{r['t_mean']:.1f}"] for r in t["outcome"]], "lcccccc")

    rows = []
    for key, v in t["hold_force"].items():
        src, fin = key.split(" | ")
        r = v["ratio"]
        rows.append([SRC_LABEL[src], fin] + [f"{v['median'][i]:.2f} [{v['iqr'][i][0]:.1f}, {v['iqr'][i][1]:.1f}]" for i in (0, 1)]
                    + ["–" if not np.isfinite(r[0]) else f"{r[0]:.2f} [{r[1]:.2f}, {r[2]:.2f}]"])
    table("Grip force in the hold [N], median [IQR] over successful episodes; ratio of the medians [95 % bootstrap] "
          "(– : a median below 0.5 N)", ["source", "finger", f"K = {ks[0]:g}", f"K = {ks[1]:g}", "ratio"], rows, "llccc")

    def fmt_ratio(r):
        return "–" if not np.isfinite(r[0]) else f"{r[0]:.2f} [{r[1]:.2f}, {r[2]:.2f}]"

    rows = []
    for key, v in t["hold_penetration"].items():
        src, fin = key.split(" | ")
        ref = v.get("ref_median")
        rows.append([SRC_LABEL[src], fin] + [f"{v['median'][i]:.1f} [{v['iqr'][i][0]:.1f}, {v['iqr'][i][1]:.1f}]"
                                             for i in (0, 1)]
                    + [fmt_ratio(v["ratio"]), f"{v['n'][0]} / {v['n'][1]}",
                       "–" if ref is None else f"{ref[0]:.1f} / {ref[1]:.1f}"])
    table("Command penetration delta_cmd in the hold [mm], median [IQR] over the successful episodes with the finger in "
          "contact; ratio of the medians K high / K low [95 % bootstrap] (– : fewer than 3 episodes or a median below "
          "1 mm); the demonstrations' median reference penetration delta_ref",
          ["source", "finger", f"K = {ks[0]:g}", f"K = {ks[1]:g}", "ratio", "episodes", "delta_ref (demos)"], rows,
          "llccccc")

    def pct(r):
        if not np.isfinite(r[0]):
            return "–"
        up = "∞" if not np.isfinite(r[2]) else f"{100 * r[2]:.0f}"
        return f"{100 * r[0]:.0f} % [{100 * r[1]:.0f}, {up}]"

    rows = []
    for fin, v in t["hold_share"].items():
        rows.append([fin] + sum([[pct(v["mean"][i]), pct(v["median"][i]),
                                  f"{100 * v['demo_in_contact'][i]:.0f} / {100 * v['policy_in_contact'][i]:.0f} %"]
                                 for i in (0, 1)], []))
    table("Hold force of the policy as a share of the demonstrations' at the same K: ratio of the means and of the "
          "medians over the successful episodes [95 % bootstrap; ∞: open upper bound] (– : the demonstrations' mean "
          "below 0.2 N / median below 0.5 N); share of the episodes with the finger in contact, demos / policy",
          ["finger", f"K = {ks[0]:g}: mean", f"K = {ks[0]:g}: median", f"K = {ks[0]:g}: in contact",
           f"K = {ks[1]:g}: mean", f"K = {ks[1]:g}: median", f"K = {ks[1]:g}: in contact"], rows, "lcccccc")

    rows = []
    for key, pair in t["hold_policy_vs_demo"].items():
        fin = key.split(" | ")[1]
        rows.append([fin] + sum([[("–" if not np.isfinite(p["diff"]) else f"{p['diff']:+.1f}"), pct(p["ratio"])]
                                 for p in pair], []))
    table("Command penetration delta_cmd in the hold, policy against the demonstrations at the same K: difference of "
          "the medians (policy − demos) [mm] and their ratio policy / demos [95 % bootstrap; ∞: open upper bound] "
          "(– : fewer than 3 episodes in contact or a median below 1 mm)",
          ["finger", f"K = {ks[0]:g}: difference", f"K = {ks[0]:g}: ratio", f"K = {ks[1]:g}: difference",
           f"K = {ks[1]:g}: ratio"], rows, "lcccc")

    fr = t["force_ratio"]
    rows = [[e["source"], f"{e['fingers_2_5']:.2f}" + (f" [{e['ci'][0][0]:.2f}, {e['ci'][0][1]:.2f}]" if e["ci"] else ""),
             f"{e['thumb']:.2f}" + (f" [{e['ci'][1][0]:.2f}, {e['ci'][1][1]:.2f}]" if e["ci"] else "")] for e in fr["entries"]]
    rows += [["admittance law alone", f"{fr['law']:.2f}", f"{fr['law']:.2f}"],
             ["series model (K_drv of the demonstrations)", f"{fr['series_model']['fingers_2_5']:.2f}",
              f"{fr['series_model']['thumb']:.2f}"]]
    table(f"Grip-force ratio K = {ks[1]:g} / {ks[0]:g} N/m", ["source", "fingers 2–5 (sum)", "thumb"], rows, "lcc")

    rows = [[k, f"{v['kd_median']:.0f}", f"{v['kd_iqr'][0]:.0f}–{v['kd_iqr'][1]:.0f}"] for k, v in t["drive_stiffness"].items()]
    table("Drive stiffness at the fingertip along the contact force, K_drv [N/m], hold samples in contact (all sources)",
          ["finger", "median", "IQR"], rows, "lcc")
    rows = []
    for k, v in t["penetration"].items():
        src, kk, fin = k.split(" | ")
        rows.append([SRC_LABEL[src], kk, fin, str(v["n"]), f"{v['force_median']:.2f}", f"{v['cmd_mm_median']:.1f}",
                     "–" if v["ref_mm_median"] is None else f"{v['ref_mm_median']:.1f}",
                     "–" if v["ref_mm_median"] is None else f"{v['ref_mm_median'] - v['cmd_mm_median']:.1f}"])
    table("Hold medians per finger in contact: force [N], command and reference penetration [mm], their gap (the yield) [mm]",
          ["source", "K [N/m]", "finger", "episodes", "force", "delta_cmd", "delta_ref", "gap"], rows, "lllccccc")

    rows = [[SRC_LABEL[k.split(" | ")[0]], k.split(" | ")[1], f"{v['slope_free_joints']:.2f}",
             f"{100 * v['share_at_limit']:.0f} %", f"{100 * v['share_saturated']:.0f} %", str(v["points"])]
            for k, v in t["drive_relation"].items()]
    table("Joint drive relation: share of the drive torque balanced by the fingertip contact (slope of -(J^T f)_j "
          "over tau_d through the origin, joints not at a limit; 1 = all of it)",
          ["source", "finger", "slope", "joints at a limit", "drive saturated", "points"], rows, "llcccc")

    if "kswap_tip" in t:
        js = t.get("kswap_joint", {})
        rows = [[f, f"{v['gain_tip']:.2f}", f"{v['gain_tip_triangle']:.2f}",
                 (f"{js[f][0]:.2f} [{js[f][1]:.2f}, {js[f][2]:.2f}]" if f in js else "–"),
                 str(v["samples"])] for f, v in t["kswap_tip"].items()]
        table(f"K swap on the states of the {KS_SET}: gain g of the policy's change of the command penetration "
              "Delta^pi on the admittance's Delta^adm (1 = as the admittance, 0 = ignores K), in contact",
              ["finger", "gain (joint limits applied, as law_capture)", "gain against the triangle (no limits)",
               "gain in joint space [95 % CI] (kswap_test.py)", "samples"], rows, "lcccc")
    if "comparison" in t:
        table(f"Policy against the admittance at the same state ({KS_SET}; training demonstrations as a reference): "
              "gain g [95 % CI] per phase, median |error| of the command penetration at the own / swapped K [mm], "
              "median signed error own / swapped [mm], median yield own / swapped [mm], the policy's yield as a share "
              "of the admittance's at the own K (least squares); approach: fingertip distance",
              ["finger", "phase", "samples", "g", "g (training)", "|error| own / swapped", "|error| (training)",
               "signed error", "yield", "policy / admittance yield, own K [95 % CI]"],
              comparison_table_rows(t["comparison"], t.get("comparison_train")), "llcccccccc")
    p = t["precontact"]
    rows = [["first step (identical state)"] + [f"{v:.1f}" for v in p["first_step_tip_mm_median"]] + [f"{p['first_step_arm_mrad_median']:.0f}"],
            ["last step before contact"] + [f"{v:.1f}" for v in p["last_before_contact_tip_mm_median"]]
            + [f"{p['last_before_contact_arm_mrad_median']:.0f}"]]
    table(f"Effect of K on the policy before contact ({p['pairs']} paired rollouts, median): finger commands at the "
          "fingertip [mm], arm targets [mrad]", ["", *TITLES, "arm"], rows, "lcccccc")

    with open(os.path.join(OUT, "tables.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(OUT, "tables.tex"), "w") as fh:
        fh.write("% needs \\usepackage{booktabs}\n" + "\n".join(tex) + "\n")
    with open(os.path.join(OUT, "tables.json"), "w") as fh:
        json.dump({k: v for k, v in t.items()}, fh, indent=1,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else (bool(o) if isinstance(o, np.bool_) else str(o)))
    print("\n".join(md))
    print(f"[tables] {OUT}/tables.md, tables.tex, tables.json")


def write_captions():
    with open(os.path.join(OUT, "captions.md"), "w") as fh:
        fh.write("# Figure captions\n\n" + "\n\n".join(f"**{n}.png** ({ROLES.get(n, '')}): {c}" for n, c in CAPTIONS.items()) + "\n")
    print(f"[captions] {OUT}/captions.md")


# ---------------------------------------------------------------------------------------------------
def main():
    global OUT
    global IL_KS
    global KS_SET
    sc = load_scripted(args.scripted) if args.scripted else None
    if sc and sc["controller"] != "dynamic":
        print(f"[warn] {args.scripted} is a {sc['controller']} run; the demonstrations use the dynamic controller")
    demos = sorted((load_il(p) for p in args.demos), key=lambda s: s["K"])
    rolls = sorted((load_il(p) for p in args.rollouts), key=lambda s: s["K"])
    ks = [s["K"] for s in rolls]
    if [s["K"] for s in demos] != ks:
        raise SystemExit("the demos and rollouts must have the same two K")
    if any(s["results"] is None for s in rolls):
        raise SystemExit("the results JSON next to a rollout file is missing")
    stem = os.path.splitext(os.path.basename(args.rollouts[0]))[0].split("_K")[0]
    OUT = args.out or os.path.join("logs", "il_eval", "figures", stem)
    os.makedirs(OUT, exist_ok=True)
    tables = {"drives": {k: v.tolist() for k, v in demos[0]["drive"].items()}, "hand_joints": demos[0]["hand"]}

    stats = [success_stats(s["results"], s["K"]) for s in rolls]
    order = [np.argsort(s["episode"]) for s in stats]
    p, n10, n01 = mcnemar(stats[0]["success"][order[0]], stats[1]["success"][order[1]])
    tables["mcnemar"] = {"p": p, "n10": n10, "n01": n01}
    tables["outcome"] = [{k: v for k, v in s.items() if k not in ("success", "episode")} for s in stats]

    IL_KS = tuple(ks)
    fig_outcome(stats, tables["mcnemar"])
    fig_hold_force(demos, rolls, tables)
    fig_hold_share(demos, rolls, tables)
    fig_grip_timeline(demos, rolls, tables)
    fig_penetration_timeline(demos, rolls, tables)
    fig_admittance_law(demos)
    fig_penetration(sc, demos, rolls, tables)
    fig_force_ratio(sc, demos, rolls, tables)
    if args.kswap:
        # the policy on demonstration states: the validation demos (never trained on) when given
        if args.val_demos:
            ks_demos = sorted((load_il(p) for p in args.val_demos), key=lambda s: s["K"])
            if [s["K"] for s in ks_demos] != ks:
                raise SystemExit("the validation demos must have the same two K as the rollouts")
            KS_SET = "validation demonstrations (never trained on)"
        else:
            ks_demos, KS_SET = demos, "training demonstrations"
        summ = os.path.join(args.kswap, "summary.json")
        if os.path.exists(summ):
            sj = json.load(open(summ))
            if args.val_demos and sj.get("split") != "val":
                print(f"[warn] {args.kswap} is not from validation demos (split = {sj.get('split')})")
            inc = sj.get("in_contact", {})
            tables["kswap_joint"] = {f: inc[f"all demos {f}"]["gain"] for f in FINGERS if f"all demos {f}" in inc}
        cmp = compare_with_admittance(args.kswap, {"by_k": {s["K"]: s for s in ks_demos}})
        res = summarise_comparison(cmp) if cmp is not None else None
        fig_kswap(args.kswap, ks_demos, tables, res)
        if cmp is not None:
            tables["comparison"] = res
            n_demos = len(ks_demos[0]["eps"])
            fig_law_capture(res, n_demos)
            fig_command_error(res, n_demos)
            if args.kswap_train:
                cmp_t = compare_with_admittance(args.kswap_train, {"by_k": {s["K"]: s for s in demos}})
                if cmp_t is not None:
                    tables["comparison_train"] = summarise_comparison(cmp_t)
    fig_precontact(rolls, tables)
    d_names = [pick_demo(s) for s in demos]
    fig_single(demos, d_names, "One demonstration per stiffness", "single_demos",
               f"One teleoperated demonstration per stiffness ({d_names[0]} at K = {ks[0]:g}, {d_names[1]} at "
               f"K = {ks[1]:g} N/m; the demonstrations with the cube closest to the spawn centre). Top: cube height. "
               "Per finger: contact force, admittance yield e and command penetration delta_cmd (the components "
               "along the contact force, shown while the finger is in contact). Time from the start of the "
               "demonstration (clutch engaged).", show_yield=True)
    pair = pick_pair(rolls)
    fig_single(rolls, [pair, pair], f"One policy rollout per stiffness (episode {int(pair[3:])})", "single_rollouts",
               f"The policy at the same cube start position with K = {ks[0]:g} and {ks[1]:g} N/m (rollout episode "
               f"{int(pair[3:])}). Top: cube height. Per finger: contact force and command penetration delta_cmd "
               "(component of J (q_cmd - q) along the contact force, while in contact). There is no admittance in "
               "the rollouts: the policy outputs q_cmd directly. Time from the moment the policy takes over.",
               show_yield=False)
    tables["single"] = {"demos": d_names, "rollout_episode": int(pair[3:])}
    fig_drive_relation(sc, demos, rolls, tables)  # validity check (backup)
    write_tables(tables, ks)
    write_captions()


if __name__ == "__main__":
    main()
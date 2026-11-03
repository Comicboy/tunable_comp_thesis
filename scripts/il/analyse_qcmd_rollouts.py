"""Has the policy learned the K-dependent grip? Compares policy rollouts with the demos, per stiffness K.

For policies whose hand action is the finger command q_cmd (the policy performs the fingertip
admittance itself; the env's admittance law is off in the rollouts). In the demos the operator's
intention q_ref was the same at both K and the admittance made the force differ:

    q_cmd = q_ref + J^+ e,   e = f / K  (at steady state)

so the same closing gave more force at K = 500 than at K = 150. In the rollouts no controller does this;
any K-dependence of the force has to come from the policy's own q_cmd. No Isaac Sim needed (h5py,
numpy, matplotlib); the inputs are the demo files (teleop_lift.py --record) and the trajectory files
of rollout_policy.py (one per K, same seed):

    python scripts/il/analyse_qcmd_rollouts.py \\
        --demos logs/il_demos/lift_K150.hdf5 logs/il_demos/lift_K500.hdf5 \\
        --rollouts logs/il_eval/act_qcmd_v1_te0.01_K150_seed0.hdf5 logs/il_eval/act_qcmd_v1_te0.01_K500_seed0.hdf5 \\
        [--out DIR] [--pdf]

Output folder (default logs/il_eval/analysis/<rollout name without _K.._seed..>):

    stiffness_hold.png          the hold (last --hold_s of every successful episode): contact force and
                                command penetration ||q_cmd - q|| of every finger and in total, one dot per
                                episode, median bars; demos and policy side by side at each K
    stiffness_force_vs_pen.png  hold force against penetration, one dot per episode, per finger.
                                Top row, demos only: against the operator's reference penetration
                                ||q_ref - q||; the slope differs with K (the admittance: the same closing
                                gives more force at K = 500). Bottom row, demos and policy: against the
                                command penetration ||q_cmd - q||, which the joint drives turn into force;
                                with no admittance in that path the slope is the same at both K, and K only
                                moves the dots along the line (less command, less force at K = 150). The
                                policy reproduces the admittance if its dots sit where the demos' do.
    stiffness_time.png          aligned on lift-off (cube bottom 5 mm above the table): cube height, thumb
                                force, fingers 2-5 force (sum), thumb and fingers 2-5 command penetration;
                                median and interquartile band per group (successful episodes)
    stiffness_thumb.png         thumb joints 1-4 around the first contact: command q_cmd and measured q, for
                                demos, successful and failed policy episodes, per K (the thumb is the passive
                                stop of this grasp; in the demos its force did not depend on K)
    precontact_paired.png       policy only, same cube position at both K: K150 minus K500 of the arm targets
                                (per joint) and finger commands (norm over each finger's 4 joints), aligned on
                                the first contact in either run; should be ~0 before contact

and printed tables: hold forces per finger with the ratio K500 / K150 and a 95 % bootstrap interval
(the demos' ratio is the target; ~1 = the policy ignores K), and the paired differences before contact.

Hold forces: successful episodes only (decided 2026-11-02: without a proper grasp the force-stiffness
relation is absent). Per episode the median over the hold window; per group the median over the
episodes. For the policy also on the paired subset (positions where both K succeeded), so that a
different selection of positions cannot make the ratio.

Paired check before contact: episode i starts from the same cube position and the same settled state
at both K, and the env does nothing K-dependent while the hand touches nothing (the admittance law is
off; the smoothed forces are exactly 0). So until the first contact in either run the only difference
is the policy's K input, and an admittance has nothing to do there. Printed: the state difference at the
policy's first step (~0 if the physics is deterministic), the difference at the last step before
contact and the largest one before contact, and the difference by time since the policy took over: at
t = 0 both runs are in the same state, so the first action's difference is the direct effect of the K
input; later differences also contain the drift of the two paths (a small difference in the approach
changes the state, and the fingers close fast just before contact, so a slight shift in timing gives
large command differences).
"""

import argparse
import json
import os
import re
import warnings

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

parser = argparse.ArgumentParser(description="Demos vs q_cmd-policy rollouts: K-dependent grip and the paired pre-contact check.")
parser.add_argument("--demos", nargs=2, required=True, help="the two demo files (one per K)")
parser.add_argument("--rollouts", nargs=2, required=True, help="the two rollout trajectory files (.hdf5, one per K, same seed)")
parser.add_argument("--out", default=None, help="folder for plots and the summary JSON")
parser.add_argument("--hold_s", type=float, default=2.5, help="hold window: the last hold_s of each successful episode [s]")
parser.add_argument("--boot", type=int, default=2000, help="bootstrap resamples for the ratio intervals")
parser.add_argument("--pdf", action="store_true", help="also save PDFs")
args = parser.parse_args()

FINGERS = ["thumb", "index", "middle", "ring", "little"]
CONTACT_N = 1e-3   # [N] summed smoothed fingertip force: first contact (exactly 0 before any touch)
LIFTOFF_M = 0.005  # [m] cube bottom above the table: lift-off
MIN_RATIO_N = 0.5  # [N] per-finger ratio only if both medians reach this
MIN_FIT_N = 0.3    # [N] hold force below this: finger not in contact, left out of the slope fit
REACH_M = 0.6      # [m] ~ distance shoulder axis to hand: shoulder_pan [rad] x REACH_M ~ sideways shift [m]
TIME_WINDOW = (-3.0, 5.0)  # [s] around lift-off
PRE_S, POST_S = 3.0, 1.5   # [s] before / after the first contact in the paired plot

# --- style (as plot_scripted_lift.py and the earlier stiffness plots) -----------------------------
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
K_COLOR = {150.0: "#5598e7", 500.0: "#184f95"}  # one blue ramp: softer = lighter
PAIR_COLOR = "#2a78d6"
SOURCE_STYLE = {"demo": {"linestyle": "-", "marker": "o"}, "policy": {"linestyle": "--", "marker": "^"}}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.6,
})


def k_color(k: float) -> str:
    return K_COLOR.get(float(k), INK)


def group_label(source: str, k: float) -> str:
    return f"{'demos' if source == 'demo' else 'policy'}, K = {k:g}"


# ---------------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------------
def load(path: str) -> dict:
    """Episodes of a demo or rollout file with the per-step quantities the analyses use."""
    with h5py.File(path, "r") as f:
        a = dict(f.attrs)
        sl = {name: slice(s, e) for name, s, e in json.loads(a["obs_terms"])}
        n_arm = len(json.loads(a["arm_joints"]))
        hand = json.loads(a["hand_joints"])
        finger_idx = [[i for i, n in enumerate(hand) if n.startswith(f"right_finger{k}_")] for k in range(1, 6)]
        thumb_idx = [hand.index(f"right_finger1_joint{j}") for j in range(1, 5)]
        rollout = a.get("source") == "policy rollout"
        if not rollout and "q_cmd" not in str(a.get("format", "")):
            raise SystemExit(f"{path}: not a demo file with q_cmd as the action")
        half = json.loads(a["success"])["half_size"]
        out = {"path": path, "K": float(a["stiffness"]), "dt": float(a["step_dt"]), "rollout": rollout,
               "source": "policy" if rollout else "demo", "seed": a.get("seed"),
               "arm_joints": json.loads(a["arm_joints"]), "finger_idx": finger_idx, "eps": {}}
        for name in sorted(f["data"].keys()):
            g = f["data"][name]
            obs, act = g["obs"][()], g["actions"][()]
            T = len(obs)
            forces = np.linalg.norm(obs[:, sl["fingertip_forces"]].reshape(T, 5, 3), axis=-1)  # (T, 5) [N]
            q, q_cmd = obs[:, sl["hand_joint_pos"]], act[:, n_arm:]
            per_finger = lambda d: np.stack([np.linalg.norm(d[:, ix], axis=-1) for ix in finger_idx], -1)  # noqa: E731
            h = obs[:, sl["cube_position"]][:, 2] - half
            up = np.nonzero(h > LIFTOFF_M)[0]
            touch = np.nonzero(forces.sum(-1) > CONTACT_N)[0]
            ep = {
                "name": name, "success": bool(g.attrs.get("success", True)),  # demo files hold successes only
                "dt": out["dt"], "forces": forces, "h": h,
                "cmd_pen": per_finger(q_cmd - q),  # (T, 5) command beyond the measured joints [rad]
                "arm_target": act[:, :n_arm], "q_cmd": q_cmd,
                "thumb_q": q[:, thumb_idx], "thumb_cmd": q_cmd[:, thumb_idx],
                "state_no_k": np.delete(obs, np.r_[sl["stiffness"]], axis=1),
                "liftoff": int(up[0]) if up.size else None,
                "contact": int(touch[0]) if touch.size else None,
            }
            if not rollout:  # the operator's reference (demos only)
                ep["ref_pen"] = per_finger(g["q_ref"][()] - q)
            out["eps"][name] = ep
    return out


def by_k(files):
    """The two files sorted by K (low first)."""
    sets = sorted((load(p) for p in files), key=lambda s: s["K"])
    if sets[0]["K"] == sets[1]["K"]:
        raise SystemExit(f"both files have K = {sets[0]['K']:g}")
    return sets


def hold(ep, arr):
    """Median over the hold window (last hold_s) of arr (T, ...)."""
    n = int(round(args.hold_s / ep["dt"]))
    return np.median(arr[-n:], axis=0)


def save(fig, out_dir, name):
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    if args.pdf:
        fig.savefig(os.path.join(out_dir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


def band(eps, fn, key, lag):
    """Median and quartiles over episodes of fn(e) (T, ...) sampled at e[key] + lag; NaN where too few."""
    m = None
    for i, e in enumerate(eps):
        y = fn(e)
        if m is None:
            m = np.full((len(eps), len(lag)) + y.shape[1:], np.nan)
        idx = e[key] + lag
        ok = (idx >= 0) & (idx < len(y))
        m[i, ok] = y[idx[ok]]
    enough = (~np.isnan(m)).sum(0) >= max(3, len(eps) // 4)  # only where enough episodes still run
    with warnings.catch_warnings():  # columns where no episode runs any more: all NaN, masked anyway
        warnings.simplefilter("ignore", RuntimeWarning)
        return tuple(np.where(enough, np.nanpercentile(m, p, 0), np.nan) for p in (50, 25, 75))


# ---------------------------------------------------------------------------------------------------
# 1. hold forces: table with ratios
# ---------------------------------------------------------------------------------------------------
COLS = [*FINGERS, "fingers 2-5", "all five"]


def hold_row(ep):
    """Hold force per finger, fingers 2-5 summed, all five summed [N] (median over the hold window)."""
    f = ep["forces"][-int(round(args.hold_s / ep["dt"])):]
    return np.array([*np.median(f, axis=0), np.median(f[:, 1:].sum(-1)), np.median(f.sum(-1))])


def ratio_ci(lo: np.ndarray, hi: np.ndarray, rng) -> tuple:
    """Ratio of medians hi / lo with a 95 % bootstrap interval (episodes resampled within each group).

    Not computed (nan) when either median is below MIN_RATIO_N: a finger that is mostly not in contact
    in the hold has a median of ~0 N and its ratio means nothing."""
    if min(np.median(lo), np.median(hi)) < MIN_RATIO_N:
        return float("nan"), float("nan"), float("nan")
    r = np.median(hi) / np.median(lo)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.array([np.median(rng.choice(hi, len(hi))) / np.median(rng.choice(lo, len(lo))) for _ in range(args.boot)])
    b[~np.isfinite(b)] = np.inf  # a resample with a 0 N median below: ratio unbounded (counts as large)
    return float(r), *(float(v) for v in np.percentile(b, [2.5, 97.5]))


def hold_table(demos, rolls, summary):
    groups = {}  # (row label, k index) -> (n_eps, 7)
    for src, sets in (("demos", demos), ("policy", rolls)):
        for i, s in enumerate(sets):
            groups[(src, i)] = np.array([hold_row(e) for e in s["eps"].values() if e["success"]])
    both = [n for n in rolls[0]["eps"] if n in rolls[1]["eps"]
            and rolls[0]["eps"][n]["success"] and rolls[1]["eps"][n]["success"]]
    for i in (0, 1):
        groups[("policy, both K succeeded", i)] = np.array([hold_row(rolls[i]["eps"][n]) for n in both])

    k_lo, k_hi = demos[0]["K"], demos[1]["K"]
    rng = np.random.default_rng(0)
    print(f"\n=== 1. Hold forces (median over the last {args.hold_s:g} s of each successful episode) [N]")
    print(f"{'':<36}" + "".join(f"{c:>13}" for c in COLS))
    for src in ("demos", "policy", "policy, both K succeeded"):
        for i, k in ((0, k_lo), (1, k_hi)):
            v = groups[(src, i)]
            print(f"{src + f' K{k:g} (n={len(v)})':<36}" + "".join(f"{np.median(v[:, c]):>13.2f}" for c in range(len(COLS))))
    print(f"\nratio K{k_hi:g} / K{k_lo:g} of the medians [95 % bootstrap interval]  "
          f"(target: the demos; ~1 = the policy ignores K; '-': median below {MIN_RATIO_N} N)")
    print(f"{'':<36}" + "".join(f"{c:>18}" for c in FINGERS))
    summary["hold_forces"] = {"hold_s": args.hold_s, "ratio_hi_over_lo": {}}
    for src in ("demos", "policy", "policy, both K succeeded"):
        cells, rec = [], {}
        for c, name in enumerate(COLS):
            r, lo, hi = ratio_ci(groups[(src, 0)][:, c], groups[(src, 1)][:, c], rng)
            cells.append("-" if np.isnan(r) else f"{r:.2f} [{lo:.2f},{hi:.2f}]")
            rec[name] = [r, lo, hi]
        summary["hold_forces"]["ratio_hi_over_lo"][src] = rec
        print(f"{src:<36}" + "".join(f"{c:>18}" for c in cells[:5]))
        print(f"{'':<36}" + "".join(f"{c:>18}" for c in cells[5:]) + "   (fingers 2-5, all five)")
    summary["hold_forces"]["medians"] = {f"{s} K{(k_lo, k_hi)[i]:g}": dict(zip(COLS, np.median(v, 0).tolist()))
                                         for (s, i), v in groups.items()}


# ---------------------------------------------------------------------------------------------------
# figures: hold, force vs penetration, time, thumb
# ---------------------------------------------------------------------------------------------------
def hold_values(sets):
    """(source, K) -> per-episode hold force (n, 5), command penetration (n, 5), reference penetration (demos)."""
    out = {}
    for s in sets:
        eps = [e for e in s["eps"].values() if e["success"]]
        v = {"force": np.array([hold(e, e["forces"]) for e in eps]).reshape(-1, 5),
             "cmd_pen": np.array([hold(e, e["cmd_pen"]) for e in eps]).reshape(-1, 5)}
        if s["source"] == "demo":
            v["ref_pen"] = np.array([hold(e, e["ref_pen"]) for e in eps]).reshape(-1, 5)
        out[(s["source"], s["K"])] = v
    return dict(sorted(out.items(), key=lambda kv: (kv[0][1], kv[0][0])))


def plot_hold(hv, out_dir):
    cols = FINGERS + ["total"]
    ks = sorted({k for _, k in hv})
    fig, axes = plt.subplots(2, len(cols), figsize=(17, 6.5), sharey="row")
    rng = np.random.default_rng(0)
    for c in range(len(cols)):
        for r, what in enumerate(("force", "cmd_pen")):
            ax = axes[r, c]
            for xi, k in enumerate(ks):
                for si, source in enumerate(("demo", "policy")):
                    if (source, k) not in hv or len(hv[(source, k)][what]) == 0:
                        continue
                    v = hv[(source, k)][what]
                    y = (v.sum(-1) if what == "force" else v.mean(-1)) if c == 5 else v[:, c]
                    x0 = xi + (si - 0.5) * 0.36
                    ax.scatter(x0 + rng.uniform(-0.08, 0.08, len(y)), y, s=12, color=k_color(k),
                               marker=SOURCE_STYLE[source]["marker"], alpha=0.55, linewidths=0)
                    ax.plot([x0 - 0.13, x0 + 0.13], [np.median(y)] * 2, color=INK, linewidth=2.0)
            ax.set_xticks(range(len(ks)))
            ax.set_xticklabels([f"K = {k:g}" for k in ks])
            ax.set_xlim(-0.6, len(ks) - 0.4)
            if r == 0:
                ax.set_title(cols[c] if c < 5 else "total (force: sum, penetration: mean)")
            if c == 0:
                ax.set_ylabel("hold contact force |f| [N]" if what == "force"
                              else "hold command penetration ||q_cmd - q|| [rad]")
    handles = [plt.Line2D([], [], color=MUTED, marker=SOURCE_STYLE[s]["marker"], linestyle="", label=lab)
               for s, lab in (("demo", "demos (left of each pair)"), ("policy", "policy (right)"))]
    handles.append(plt.Line2D([], [], color=INK, linewidth=2.0, label="median"))
    axes[0, 0].legend(handles=handles, loc="upper left")
    fig.suptitle(f"Hold (last {args.hold_s:g} s of every successful episode): one dot per episode", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, out_dir, "stiffness_hold")


def slope_lines(ax, hv, f, x_key, sources, xmax):
    """Least-squares slope through the origin per K over the given sources (fingers in contact only)."""
    for k in sorted({k for _, k in hv}):
        xs = [hv[(s, k)][x_key][:, f] for s in sources if (s, k) in hv and x_key in hv[(s, k)]]
        ys = [hv[(s, k)]["force"][:, f] for s in sources if (s, k) in hv and x_key in hv[(s, k)]]
        if not xs:
            continue
        x, y = np.concatenate(xs), np.concatenate(ys)
        use = y > MIN_FIT_N
        if use.sum() >= 5:
            slope = (x[use] * y[use]).sum() / (x[use] ** 2).sum()
            ax.plot([0, xmax], [0, slope * xmax], color=k_color(k), linestyle=":", linewidth=1.2)
            ax.text(0.98, 0.04 + 0.08 * (k > 200), f"K = {k:g}: {slope:.0f} N/rad (n = {use.sum()})",
                    transform=ax.transAxes, ha="right", fontsize=8, color=k_color(k))


def plot_force_vs_pen(hv, out_dir):
    fig, axes = plt.subplots(2, 5, figsize=(17, 8))
    rows = [("ref_pen", ("demo",), "reference penetration ||q_ref - q|| [rad]", "demos, operator's reference"),
            ("cmd_pen", ("demo", "policy"), "command penetration ||q_cmd - q|| [rad]", "demos and policy, command")]
    for r, (x_key, sources, xlabel, row_title) in enumerate(rows):
        for f in range(5):
            ax = axes[r, f]
            xmax = 0.0
            for (source, k), v in hv.items():
                if source not in sources or x_key not in v or len(v["force"]) == 0:
                    continue
                x, y = v[x_key][:, f], v["force"][:, f]
                ax.scatter(x, y, s=14, color=k_color(k), marker=SOURCE_STYLE[source]["marker"], alpha=0.6,
                           linewidths=0, label=group_label(source, k))
                xmax = max(xmax, float(x.max()))
            slope_lines(ax, hv, f, x_key, sources, xmax)
            ax.set_title(f"{FINGERS[f]} ({row_title})" if f == 0 else FINGERS[f])
            ax.set_xlabel(xlabel)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            if f == 0:
                ax.set_ylabel("hold contact force |f| [N]")
                ax.legend(loc="upper left")
    fig.suptitle("Hold force vs penetration, one dot per episode; dotted: slope through the origin per K "
                 f"(fingers with > {MIN_FIT_N} N)\ntop: the admittance in the demos (slope depends on K); "
                 "bottom: the joint drives (same slope at both K, K moves the dots along the line)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, out_dir, "stiffness_force_vs_pen")


def plot_time(sets, out_dir):
    rows = [
        ("Cube bottom above the table", "cm", lambda e: 100 * e["h"]),
        ("Thumb contact force |f|", "N", lambda e: e["forces"][:, 0]),
        ("Fingers 2-5 contact force (sum)", "N", lambda e: e["forces"][:, 1:].sum(-1)),
        ("Thumb command penetration ||q_cmd - q||", "rad", lambda e: e["cmd_pen"][:, 0]),
        ("Fingers 2-5 command penetration (mean)", "rad", lambda e: e["cmd_pen"][:, 1:].mean(-1)),
    ]
    fig, axes = plt.subplots(len(rows), 1, figsize=(8.5, 11), sharex=True)
    for s in sorted(sets, key=lambda s: (s["K"], s["source"])):
        eps = [e for e in s["eps"].values() if e["success"] and e["liftoff"] is not None]
        if not eps:
            continue
        dt = eps[0]["dt"]
        lag = np.arange(int(round(TIME_WINDOW[0] / dt)), int(round(TIME_WINDOW[1] / dt)) + 1)
        for ax, (_, _, fn) in zip(axes, rows):
            med, q1, q3 = band(eps, fn, "liftoff", lag)
            ax.fill_between(lag * dt, q1, q3, color=k_color(s["K"]), alpha=0.12 if s["source"] == "demo" else 0.08, lw=0)
            ax.plot(lag * dt, med, color=k_color(s["K"]), linestyle=SOURCE_STYLE[s["source"]]["linestyle"],
                    label=f"{group_label(s['source'], s['K'])} (n = {len(eps)})")
    for ax, (title, unit, _) in zip(axes, rows):
        ax.axvline(0.0, color=AXIS, linewidth=1.0)
        ax.set_title(title)
        ax.set_ylabel(unit)
    axes[0].text(0.0, 1.02, "lift-off", transform=axes[0].get_xaxis_transform(), ha="center", fontsize=7.5, color=MUTED)
    axes[-1].set_xlabel("time relative to lift-off [s]")
    axes[0].legend(loc="upper left")
    fig.suptitle("Demos vs policy, aligned on lift-off: median and interquartile band (successful episodes)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, out_dir, "stiffness_time")


def plot_thumb(sets, out_dir):
    """Thumb joint commands and angles around the first contact: demos, successful and failed policy episodes."""
    lines = []  # (label, colour, linestyle, episodes)
    for s in sorted(sets, key=lambda s: (s["K"], s["source"])):
        eps = [e for e in s["eps"].values() if e["contact"] is not None]
        if s["source"] == "demo":
            lines.append((f"demos, K = {s['K']:g}", k_color(s["K"]), "-", eps))
        else:
            lines.append((f"policy, K = {s['K']:g}, successful", k_color(s["K"]), "--", [e for e in eps if e["success"]]))
            lines.append((f"policy, K = {s['K']:g}, failed", k_color(s["K"]), ":", [e for e in eps if not e["success"]]))
    fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True)
    for label, color, ls, eps in lines:
        if len(eps) < 3:
            continue
        dt = eps[0]["dt"]
        lag = np.arange(int(round(-2.0 / dt)), int(round(3.0 / dt)) + 1)
        for c, key in enumerate(("thumb_cmd", "thumb_q")):
            med, q1, q3 = band(eps, lambda e: e[key], "contact", lag)
            for j in range(4):
                axes[j, c].fill_between(lag * dt, q1[:, j], q3[:, j], color=color, alpha=0.07, lw=0)
                axes[j, c].plot(lag * dt, med[:, j], color=color, linestyle=ls, label=f"{label} (n = {len(eps)})")
    for j in range(4):
        for c, what in enumerate(("command q_cmd (action)", "angle q (measured)")):
            axes[j, c].axvline(0.0, color=AXIS, linewidth=1.0)
            axes[j, c].set_title(f"thumb joint{j + 1}: {what}")
            if c == 0:
                axes[j, c].set_ylabel("rad")
    for ax in axes[-1]:
        ax.set_xlabel("time relative to the first contact [s]")
    axes[0, 0].legend(loc="best", fontsize=7)
    fig.suptitle("Thumb joints around the first contact: median and interquartile band", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, out_dir, "stiffness_thumb")


# ---------------------------------------------------------------------------------------------------
# 2. paired check before contact
# ---------------------------------------------------------------------------------------------------
def precontact(rolls, out_dir, summary):
    lo_s, hi_s = rolls
    if lo_s["seed"] != hi_s["seed"]:
        raise SystemExit(f"the rollout files have different seeds ({lo_s['seed']} vs {hi_s['seed']}): not paired")
    dt = lo_s["dt"]
    arm_names = [n.replace("_joint", "") for n in lo_s["arm_joints"]]
    names = arm_names + [f"{f} q_cmd" for f in FINGERS]
    pre, post = int(round(PRE_S / dt)), int(round(POST_S / dt))
    curves, at_start, before, worst, tc_list, full = [], [], [], [], [], []
    for n in sorted(set(lo_s["eps"]) & set(hi_s["eps"])):
        a, b = lo_s["eps"][n], hi_s["eps"][n]
        T = min(len(a["q_cmd"]), len(b["q_cmd"]))
        tc = min(x if x is not None else T for x in (a["contact"], b["contact"]))
        if tc < 2:
            continue
        # per step: arm target differences (6, signed) and finger command differences (5, norm over 4 joints)
        d_q = a["q_cmd"][:T] - b["q_cmd"][:T]
        d = np.concatenate([a["arm_target"][:T] - b["arm_target"][:T],
                            np.stack([np.linalg.norm(d_q[:, ix], axis=-1) for ix in lo_s["finger_idx"]], -1)], -1)
        at_start.append(np.abs(a["state_no_k"][0] - b["state_no_k"][0]).max())
        full.append((d, tc))
        before.append(d[tc - 1])
        worst.append(np.abs(d[:tc]).max(0))
        tc_list.append(tc * dt)
        w = np.full((pre + post, d.shape[1]), np.nan)
        i0, i1 = max(tc - pre, 0), min(tc + post, T)
        w[i0 - (tc - pre):i1 - (tc - pre)] = d[i0:i1]
        curves.append(w)
    before, worst, curves = np.array(before), np.array(worst), np.array(curves)

    print(f"\n=== 2. Paired check before contact: policy K{lo_s['K']:g} minus K{hi_s['K']:g}, same cube position "
          f"({len(before)} pairs)")
    print(f"first contact (either run) after the policy took over: median {np.median(tc_list):.1f} s "
          f"[{np.percentile(tc_list, 25):.1f}, {np.percentile(tc_list, 75):.1f}]")
    print(f"state difference at the policy's first step (K input excluded): max over pairs {max(at_start):.1e} "
          f"(~0: same start, deterministic physics)")
    print(f"\n{'':<16}{'last step before contact: median [IQR]':>42}{'max over pairs':>16}"
          f"{'largest |diff| before contact: median':>40}{'max':>10}   [mrad]")
    for j, nm in enumerate(names):
        v, w = 1000 * before[:, j], 1000 * worst[:, j]
        print(f"{nm:<16}{np.median(v):>+20.1f} [{np.percentile(v, 25):+.1f}, {np.percentile(v, 75):+.1f}]"
              f"{'':>4}{np.abs(v).max():>16.1f}{np.median(w):>40.1f}{w.max():>10.1f}")
    # by time since the policy took over: at 0 s both runs are in the same state, so the difference of the
    # first action is the direct effect of the K input; later ones also contain the drift of the two paths
    times = [t for t in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5) if any(tc > round(t / dt) for _, tc in full)]
    print("\nby time since the policy took over (pairs still without contact at that time): median |diff| [mrad]")
    print(f"{'':<16}" + "".join(f"{f't = {t:g} s':>11}" for t in times))
    by_time = {}
    for j, nm in enumerate(names):
        row = []
        for t in times:
            k = int(round(t / dt))
            row.append(1000 * float(np.median([abs(d[k, j]) for d, tc in full if tc > k])))
        by_time[nm] = row
        print(f"{nm:<16}" + "".join(f"{v:>11.1f}" for v in row))
    print(f"{'(pairs)':<16}" + "".join(f"{sum(tc > round(t / dt) for _, tc in full):>11d}" for t in times))
    pan = before[:, 0]
    print(f"\nshoulder_pan before contact: median {np.median(pan):+.4f} rad ~ {100 * REACH_M * np.median(pan):+.2f} cm "
          f"sideways (+ = left); finger commands are norms over each finger's 4 joints (>= 0)")
    summary["precontact"] = {
        "pairs": len(before), "first_contact_s_median": float(np.median(tc_list)),
        "state_diff_at_start_max": float(max(at_start)),
        "by_time_since_start_median_abs_rad": {"times_s": times, **{k: [v / 1000 for v in r] for k, r in by_time.items()}},
        "last_step_before_contact_median_rad": dict(zip(names, np.median(before, 0).tolist())),
        "last_step_before_contact_abs_max_rad": dict(zip(names, np.abs(before).max(0).tolist())),
        "largest_before_contact_median_rad": dict(zip(names, np.median(worst, 0).tolist())),
        "largest_before_contact_max_rad": dict(zip(names, worst.max(0).tolist())),
    }

    # figure: arm joints down the left column, fingers down the right
    t = np.arange(-pre, post) * dt
    fig, axes = plt.subplots(6, 2, figsize=(12, 13), sharex=True)
    panels = [axes[i, 0] for i in range(6)] + [axes[i, 1] for i in range(5)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        q1, med, q3 = (np.nanpercentile(1000 * curves, p, axis=0) for p in (25, 50, 75))
    for j, (nm, ax) in enumerate(zip(names, panels)):
        ax.fill_between(t, q1[:, j], q3[:, j], color=PAIR_COLOR, alpha=0.15, lw=0)
        ax.plot(t, med[:, j], color=PAIR_COLOR)
        ax.axvline(0.0, color=AXIS, linewidth=1.0)
        ax.axhline(0.0, color=AXIS, linewidth=1.0)
        ax.set_title(f"{nm}: arm target" if j < 6 else f"{nm} (norm over 4 joints)")
        ax.set_ylabel("mrad")
    axes[5, 1].set_visible(False)
    axes[0, 0].text(0.0, 1.02, "first contact", transform=axes[0, 0].get_xaxis_transform(), ha="center",
                    fontsize=7.5, color=MUTED)
    axes[5, 0].set_xlabel("time relative to the first contact in either run [s]")
    axes[4, 1].set_xlabel("time relative to the first contact in either run [s]")
    axes[4, 1].xaxis.set_tick_params(labelbottom=True)
    fig.suptitle(f"Same cube position, policy at K = {lo_s['K']:g} minus K = {hi_s['K']:g}: median and interquartile "
                 f"band over {len(curves)} pairs\n(no force before contact: an admittance has nothing to do there, "
                 "the difference should be ~0)", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, out_dir, "precontact_paired")


def main():
    demos, rolls = by_k(args.demos), by_k(args.rollouts)
    if [s["K"] for s in demos] != [s["K"] for s in rolls]:
        raise SystemExit(f"K of the demos {[s['K'] for s in demos]} and rollouts {[s['K'] for s in rolls]} differ")
    if not (rolls[0]["rollout"] and rolls[1]["rollout"]) or demos[0]["rollout"] or demos[1]["rollout"]:
        raise SystemExit("--demos must be demo files and --rollouts rollout trajectory files")
    stem = re.sub(r"_K[\d.]+_seed\d+.*$", "", os.path.splitext(os.path.basename(args.rollouts[0]))[0])
    out_dir = args.out or os.path.join("logs", "il_eval", "analysis", stem)
    os.makedirs(out_dir, exist_ok=True)
    for s in demos + rolls:
        n_ok = sum(e["success"] for e in s["eps"].values())
        print(f"[load] {s['path']}: K = {s['K']:g}, {len(s['eps'])} episodes, {n_ok} successful")
    summary = {"demos": args.demos, "rollouts": args.rollouts}
    hold_table(demos, rolls, summary)
    hv = hold_values(demos + rolls)
    plot_hold(hv, out_dir)
    plot_force_vs_pen(hv, out_dir)
    plot_time(demos + rolls, out_dir)
    plot_thumb(demos + rolls, out_dir)
    precontact(rolls, out_dir, summary)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    print(f"\n[analyse] summary -> {os.path.join(out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
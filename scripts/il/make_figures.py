"""Figures and tables of the simulation IL stage (paper / presentation), from the files on disk.

No Isaac Sim, no LeRobot (h5py, numpy, matplotlib). Everything is recomputed from the result files, so
the figures can be rebuilt whenever a run changes:

    python scripts/il/make_figures.py \\
        --demos logs/il_demos/lift_K150.hdf5 logs/il_demos/lift_K500.hdf5 \\
        --rollouts logs/il_eval/act_qcmd_v1_te0.01_K150_seed0.hdf5 logs/il_eval/act_qcmd_v1_te0.01_K500_seed0.hdf5 \\
        --kswap logs/il_eval/analysis/act_qcmd_v1_kswap/summary.json \\
        [--pair_episode 94] [--out logs/il_eval/figures/act_qcmd_v1]

--rollouts are the trajectory files of rollout_policy.py; the results JSON next to each (same name,
.json) gives the outcome of every episode. --kswap (optional) is the summary.json of kswap_test.py.

Figures (PNG and PDF):
    fig1_outcomes           outcome of the rollouts per K (success / lifted without the 3 s hold / lifted
                            < 15 cm / never lifted), success rate with its 95 % Wilson interval
    fig2_success_map        cube start positions and outcome per K (q_cmd policy)
    fig3_hold_force         grip force in the hold per K, demos vs policy, with the ratio K500/K150
    fig4_force_liftoff      grip force around lift-off, demos vs policy, per K (median, IQR)
    fig5_precontact         paired rollouts (same cube position): |K150 - K500| of the policy's output
                            around the first contact (an admittance has nothing to do before contact)
    fig6_kswap              offline K-swap: the policy's change of the finger commands vs the admittance
                            law's, before and in contact, and the gain per finger
    fig7_pair_overview      one successful run per K at the same cube position (--pair_episode, default:
                            the position where both K succeeded closest to the spawn centre): cube
                            height, forces, command penetration, as for the scripted lift
    fig8_pair_fingers       the same pair: force and command penetration of every finger
    fig9_pair_joints        the same pair: q_cmd and q of thumb joint3 and the middle finger's joint1
    fig10_demo_joints       one demo per K (closest to the spawn centre): q_ref (operator), q_cmd
                            (admittance, = the action the policy learns) and q, same joints
Tables: tables.md (Markdown), tables.tex (LaTeX, booktabs), and the numbers in tables.json.
"""

import argparse
import json
import math
import os
import warnings

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

parser = argparse.ArgumentParser(description="Paper/presentation figures and tables of the sim IL stage.")
parser.add_argument("--demos", nargs=2, required=True, help="the two demo files (one per K)")
parser.add_argument("--rollouts", nargs=2, required=True, help="rollout trajectory files (.hdf5, one per K, same seed)")
parser.add_argument("--kswap", default=None, help="summary.json of kswap_test.py")
parser.add_argument("--pair_episode", type=int, default=None, help="rollout episode for the single-run figures")
parser.add_argument("--hold_s", type=float, default=2.5, help="hold window: the last hold_s of a successful episode [s]")
parser.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
parser.add_argument("--out", default=None, help="output folder (default logs/il_eval/figures/<rollout name>)")
args = parser.parse_args()

FINGERS = ["thumb", "index", "middle", "ring", "little"]
CENTRE = (0.60, 0.11)  # [m] centre of the cube spawn square (base frame)
TOUCH_N = 1e-3        # [N] summed smoothed fingertip force: first contact
LIFTOFF_M = 0.005     # [m] cube bottom above the table: lift-off
LIFTED_M = 0.03       # [m] max lift below this: "never lifted"
REACH_M = 0.6         # [m] shoulder_pan [rad] x REACH_M ~ sideways shift of the hand

# --- style: as the earlier plots, larger type for slides and papers --------------------------------
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#ffffff"
K_COLOR = {150.0: "#5598e7", 500.0: "#184f95"}  # one blue ramp: softer = lighter
OUTCOME = {  # outcome -> (label, colour, marker)
    "success": ("success", None, "o"),
    "never": ("never lifted", "#e34948", "x"),
    "partial": ("lifted < 15 cm", "#eda100", "s"),
    "nohold": ("lifted, 3 s hold not kept", "#eb6834", "^"),
}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 10.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "legend.frameon": False, "legend.fontsize": 9.5, "lines.linewidth": 1.8,
})


def kc(k):
    return K_COLOR.get(float(k), INK)


def k_lab(k):
    return f"K = {k:g} N/m"


def save(fig, name):
    for ext, kw in (("png", {"dpi": 200}), ("pdf", {})):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"[fig] {os.path.join(OUT, name)}.png/.pdf")


def suptitle(fig, text, sub=None):
    """Bold title and a lighter, wrapped subtitle above the axes (placed in points above the top edge)."""
    import textwrap

    from matplotlib.text import Annotation

    width = int(fig.get_figwidth() * 13)  # characters per line at the subtitle's size
    sub_lines = textwrap.fill(sub, width) if sub else ""
    n = sub_lines.count("\n") + 1 if sub else 0
    if sub:
        fig.add_artist(Annotation(sub_lines, xy=(0.01, 1.0), xycoords="figure fraction", xytext=(0, 12),
                                  textcoords="offset points", ha="left", va="bottom", fontsize=10, color=INK_2))
    fig.add_artist(Annotation(text, xy=(0.01, 1.0), xycoords="figure fraction", xytext=(0, 18 + 14 * n),
                              textcoords="offset points", ha="left", va="bottom", fontsize=13, fontweight="bold",
                              color=INK))


def quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(*a, **k)


# ---------------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(c - h, 0.0), min(c + h, 1.0)


def mcnemar(a, b):
    """Exact two-sided McNemar p for paired booleans."""
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n = n10 + n01
    if n == 0:
        return 1.0, n10, n01
    k = min(n10, n01)
    p = 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n
    return min(p, 1.0), n10, n01


def ratio_ci(lo, hi, rng):
    r = np.median(hi) / np.median(lo)
    b = np.array([np.median(rng.choice(hi, len(hi))) / np.median(rng.choice(lo, len(lo))) for _ in range(args.boot)])
    b[~np.isfinite(b)] = np.inf
    return float(r), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ---------------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------------
def load(path):
    with h5py.File(path, "r") as f:
        a = dict(f.attrs)
        sl = {n: slice(s, e) for n, s, e in json.loads(a["obs_terms"])}
        hand = json.loads(a["hand_joints"])
        n_arm = len(json.loads(a["arm_joints"]))
        half = json.loads(a["success"])["half_size"]
        out = {"path": path, "K": float(a["stiffness"]), "dt": float(a["step_dt"]),
               "rollout": a.get("source") == "policy rollout", "seed": a.get("seed"), "hand": hand,
               "arm": [n.replace("_joint", "") for n in json.loads(a["arm_joints"])],
               "fidx": [[i for i, n in enumerate(hand) if n.startswith(f"right_finger{k}_")] for k in range(1, 6)],
               "eps": {}}
        for name in sorted(f["data"].keys()):
            g = f["data"][name]
            obs, act = g["obs"][()], g["actions"][()]
            T = len(obs)
            q = obs[:, sl["hand_joint_pos"]]
            q_cmd = act[:, n_arm:]
            forces = np.linalg.norm(obs[:, sl["fingertip_forces"]].reshape(T, 5, 3), axis=-1)
            h = obs[:, sl["cube_position"]][:, 2] - half
            touch = np.nonzero(forces.sum(-1) > TOUCH_N)[0]
            up = np.nonzero(h > LIFTOFF_M)[0]
            ep = {"name": name, "success": bool(g.attrs.get("success", True)), "T": T, "obs": obs,
                  "q": q, "q_cmd": q_cmd, "arm_target": act[:, :n_arm], "forces": forces, "h": h,
                  "cube0": obs[0, sl["cube_position"]][:2].astype(float),
                  "cmd_pen": np.stack([np.linalg.norm(q_cmd[:, ix] - q[:, ix], axis=-1) for ix in out["fidx"]], -1),
                  "first": int(touch[0]) if touch.size else T, "liftoff": int(up[0]) if up.size else None,
                  "state_no_k": np.delete(obs, np.r_[sl["stiffness"]], axis=1)}
            if "q_ref" in g:
                ep["q_ref"] = g["q_ref"][()]
            out["eps"][name] = ep
    js = os.path.splitext(path)[0] + ".json"
    out["results"] = json.load(open(js)) if out["rollout"] and os.path.exists(js) else None
    return out


def by_k(paths):
    return sorted((load(p) for p in paths), key=lambda s: s["K"])


def outcome_class(r):
    if r["success"]:
        return "success"
    lift = r.get("max_lift_m", 0.0)
    if lift < LIFTED_M:
        return "never"
    return "nohold" if lift >= 0.145 else "partial"


def success_stats(results):
    eps = results["episodes"]
    n = len(eps)
    k = sum(e["success"] for e in eps)
    cls = [outcome_class(e) for e in eps]
    t = [e["duration_s"] for e in eps if e["success"]]
    lo, hi = wilson(k, n)
    return {"n": n, "k": k, "rate": k / n, "lo": lo, "hi": hi, "never": cls.count("never"),
            "partial": cls.count("partial"), "nohold": cls.count("nohold"),
            "t_mean": float(np.mean(t)) if t else float("nan"), "success": np.array([e["success"] for e in eps]),
            "episode": [e["episode"] for e in eps]}


# ---------------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------------
def fig_outcomes(pol, ks, mc):
    """Per K one horizontal bar split into the outcomes; success rate with its 95 % Wilson interval."""
    parts = [("success", None), ("nohold", OUTCOME["nohold"][1]), ("partial", OUTCOME["partial"][1]),
             ("never", OUTCOME["never"][1])]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for row, (s, k) in enumerate(zip(pol, ks)):
        left = 0.0
        for key, col in parts:
            n = s["k"] if key == "success" else s[key]
            w = 100 * n / s["n"]
            if w <= 0:
                continue
            ax.barh(row, w, left=left, height=0.6, color=kc(k) if key == "success" else col, edgecolor=SURFACE,
                    linewidth=2, label=OUTCOME[key][0] if row == 0 else None)
            if w >= 6:
                ax.text(left + w / 2, row, f"{n}", ha="center", va="center", fontsize=10,
                        color="#ffffff" if key in ("success", "never", "nohold") else INK)
            left += w
        ax.text(101, row, f"{100 * s['rate']:.0f} % [{100 * s['lo']:.0f}, {100 * s['hi']:.0f}]", va="center",
                fontsize=10.5, color=INK)
    ax.set_yticks(range(len(ks)), [k_lab(k) for k in ks])
    ax.invert_yaxis()
    ax.set_xlim(0, 125)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("share of the rollouts [%]")
    ax.grid(axis="y", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    suptitle(fig, "Task outcome per stiffness",
             f"{pol[0]['n']} rollouts per K at the same cube positions; right: success rate [95 % Wilson interval]; "
             f"K150 vs K500 paired: exact McNemar p = {mc['p']:.2f}")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "fig1_outcomes")


def fig_success_map(rolls):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True, sharey=True)
    for ax, s in zip(axes, rolls):
        eps = s["results"]["episodes"]
        for key, (label, col, mk) in OUTCOME.items():
            pts = np.array([e["cube_start"][:2] for e in eps if outcome_class(e) == key]).reshape(-1, 2)
            if not len(pts):
                continue
            c = kc(s["K"]) if key == "success" else col
            ax.scatter(pts[:, 0], pts[:, 1], s=42 if mk != "x" else 50, marker=mk, color=c,
                       linewidths=1.6 if mk == "x" else 0.0, alpha=0.9, label=f"{label} ({len(pts)})", zorder=3)
        n_ok = sum(e["success"] for e in eps)
        ax.set_title(f"{k_lab(s['K'])}: {n_ok}/{len(eps)} successful")
        ax.set_xlabel("cube start x [m] (away from the robot)")
        ax.set_aspect("equal")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9)
    axes[0].set_ylabel("cube start y [m] (+ = left)")
    suptitle(fig, "Where the policy succeeds", "cube start positions of the rollouts (identical at both K)")
    fig.tight_layout()
    save(fig, "fig2_success_map")


def hold_rows(s, names=None):
    n_h = lambda e: int(round(args.hold_s / s["dt"]))  # noqa: E731
    eps = [s["eps"][n] for n in names] if names is not None else [e for e in s["eps"].values() if e["success"]]
    return np.array([[np.median(e["forces"][-n_h(e):, 0]), np.median(e["forces"][-n_h(e):, 1:].sum(-1))]
                     for e in eps]).reshape(-1, 2)  # (n, [thumb, fingers 2-5])


def fig_hold(demos, rolls, tables):
    both = [n for n in rolls[0]["eps"] if n in rolls[1]["eps"] and rolls[0]["eps"][n]["success"]
            and rolls[1]["eps"][n]["success"]]
    groups = {("demos", i): hold_rows(demos[i]) for i in (0, 1)}
    groups.update({("policy", i): hold_rows(rolls[i]) for i in (0, 1)})
    groups.update({("policy, same positions", i): hold_rows(rolls[i], both) for i in (0, 1)})
    rng = np.random.default_rng(0)
    ks = [demos[0]["K"], demos[1]["K"]]
    rec = {}
    for src in ("demos", "policy", "policy, same positions"):
        for c, nm in enumerate(("thumb", "fingers 2-5")):
            rec[(src, nm)] = {"med": [float(np.median(groups[(src, i)][:, c])) for i in (0, 1)],
                              "iqr": [[float(v) for v in np.percentile(groups[(src, i)][:, c], [25, 75])] for i in (0, 1)],
                              "n": [len(groups[(src, i)]) for i in (0, 1)],
                              "ratio": ratio_ci(groups[(src, 0)][:, c], groups[(src, 1)][:, c], rng)}
    tables["hold"] = {f"{s} | {n}": v for (s, n), v in rec.items()}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    jit = np.random.default_rng(1)
    for ax, (c, nm) in zip(axes, ((1, "fingers 2-5"), (0, "thumb"))):
        for xi, (src, i) in enumerate([("demos", 0), ("demos", 1), ("policy", 0), ("policy", 1)]):
            v = groups[(src, i)][:, c]
            x = xi + (0.4 if src == "policy" else 0.0)
            filled = src == "policy"
            ax.scatter(x + jit.uniform(-0.12, 0.12, len(v)), v, s=20, marker="o" if not filled else "^",
                       facecolors=kc(ks[i]) if filled else "none", edgecolors=kc(ks[i]), linewidths=1.1,
                       alpha=0.75, zorder=3)
            ax.plot([x - 0.2, x + 0.2], [np.median(v)] * 2, color=INK, linewidth=2.4, zorder=4)
        ax.set_xticks([0, 1, 2.4, 3.4], [f"demos\n{k_lab(ks[0])}", f"demos\n{k_lab(ks[1])}",
                                         f"policy\n{k_lab(ks[0])}", f"policy\n{k_lab(ks[1])}"], fontsize=9)
        rd, rp = rec[("demos", nm)]["ratio"], rec[("policy", nm)]["ratio"]
        ax.set_title(f"{nm}" + (" (summed)" if c == 1 else ""))
        ax.text(0.02, 0.98, f"ratio K{ks[1]:g} / K{ks[0]:g}\ndemos   {rd[0]:.2f}  [{rd[1]:.2f}, {rd[2]:.2f}]\n"
                            f"policy  {rp[0]:.2f}  [{rp[1]:.2f}, {rp[2]:.2f}]", transform=ax.transAxes, va="top",
                fontsize=9.5, color=INK, family="monospace")
        top = max(float(np.max(groups[(sg, i)][:, c])) for sg in ("demos", "policy") for i in (0, 1))
        ax.set_ylim(0, top * 1.4)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("contact force in the hold [N]")
    hl = [plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=MUTED, label="demonstration"),
          plt.Line2D([], [], marker="^", linestyle="", color=MUTED, label="policy rollout"),
          plt.Line2D([], [], color=INK, linewidth=2.4, label="median")]
    fig.legend(handles=hl, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    suptitle(fig, "Grip force in the hold: the policy makes the stiffness-dependent force itself",
             f"one marker per successful episode (median over its last {args.hold_s:g} s); [95 % bootstrap interval]")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig3_hold_force")


def band(rows):
    m = np.array(rows)
    enough = (~np.isnan(m)).sum(0) >= max(3, len(rows) // 4)
    return tuple(np.where(enough, quiet(np.nanpercentile, m, p, 0), np.nan) for p in (50, 25, 75))


def aligned(eps, fn, key, lag):
    rows = []
    for e in eps:
        y = fn(e)
        idx = e[key] + lag
        ok = (idx >= 0) & (idx < len(y))
        w = np.full(len(lag), np.nan)
        w[ok] = y[idx[ok]]
        rows.append(w)
    return rows


def fig_liftoff(demos, rolls):
    dt = demos[0]["dt"]
    lag = np.arange(int(round(-1.5 / dt)), int(round(4.0 / dt)) + 1)
    t = lag * dt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharex=True)
    for src, sets, ls in (("demos", demos, "--"), ("policy", rolls, "-")):
        for s in sets:
            eps = [e for e in s["eps"].values() if e["success"] and e["liftoff"] is not None]
            for ax, fn in zip(axes, (lambda e: e["forces"][:, 1:].sum(-1), lambda e: e["forces"][:, 0])):
                med, q1, q3 = band(aligned(eps, fn, "liftoff", lag))
                if src == "policy":
                    ax.fill_between(t, q1, q3, color=kc(s["K"]), alpha=0.12, lw=0)
                ax.plot(t, med, ls, color=kc(s["K"]), label=f"{src}, {k_lab(s['K'])} (n = {len(eps)})")
    for ax, title in zip(axes, ("fingers 2-5 (summed)", "thumb")):
        ax.axvline(0, color=AXIS, linewidth=1.0)
        ax.set_title(f"{title} contact force")
        ax.set_xlabel("time from lift-off [s]")
    axes[0].set_ylabel("force [N]")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    suptitle(fig, "Grip force around lift-off", "median of the successful episodes; band: interquartile range of the policy")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig4_force_liftoff")


def paired_diffs(rolls):
    lo, hi = rolls
    out = []
    for n in sorted(set(lo["eps"]) & set(hi["eps"])):
        a, b = lo["eps"][n], hi["eps"][n]
        T = min(a["T"], b["T"])
        tc = min(a["first"], b["first"], T)
        if tc < 2:
            continue
        d_q = a["q_cmd"][:T] - b["q_cmd"][:T]
        fing = np.stack([np.linalg.norm(d_q[:, ix], axis=-1) for ix in lo["fidx"]], -1)
        out.append({"arm": np.abs(a["arm_target"][:T] - b["arm_target"][:T]), "fing": fing, "tc": tc,
                    "start": float(np.abs(a["state_no_k"][0] - b["state_no_k"][0]).max())})
    return out


def fig_precontact(rolls, tables):
    pairs = paired_diffs(rolls)
    dt = rolls[0]["dt"]
    lag = np.arange(int(round(-3.0 / dt)), int(round(1.5 / dt)) + 1)
    t = lag * dt
    lines = [("arm targets (largest joint)", lambda p: p["arm"].max(-1), INK_2),
             ("thumb command", lambda p: p["fing"][:, 0], "#eb6834"),
             ("fingers 2-5 command (mean)", lambda p: p["fing"][:, 1:].mean(-1), "#2a78d6")]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for label, fn, col in lines:
        rows = []
        for p in pairs:
            y = 1000 * fn(p)
            idx = p["tc"] + lag
            ok = (idx >= 0) & (idx < len(y))
            w = np.full(len(lag), np.nan)
            w[ok] = y[idx[ok]]
            rows.append(w)
        med, q1, q3 = band(rows)
        ax.fill_between(t, q1, q3, color=col, alpha=0.12, lw=0)
        ax.plot(t, med, color=col, label=label)
    ax.axvline(0, color=AXIS, linewidth=1.0)
    ax.text(0, 1.01, "first contact", transform=ax.get_xaxis_transform(), ha="center", fontsize=9, color=MUTED)
    ax.set_xlabel("time from the first contact in either run [s]")
    ax.set_ylabel("|K150 − K500| [mrad]")
    ax.legend(loc="upper left")
    suptitle(fig, "Same cube position, K = 150 vs K = 500: the policy's output differs before contact",
             f"{len(pairs)} paired rollouts; no force before contact, so an admittance would give 0 there; "
             "median and IQR (finger commands: norm over 4 joints)")
    fig.tight_layout()
    save(fig, "fig5_precontact")
    # numbers: first step (same state: direct effect of K) and last step before contact
    first = np.array([[p["arm"][0].max(), p["fing"][0, 0], p["fing"][0, 1:].mean()] for p in pairs])
    last = np.array([[p["arm"][p["tc"] - 1].max(), p["fing"][p["tc"] - 1, 0], p["fing"][p["tc"] - 1, 1:].mean()]
                     for p in pairs])
    tables["precontact_rollouts"] = {
        "pairs": len(pairs), "start_state_diff_max": max(p["start"] for p in pairs),
        "first_contact_s_median": float(np.median([p["tc"] * dt for p in pairs])),
        "first_step_mrad_median": dict(zip(["arm max joint", "thumb", "fingers 2-5 mean"], (1000 * np.median(first, 0)).tolist())),
        "before_contact_mrad_median": dict(zip(["arm max joint", "thumb", "fingers 2-5 mean"], (1000 * np.median(last, 0)).tolist())),
    }


def fig_kswap(ks_sum, tables):
    pre = ks_sum["before_contact_rad_median_p90"]["all demos"]
    inc = ks_sum["in_contact"]
    names = [f for f in FINGERS if f"all demos {f}" in inc]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    x = np.arange(len(names))
    w = 0.26
    v_pre = [1000 * pre[f][0] for f in names]
    v_in = [1000 * inc[f"all demos {f}"]["median_abs_pred"] for f in names]
    v_law = [1000 * inc[f"all demos {f}"]["median_abs_star"] for f in names]
    ax.bar(x - w, v_pre, w * 0.92, color=SURFACE, edgecolor="#eb6834", hatch="//", linewidth=1.6,
           label="policy, before contact (law: 0)")
    ax.bar(x, v_in, w * 0.92, color="#2a78d6", label="policy, in contact")
    ax.bar(x + w, v_law, w * 0.92, color=SURFACE, edgecolor=INK_2, linewidth=1.6, label="admittance law, in contact")
    ax.set_xticks(x, names)
    ax.set_ylabel("median |a(K150) − a(K500)| [mrad]")
    ax.set_title("Change of the finger command when only K is swapped")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    show = [f for f in names if f in ("index", "middle", "ring")]
    for i, f in enumerate(show):
        g, lo, hi = inc[f"all demos {f}"]["gain"]
        ax.errorbar(i, g, yerr=[[g - lo], [hi - g]], fmt="o", color="#2a78d6", capsize=5, markersize=8)
        ax.text(i + 0.12, g, f"{g:.2f}", va="center", fontsize=10, color=INK)
    ax.axhline(1.0, color=INK_2, linestyle="--", linewidth=1.2)
    ax.axhline(0.0, color=AXIS, linewidth=1.0)
    ax.text(len(show) - 0.5, 1.0, "admittance law", ha="right", va="bottom", fontsize=9, color=INK_2)
    ax.text(len(show) - 0.5, 0.0, "ignores K", ha="right", va="bottom", fontsize=9, color=MUTED)
    ax.set_xticks(range(len(show)), show)
    ax.set_xlim(-0.5, len(show) - 0.4)
    ax.set_ylim(-0.1, 1.15)
    ax.set_ylabel("gain g")
    ax.set_title("Gain vs the admittance law, in contact")
    ax.grid(axis="x", visible=False)
    suptitle(fig, "Offline K-swap on the demonstration states",
             "every demo state predicted with log10 K = log10 150 and log10 500; thumb: law ≈ 0 (offset clamped "
             "at both K), little: hardly in contact; error bars: 95 % bootstrap over demos")
    fig.tight_layout()
    save(fig, "fig6_kswap")
    tables["kswap"] = {"before_contact_mrad": {f: 1000 * pre[f][0] for f in FINGERS},
                       "before_contact_arm_max_joint_mrad": 1000 * pre["arm_max_joint"][0],
                       "in_contact": {f: inc[f"all demos {f}"] for f in names},
                       "hold": {f: ks_sum["hold"][f"all demos {f}"] for f in names if f"all demos {f}" in ks_sum["hold"]}}


def pick_pair(rolls):
    lo, hi = rolls
    if args.pair_episode is not None:
        name = f"ep_{args.pair_episode:03d}"
        if name not in lo["eps"] or name not in hi["eps"]:
            raise SystemExit(f"episode {args.pair_episode} is not in both rollout files")
        return name
    both = [n for n in lo["eps"] if n in hi["eps"] and lo["eps"][n]["success"] and hi["eps"][n]["success"]]
    return min(both, key=lambda n: math.hypot(*(lo["eps"][n]["cube0"] - np.array(CENTRE))))


def fig_pair(rolls, name):
    lo, hi = rolls
    eps = [(lo["K"], lo["eps"][name]), (hi["K"], hi["eps"][name])]
    dt = lo["dt"]
    rows = [("Cube bottom above the table", "cm", lambda e: 100 * e["h"]),
            ("Thumb contact force", "N", lambda e: e["forces"][:, 0]),
            ("Fingers 2-5 contact force (sum)", "N", lambda e: e["forces"][:, 1:].sum(-1)),
            ("Fingers 2-5 command penetration ‖q_cmd − q‖ (mean)", "rad", lambda e: e["cmd_pen"][:, 1:].mean(-1))]
    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 9.5), sharex=True)
    for k, e in eps:
        t = np.arange(e["T"]) * dt
        for ax, (_, _, fn) in zip(axes, rows):
            ax.plot(t, fn(e), color=kc(k), label=k_lab(k))
        for ax in axes:
            ax.axvline(e["first"] * dt, color=kc(k), linestyle=":", linewidth=1.0)
    for ax, (title, unit, _) in zip(axes, rows):
        ax.set_title(title)
        ax.set_ylabel(unit)
    axes[-1].set_xlabel("time since the policy took over [s]")
    axes[0].legend(loc="upper left")
    c0 = eps[0][1]["cube0"]
    suptitle(fig, "One successful policy run per stiffness, same cube position",
             f"rollout episode {int(name[3:])}, cube at ({c0[0]:.3f}, {c0[1]:.3f}) m; dotted: first contact")
    fig.tight_layout()
    save(fig, "fig7_pair_overview")

    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True, sharey="row")
    for f in range(5):
        for k, e in eps:
            t = np.arange(e["T"]) * dt
            axes[0, f].plot(t, e["forces"][:, f], color=kc(k), label=k_lab(k))
            axes[1, f].plot(t, e["cmd_pen"][:, f], color=kc(k), label=k_lab(k))
        axes[0, f].set_title(f"{FINGERS[f]}: contact force")
        axes[1, f].set_title(f"{FINGERS[f]}: ‖q_cmd − q‖")
        axes[1, f].set_xlabel("time [s]")
    axes[0, 0].set_ylabel("N")
    axes[1, 0].set_ylabel("rad")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    suptitle(fig, "Same pair of runs, every finger", f"rollout episode {int(name[3:])}; the command penetration "
                                                     "is what the joint drives turn into force")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig8_pair_fingers")

    hand = lo["hand"]
    joints = [("thumb joint3", hand.index("right_finger1_joint3")), ("middle finger joint1", hand.index("right_finger3_joint1"))]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.2), sharex=True)
    for r, (k, e) in enumerate(eps):
        t = np.arange(e["T"]) * dt
        for c, (lab, j) in enumerate(joints):
            ax = axes[r, c]
            ax.plot(t, e["q_cmd"][:, j], "--", color=kc(k), label="q_cmd (policy's action)")
            ax.plot(t, e["q"][:, j], color=kc(k), label="q (measured)")
            ax.set_title(f"{lab}, {k_lab(k)}")
            if c == 0:
                ax.set_ylabel("rad")
            ax.legend(loc="lower right", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("time since the policy took over [s]")
    suptitle(fig, "Joint command and position, policy", f"rollout episode {int(name[3:])}")
    fig.tight_layout()
    save(fig, "fig9_pair_joints")


def fig_demo_joints(demos):
    hand = demos[0]["hand"]
    joints = [("thumb joint3", hand.index("right_finger1_joint3")), ("middle finger joint1", hand.index("right_finger3_joint1"))]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.2), sharex=False)
    picked = []
    for r, s in enumerate(demos):
        e = min(s["eps"].values(), key=lambda e: math.hypot(*(e["cube0"] - np.array(CENTRE))))
        picked.append(e["name"])
        t = np.arange(e["T"]) * s["dt"]
        for c, (lab, j) in enumerate(joints):
            ax = axes[r, c]
            if "q_ref" in e:
                ax.plot(t, e["q_ref"][:, j], ":", color=MUTED, linewidth=2.0, label="q_ref (operator)")
            ax.plot(t, e["q_cmd"][:, j], "--", color=kc(s["K"]), label="q_cmd = q_ref + J⁺e (action)")
            ax.plot(t, e["q"][:, j], color=kc(s["K"]), label="q (measured)")
            ax.axvline(e["first"] * s["dt"], color=AXIS, linewidth=1.0)
            ax.set_title(f"{lab}, {k_lab(s['K'])} ({e['name']})")
            if c == 0:
                ax.set_ylabel("rad")
            ax.legend(loc="lower right", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("time since the clutch [s]")
    suptitle(fig, "Demonstrations: the admittance turns the operator's reference into the command the policy learns",
             "one demo per K (cube closest to the spawn centre); vertical line: first contact")
    fig.tight_layout()
    save(fig, "fig10_demo_joints")
    return picked


# ---------------------------------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------------------------------
def write_tables(t):
    md, tex = [], []
    # T1 success
    md += [f"### Table 1: task outcome ({t['success'][0]['n']} rollouts per K, same cube positions)", "",
           "| K [N/m] | success | 95 % CI | lifted, no 3 s hold | lifted < 15 cm | never lifted | mean time to success [s] |",
           "|---|---|---|---|---|---|---|"]
    tex += [r"\begin{table}[t]\centering\small", r"\caption{Task outcome per stiffness (" + str(t["success"][0]["n"]) + r" rollouts per $K$, same cube positions).}",
            r"\begin{tabular}{lcccccc}\toprule",
            r"$K$ [N/m] & success & 95\,\% CI & no 3\,s hold & lifted $<15$\,cm & never lifted & $\bar t_{\mathrm{succ}}$ [s]\\\midrule"]
    for r in t["success"]:
        md.append(f"| {r['K']:g} | {r['k']}/{r['n']} ({100 * r['rate']:.0f} %) | {100 * r['lo']:.0f}–{100 * r['hi']:.0f} % "
                  f"| {r['nohold']} | {r['partial']} | {r['never']} | {r['t_mean']:.1f} |")
        tex.append(f"{r['K']:g} & {r['k']}/{r['n']} & {100 * r['lo']:.0f}--{100 * r['hi']:.0f}\\,\\% & "
                   f"{r['nohold']} & {r['partial']} & {r['never']} & {r['t_mean']:.1f}\\\\")
    m = t["mcnemar"]
    md.append(f"| paired K150 vs K500 | only K150: {m['n10']}, only K500: {m['n01']} | exact McNemar p = {m['p']:.2g} | | | | |")
    tex.append(f"\\multicolumn{{7}}{{l}}{{paired: only $K_{{150}}$ {m['n10']}, only $K_{{500}}$ {m['n01']}, "
               f"exact McNemar $p={m['p']:.2g}$}}\\\\")
    tex += [r"\bottomrule\end{tabular}\end{table}", ""]
    md.append("")
    # T2 hold forces
    md += ["### Table 2: grip force in the hold (median [IQR] over successful episodes, N)", "",
           "| group | finger(s) | K = 150 | K = 500 | ratio K500/K150 [95 % CI] |", "|---|---|---|---|---|"]
    tex += [r"\begin{table}[t]\centering\small", r"\caption{Grip force in the hold, median [IQR] over successful episodes.}",
            r"\begin{tabular}{llccc}\toprule", r"group & finger(s) & $K=150$ [N] & $K=500$ [N] & ratio $K_{500}/K_{150}$ [95\,\% CI]\\\midrule"]
    for key, v in t["hold"].items():
        src, nm = key.split(" | ")
        cells = [f"{v['med'][i]:.1f} [{v['iqr'][i][0]:.1f}, {v['iqr'][i][1]:.1f}] (n={v['n'][i]})" for i in (0, 1)]
        r = v["ratio"]
        md.append(f"| {src} | {nm} | {cells[0]} | {cells[1]} | {r[0]:.2f} [{r[1]:.2f}, {r[2]:.2f}] |")
        tex.append(f"{src} & {nm} & {cells[0]} & {cells[1]} & {r[0]:.2f} [{r[1]:.2f}, {r[2]:.2f}]\\\\")
    tex += [r"\bottomrule\end{tabular}\end{table}", ""]
    md.append("")
    # T3 pre-contact
    p = t["precontact_rollouts"]
    md += ["### Table 3: K dependence before contact (admittance law: 0), median, mrad", "",
           "| measurement | arm (largest joint) | thumb command | fingers 2-5 command |", "|---|---|---|---|"]
    tex += [r"\begin{table}[t]\centering\small", r"\caption{Dependence of the policy's output on $K$ before contact (median, mrad); the admittance law gives 0.}",
            r"\begin{tabular}{lccc}\toprule", r"measurement & arm (largest joint) & thumb & fingers 2--5\\\midrule"]
    for lab, d in ((f"rollouts, first step (same state; {p['pairs']} pairs)", p["first_step_mrad_median"]),
                   ("rollouts, last step before contact", p["before_contact_mrad_median"])):
        vals = [d["arm max joint"], d["thumb"], d["fingers 2-5 mean"]]
        md.append(f"| {lab} | " + " | ".join(f"{v:.0f}" for v in vals) + " |")
        tex.append(f"{lab} & " + " & ".join(f"{v:.0f}" for v in vals) + r"\\")
    if "kswap" in t:
        k = t["kswap"]
        fm = np.mean([k["before_contact_mrad"][f] for f in FINGERS[1:]])
        md.append(f"| K-swap on demo states, before contact | {k['before_contact_arm_max_joint_mrad']:.0f} | "
                  f"{k['before_contact_mrad']['thumb']:.0f} | {fm:.0f} |")
        tex.append(f"K-swap on demo states, before contact & {k['before_contact_arm_max_joint_mrad']:.0f} & "
                   f"{k['before_contact_mrad']['thumb']:.0f} & {fm:.0f}\\\\")
    tex += [r"\bottomrule\end{tabular}\end{table}", ""]
    md.append(f"(start-state difference of the paired rollouts: {p['start_state_diff_max']:.1e}; median first contact "
              f"{p['first_contact_s_median']:.1f} s after the policy takes over)")
    md.append("")
    # T4 K-swap in contact
    if "kswap" in t:
        md += ["### Table 4: K-swap in contact (finger force > 0.5 N), all demos", "",
               "| finger | law ‖Δ*‖ [mrad] | policy ‖Δ‖ [mrad] | gain g [95 % CI] | g, hold only | cos(Δ, Δ*) |", "|---|---|---|---|---|---|"]
        tex += [r"\begin{table}[t]\centering\small", r"\caption{Offline $K$-swap in contact: change of the predicted finger command vs the admittance law.}",
                r"\begin{tabular}{lccccc}\toprule", r"finger & law $\|\Delta^*\|$ [mrad] & policy $\|\Delta\|$ [mrad] & gain $g$ [95\,\% CI] & $g$, hold & $\cos$\\\midrule"]
        for f, v in t["kswap"]["in_contact"].items():
            h = t["kswap"]["hold"].get(f, {}).get("gain", [float("nan")] * 3)
            note = " (law ≈ 0: offset clamped)" if f == "thumb" else (" (few contacts)" if f == "little" else "")
            md.append(f"| {f}{note} | {1000 * v['median_abs_star']:.0f} | {1000 * v['median_abs_pred']:.0f} | "
                      f"{v['gain'][0]:.2f} [{v['gain'][1]:.2f}, {v['gain'][2]:.2f}] | {h[0]:.2f} | {v['median_cos']:.2f} |")
            tex.append(f"{f} & {1000 * v['median_abs_star']:.0f} & {1000 * v['median_abs_pred']:.0f} & "
                       f"{v['gain'][0]:.2f} [{v['gain'][1]:.2f}, {v['gain'][2]:.2f}] & {h[0]:.2f} & {v['median_cos']:.2f}\\\\")
        tex += [r"\bottomrule\end{tabular}\end{table}", ""]
    with open(os.path.join(OUT, "tables.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(OUT, "tables.tex"), "w") as fh:
        fh.write("% needs \\usepackage{booktabs}\n" + "\n".join(tex) + "\n")
    with open(os.path.join(OUT, "tables.json"), "w") as fh:
        json.dump(t, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("\n".join(md))
    print(f"[tables] {OUT}/tables.md, tables.tex, tables.json")


def main():
    global OUT
    demos, rolls = by_k(args.demos), by_k(args.rollouts)
    ks = [s["K"] for s in rolls]
    if [s["K"] for s in demos] != ks:
        raise SystemExit("the demos and rollouts must have the same two K")
    if any(s["results"] is None for s in rolls):
        raise SystemExit("the results JSON next to a rollout file is missing")
    stem = os.path.splitext(os.path.basename(args.rollouts[0]))[0].split("_K")[0]
    OUT = args.out or os.path.join("logs", "il_eval", "figures", stem)
    os.makedirs(OUT, exist_ok=True)
    tables = {}

    pol = [success_stats(s["results"]) for s in rolls]
    tables["success"] = [{**{k: v for k, v in s.items() if k not in ("success", "episode")}, "K": k}
                         for s, k in zip(pol, ks)]
    order = [np.argsort(s["episode"]) for s in pol]
    p, n10, n01 = mcnemar(pol[0]["success"][order[0]], pol[1]["success"][order[1]])
    tables["mcnemar"] = {"p": p, "n10": n10, "n01": n01}

    fig_outcomes(pol, ks, tables["mcnemar"])
    fig_success_map(rolls)
    fig_hold(demos, rolls, tables)
    fig_liftoff(demos, rolls)
    fig_precontact(rolls, tables)
    if args.kswap:
        fig_kswap(json.load(open(args.kswap)), tables)
    pair = pick_pair(rolls)
    fig_pair(rolls, pair)
    tables["pair_episode"] = int(pair[3:])
    tables["demo_joints"] = fig_demo_joints(demos)
    print(f"[fig] single-run pair: rollout episode {int(pair[3:])} (use it for the policy video: --only {int(pair[3:])})")
    write_tables(tables)


if __name__ == "__main__":
    main()
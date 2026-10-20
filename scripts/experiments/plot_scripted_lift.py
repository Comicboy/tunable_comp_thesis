"""Plots for the scripted lift runs (logs/scripted_lift/<controller>_<time>.npz).

For every run (one controller, one env per stiffness K):
    <run>_overview.png            cube height, thumb force, summed finger force, thumb offset, cube rotation
    <run>_fingers.png             contact force |f| and admittance offset |e| of every fingertip
    <run>_joints.png              q_ref, q_cmd and q of thumb joint3 and of one finger's joint1, one row per K
    <run>_torque.png              joint torque of each finger's main joint over time, and against K
    <run>_force_vs_angle.png      contact force against the joint reference penetration q_ref - q
    <run>_force_vs_position.png   contact force against the fingertip reference penetration |J (q_ref - q)|,
                                  with the series stiffness K_r = (1/K + 1/K_d)^-1 predicted from the rigid env
If runs of both controllers are given, also
    compare_controllers.png   thumb force over time for the K both runs share, the force at the end
                              of the squeeze against K, and the admittance law |e| against |f|/K
    compare_torque.png        main-joint torque at the end of the squeeze against K, both controllers

The "main joint" is joint3 of every finger: its torque follows the fingertip load. (joint1 of
fingers 2-5 is also loaded where the cube touches the finger above the tip, which the fingertip
admittance does not see, so its torque hardly changes with K.) Runs logged before the torque was recorded use the drive deflection q_cmd - q instead
(the PD torque is proportional to it at rest), and have no force-vs-position figure.

    python scripts/experiments/plot_scripted_lift.py logs/scripted_lift/dynamic_<time>.npz \
        logs/scripted_lift/static_<time>.npz [--out logs/scripted_lift/plots] [--finger 3] [--pdf]

Only numpy and matplotlib are needed (no Isaac Sim).
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FINGERS = ["thumb", "index", "middle", "ring", "little"]
# finger -> joint whose torque shows the fingertip load: joint3 of every finger. (joint1 of fingers 2-5
# carries an offset that does not change with K, from the cube touching the finger above the tip)
MAIN_JOINT = {1: 3, 2: 3, 3: 3, 4: 3, 5: 3}
LOADING = ("close", "squeeze")  # phases in which the fingers press on the cube resting on the table

# --- style ---------------------------------------------------------------------------------------
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE, PHASE_BAND = "#fcfcfb", "#f0efec"
# stiffness is an ordered magnitude: one blue ramp, softer = lighter (validated as an ordinal ramp);
# the rigid baseline (K = inf) is dark gray and dashed
K_STEPS = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]
RIGID = "#52514e"
# controller identity: categorical slots 1 and 2
CONTROLLER_COLOR = {"dynamic": "#2a78d6", "static": "#eb6834"}
CONTROLLER_STYLE = {"dynamic": "-", "static": "--"}
# fingers as categories: categorical slots 1-5 in fixed order
FINGER_COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
FINGER_MARKER = ["o", "s", "^", "D", "v"]

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


def load(path: str) -> dict:
    d = dict(np.load(path, allow_pickle=True))
    d["path"] = path
    d["name"] = os.path.splitext(os.path.basename(path))[0]
    d["controller"] = str(d["controller"])
    d["phase_names"] = [str(p) for p in d["phase_names"]]
    d["joint_names"] = [str(j) for j in d["joint_names"]]
    d["f"] = np.linalg.norm(d["tip_force"], axis=-1)  # (T, N, 5) [N]
    d["e"] = np.linalg.norm(d["tip_offset"], axis=-1) * 1000.0  # (T, N, 5) [mm]
    d["height"] = (d["cube_pos"][:, :, 2] - d["cube_start"][None, :, 2]) * 100.0  # (T, N) [cm]
    q0 = d["cube_quat"][0]
    dot = np.abs((d["cube_quat"] * q0[None]).sum(-1)).clip(0.0, 1.0)
    d["rot"] = np.degrees(2.0 * np.arccos(dot))  # (T, N) [deg] since the start
    main = [d["joint_names"].index(f"right_finger{f}_joint{MAIN_JOINT[f]}") for f in range(1, 6)]
    d["main"] = main
    # joint torque of the main joints (T, N, 5); old logs: drive deflection q_cmd - q instead
    if "tau_pd" in d:
        d["tau"], d["tau_unit"], d["tau_name"] = np.abs(d["tau_pd"][:, :, main]), "Nm", "joint torque |tau|"
    else:
        d["tau"] = np.abs(d["q_cmd"][:, :, main] - d["q"][:, :, main])
        d["tau_unit"], d["tau_name"] = "rad", "drive deflection |q_cmd - q| (proportional to the joint torque)"
    d["dq_ref"] = np.abs(d["q_ref"][:, :, main] - d["q"][:, :, main])  # (T, N, 5) [rad]
    d["delta_ref"] = np.linalg.norm(d["tip_ref_offset"], axis=-1) * 1000.0 if "tip_ref_offset" in d else None
    return d


def k_label(k: float) -> str:
    return "K = ∞ (rigid)" if np.isinf(k) else f"K = {k:g} N/m"


def k_style(k: float, all_k: list[float]) -> dict:
    """Color and dash for a stiffness; finite K take the ramp in increasing order across all runs."""
    if np.isinf(k):
        return {"color": RIGID, "linestyle": "--"}
    finite = sorted({x for x in all_k if not np.isinf(x)})
    steps = K_STEPS if len(finite) <= len(K_STEPS) else plt.cm.Blues(np.linspace(0.35, 0.95, len(finite)))
    # spread the runs' K over the ramp so the extremes use its ends
    idx = finite.index(k)
    pos = 0 if len(finite) == 1 else round(idx * (len(steps) - 1) / (len(finite) - 1))
    return {"color": steps[pos], "linestyle": "-"}


def shade_phases(ax, d, labels: bool = False):
    """Alternate light bands for the motion phases (labels on the top axis only)."""
    t, ph = d["t"], d["phase"]
    for p, name in enumerate(d["phase_names"]):
        idx = np.where(ph == p)[0]
        if idx.size == 0:
            continue
        t0, t1 = t[idx[0]] - (t[1] - t[0]), t[idx[-1]]
        if p % 2 == 1:
            ax.axvspan(t0, t1, color=PHASE_BAND, zorder=0, lw=0)
        if labels and t1 - t0 >= 0.8:  # short phases (settle) stay unlabelled so names do not collide
            ax.text((t0 + t1) / 2, 1.02, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=7.5, color=MUTED)


def phase_end(d, name: str) -> int:
    return int(np.where(d["phase"] == d["phase_names"].index(name))[0][-1])


def squeeze_end(d, arr: np.ndarray, window: float = 0.5) -> np.ndarray:
    """Mean of arr (T, ...) over the last `window` seconds of the squeeze: robust to oscillation."""
    i = phase_end(d, "squeeze")
    n = max(1, int(round(window / float(d["dt"]))))
    return arr[i - n + 1: i + 1].mean(axis=0)


def phase_mask(d, names) -> np.ndarray:
    return np.isin(d["phase"], [d["phase_names"].index(n) for n in names])


def k_axis(ax, all_k):
    """K as ordered, evenly spaced categories (1, 50, 75, ... , inf); returns a function K -> x."""
    ks = sorted(set(all_k), key=lambda k: (np.isinf(k), k))
    pos = {k: i for i, k in enumerate(ks)}
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(["∞" if np.isinf(k) else f"{k:g}" for k in ks])
    ax.set_xlim(-0.4, len(ks) - 0.6)
    ax.set_xlabel("K [N/m]  (ordered, not to scale)")
    return lambda K: np.array([pos[float(k)] for k in K], dtype=float)


def save(fig, out_dir: str, name: str, pdf: bool):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    if pdf:
        fig.savefig(os.path.join(out_dir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved {path}")


# --- per run ---------------------------------------------------------------------------------------
def plot_overview(d, all_k, out_dir, pdf):
    t, K = d["t"], d["stiffness"]
    rows = [
        ("Cube height above rest", "cm", d["height"]),
        ("Thumb contact force", "N", d["f"][:, :, 0]),
        ("Fingers 2-5 contact force (sum)", "N", d["f"][:, :, 1:].sum(-1)),
        ("Thumb admittance offset |e|", "mm", d["e"][:, :, 0]),
        ("Cube rotation", "deg", d["rot"]),
    ]
    fig, axes = plt.subplots(len(rows), 1, figsize=(8.5, 10.5), sharex=True)
    for r, (ax, (title, unit, y)) in enumerate(zip(axes, rows)):
        shade_phases(ax, d, labels=(r == 0))
        for i, k in enumerate(K):
            ax.plot(t, y[:, i], label=k_label(k), **k_style(k, all_k))
        ax.set_title(title, pad=14 if r == 0 else 6)
        ax.set_ylabel(unit)
    axes[-1].set_xlabel("time [s]")
    axes[0].legend(loc="center left", ncol=2)
    fig.suptitle(f"Scripted lift, {d['controller']} admittance ({d['name']})", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, out_dir, f"{d['name']}_overview", pdf)


def plot_fingers(d, all_k, out_dir, pdf):
    t, K = d["t"], d["stiffness"]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=True, sharey="row")
    for f in range(5):
        for r, (y, unit) in enumerate([(d["f"], "N"), (d["e"], "mm")]):
            ax = axes[r, f]
            shade_phases(ax, d, labels=False)
            for i, k in enumerate(K):
                ax.plot(t, y[:, i, f], label=k_label(k), **k_style(k, all_k))
            ax.set_title(f"{FINGERS[f]}: {'contact force |f|' if r == 0 else 'offset |e|'}")
            if f == 0:
                ax.set_ylabel(unit)
            if r == 1:
                ax.set_xlabel("time [s]")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle(f"Fingertips, {d['controller']} admittance ({d['name']}); shaded bands = phases "
                 + ", ".join(d["phase_names"]), x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, out_dir, f"{d['name']}_fingers", pdf)


def plot_joints(d, all_k, out_dir, pdf, finger: int):
    t, K, names = d["t"], d["stiffness"], d["joint_names"]
    cols = [("thumb joint3", names.index("right_finger1_joint3")),
            (f"{FINGERS[finger - 1]} joint1", names.index(f"right_finger{finger}_joint1"))]
    fig, axes = plt.subplots(len(K), 2, figsize=(11, 2.2 * len(K) + 0.8), sharex=True, squeeze=False)
    for i, k in enumerate(K):
        st = k_style(k, all_k)
        for c, (label, j) in enumerate(cols):
            ax = axes[i, c]
            shade_phases(ax, d, labels=(i == 0))
            ax.plot(t, d["q_ref"][:, i, j], color=MUTED, linestyle=":", linewidth=1.6, label="q_ref (script)")
            ax.plot(t, d["q_cmd"][:, i, j], color=st["color"], linestyle="--", linewidth=1.4,
                    label="q_cmd = q_ref + J⁺e")
            ax.plot(t, d["q"][:, i, j], color=st["color"], linestyle="-", label="q (measured)")
            ax.set_title(f"{label}, {k_label(k)}", pad=14 if i == 0 else 6)
            if c == 0:
                ax.set_ylabel("rad")
    for ax in axes[-1]:
        ax.set_xlabel("time [s]")
    axes[0, 1].legend(loc="upper left")
    fig.suptitle(f"Joint reference, command and position, {d['controller']} admittance ({d['name']})",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, out_dir, f"{d['name']}_joints", pdf)


def plot_torque(d, all_k, out_dir, pdf):
    """Main-joint torque of every finger over time (top) and at the end of the squeeze against K (bottom)."""
    t, K = d["t"], d["stiffness"]
    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.1])
    top = []
    for f in range(5):
        ax = fig.add_subplot(gs[0, f], sharey=top[0] if top else None)
        top.append(ax)
        shade_phases(ax, d, labels=False)
        for i, k in enumerate(K):
            ax.plot(t, d["tau"][:, i, f], label=k_label(k), **k_style(k, all_k))
        jn = MAIN_JOINT[f + 1]
        ax.set_title(f"{FINGERS[f]} joint{jn}")
        ax.set_xlabel("time [s]")
        if f == 0:
            ax.set_ylabel(d["tau_unit"])
            ax.legend(loc="upper left")
    ax = fig.add_subplot(gs[1, :])
    to_x = k_axis(ax, all_k)
    x = to_x(K)
    order = np.argsort(x)
    tau_end = squeeze_end(d, d["tau"])  # (N, 5)
    for f in range(5):
        y = tau_end[order, f]
        ax.plot(x[order], y, color=FINGER_COLOR[f], marker=FINGER_MARKER[f], markersize=8, label=FINGERS[f])
        ax.annotate(FINGERS[f], (x[order][-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2)
    ax.set_xlim(-0.4, len(set(all_k)) - 0.2)  # room for the direct labels
    ax.set_ylabel(d["tau_unit"])
    ax.set_title("End of the squeeze (mean of the last 0.5 s): a stiffer fingertip presses harder, so its joint "
                 "carries more torque")
    ax.legend(loc="upper left", ncol=5)
    fig.suptitle(f"Main-joint {d['tau_name']}, {d['controller']} admittance ({d['name']})", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, out_dir, f"{d['name']}_torque", pdf)


def plot_force_vs(d, all_k, out_dir, pdf, kind: str):
    """Contact force against the reference penetration while the cube rests on the table (close, squeeze).

    kind "angle": x = |q_ref - q| of the main joint [rad] (joint space)
    kind "position": x = |J (q_ref - q)| [mm], how far the reference puts the fingertip into the cube;
    the slope is the stiffness the fingertip renders, K_r = (1/K + 1/K_d)^-1, with K_d (the drives'
    stiffness at the fingertip) fitted on the rigid env."""
    K, m = d["stiffness"], phase_mask(d, LOADING)
    if kind == "angle":
        x_all, x_label, x_unit = d["dq_ref"], "joint reference penetration |q_ref - q| (main joint)", "rad"
    else:
        x_all, x_label, x_unit = d["delta_ref"], "fingertip reference penetration |J (q_ref - q)|", "mm"
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.4), sharey=True)
    kd = np.full(5, np.nan)
    for f in range(5):
        ax = axes[f]
        for i, k in enumerate(K):
            x, y = x_all[m, i, f], d["f"][m, i, f]
            ax.plot(x, y, label=k_label(k), linewidth=1.4, **k_style(k, all_k))
            if kind == "position" and np.isinf(k):  # drives' fingertip stiffness: slope through the origin
                use = y > 0.3
                if use.sum() >= 5:
                    kd[f] = (x[use] * y[use]).sum() / (x[use] ** 2).sum()  # [N/mm]
        if kind == "position" and np.isfinite(kd[f]):
            xm = np.nanmax(x_all[m, :, f])
            for k in K:  # predicted series stiffness, dotted
                kr = kd[f] if np.isinf(k) else 1.0 / (1.0 / (k / 1000.0) + 1.0 / kd[f])
                st = k_style(k, all_k)
                ax.plot([0, xm], [0, kr * xm], color=st["color"], linestyle=":", linewidth=1.2)
            ax.text(0.98, 0.04, f"K_d ≈ {1000 * kd[f]:.0f} N/m", transform=ax.transAxes, ha="right",
                    fontsize=8, color=INK_2)
        ax.set_title(FINGERS[f])
        ax.set_xlabel(x_unit)
        if f == 0:
            ax.set_ylabel("contact force |f| [N]")
    axes[0].legend(loc="upper left")
    note = ("; dotted: K_r = (1/K + 1/K_d)^-1, K_d fitted on the rigid env" if kind == "position" else "")
    fig.suptitle(f"Contact force vs {x_label}, close + squeeze, {d['controller']} admittance ({d['name']}){note}",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, out_dir, f"{d['name']}_force_vs_{kind}", pdf)


# --- controller comparison -------------------------------------------------------------------------
def plot_compare(runs, out_dir, pdf):
    shared = sorted(set.intersection(*[set(r["stiffness"].tolist()) for r in runs]), key=lambda k: (np.isinf(k), k))
    if len(shared) > 4:  # at most four panels: spread over the range, always with the softest and rigid
        shared = [shared[i] for i in sorted(set(np.linspace(0, len(shared) - 1, 4).round().astype(int)))]
    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(2, max(len(shared), 2))

    # top row: thumb force over time for each K both runs have
    for c, k in enumerate(shared):
        ax = fig.add_subplot(gs[0, c])
        shade_phases(ax, runs[0], labels=False)
        for r in runs:
            i = int(np.where(r["stiffness"] == k)[0][0])
            ax.plot(r["t"], r["f"][:, i, 0], color=CONTROLLER_COLOR.get(r["controller"], INK),
                    linestyle=CONTROLLER_STYLE.get(r["controller"], "-"), label=r["controller"])
        ax.set_title(f"Thumb force, {k_label(k)}")
        ax.set_xlabel("time [s]")
        if c == 0:
            ax.set_ylabel("N")
            ax.legend(loc="upper left")

    # bottom left: force at the end of the squeeze against K (thumb and fingers 2-5)
    ax = fig.add_subplot(gs[1, : max(len(shared), 2) // 2])
    to_x = k_axis(ax, [k for r in runs for k in r["stiffness"].tolist()])
    for r in runs:
        f_end = squeeze_end(r, r["f"])  # (N, 5)
        x = to_x(r["stiffness"])
        order = np.argsort(x)
        col = CONTROLLER_COLOR.get(r["controller"], INK)
        ax.plot(x[order], f_end[order, 0], marker="o", markersize=7, color=col,
                linestyle=CONTROLLER_STYLE.get(r["controller"], "-"), label=f"{r['controller']}: thumb")
        ax.plot(x[order], f_end[order, 1:].sum(-1), marker="s", markersize=7, color=col,
                linestyle=":", label=f"{r['controller']}: fingers 2-5 (sum)")
    ax.set_ylabel("N")
    ax.set_title("Contact force at the end of the squeeze (mean of the last 0.5 s)")
    ax.legend(loc="upper left")

    # bottom right: the admittance law, |e| against |f|/K (squeeze phase, all fingertips, finite K)
    ax = fig.add_subplot(gs[1, max(len(shared), 2) // 2:])
    lim = 0.0
    for r in runs:
        idx = np.where(r["phase"] == r["phase_names"].index("squeeze"))[0]
        for i, k in enumerate(r["stiffness"]):
            if np.isinf(k):
                continue
            fk = r["f"][idx, i].mean(0) / k * 1000.0  # (5,) [mm]
            e = r["e"][idx, i].mean(0)
            ax.scatter(fk, e, s=36, color=CONTROLLER_COLOR.get(r["controller"], INK),
                       marker="o" if r["controller"] == "dynamic" else "^", edgecolors=SURFACE, linewidths=1.0,
                       label=r["controller"] if i == 0 else None, zorder=3)
            lim = max(lim, fk.max(), e.max())
    lim = min(lim * 1.1, 80.0)
    ax.plot([0, lim], [0, lim], color=MUTED, linestyle=":", linewidth=1.2, label="|e| = |f| / K")
    ax.axhline(30.0, color=AXIS, linewidth=1.0)
    ax.text(lim, 30.0, "clamp 30 mm", ha="right", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, max(lim, 32))
    ax.set_xlabel("|f| / K [mm]   (mean over the squeeze)")
    ax.set_ylabel("|e| [mm]")
    ax.set_title("Admittance law at steady state (every fingertip, finite K)")
    ax.legend(loc="upper left")

    fig.suptitle("Dynamic vs static admittance: " + " vs ".join(r["name"] for r in runs)
                 + "  (shaded: approach, close, lift)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, out_dir, "compare_controllers", pdf)


def plot_compare_torque(runs, all_k, out_dir, pdf):
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.2), sharey=True)
    for f in range(5):
        ax = axes[f]
        to_x = k_axis(ax, all_k)
        for r in runs:
            x = to_x(r["stiffness"])
            order = np.argsort(x)
            ax.plot(x[order], squeeze_end(r, r["tau"])[order, f], marker="o", markersize=7,
                    color=CONTROLLER_COLOR.get(r["controller"], INK),
                    linestyle=CONTROLLER_STYLE.get(r["controller"], "-"), label=r["controller"])
        ax.set_title(f"{FINGERS[f]} joint{MAIN_JOINT[f + 1]}")
        if f == 0:
            ax.set_ylabel(runs[0]["tau_unit"])
            ax.legend(loc="upper left")
    fig.suptitle(f"Main-joint {runs[0]['tau_name']} at the end of the squeeze (mean of the last 0.5 s) against K",
                 x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, out_dir, "compare_torque", pdf)


def print_table(runs):
    print("\nend of squeeze / end of hold: |f| [N] thumb, fingers 2-5 sum; |e| thumb [mm]; height [cm]; rotation [deg]")
    for r in runs:
        i_sq, i_end = phase_end(r, "squeeze"), len(r["t"]) - 1
        for i, k in enumerate(r["stiffness"]):
            print(f"  {r['controller']:>7} {k_label(k):>16} | squeeze: {r['f'][i_sq, i, 0]:5.2f} "
                  f"{r['f'][i_sq, i, 1:].sum():5.2f} {r['e'][i_sq, i, 0]:5.1f} | hold: {r['f'][i_end, i, 0]:5.2f} "
                  f"{r['f'][i_end, i, 1:].sum():5.2f} {r['e'][i_end, i, 0]:5.1f} | h {r['height'][i_end, i]:5.1f} "
                  f"rot {r['rot'][i_end, i]:4.1f}")


def main():
    parser = argparse.ArgumentParser(description="Plot scripted lift runs.")
    parser.add_argument("runs", nargs="+", help="npz files written by scripted_lift.py")
    parser.add_argument("--out", default=os.path.join("logs", "scripted_lift", "plots"))
    parser.add_argument("--finger", type=int, default=3, help="finger (2-5) whose joint1 the joint plot shows")
    parser.add_argument("--pdf", action="store_true", help="also save PDFs (for the thesis)")
    args = parser.parse_args()

    runs = [load(p) for p in args.runs]
    all_k = [float(k) for r in runs for k in r["stiffness"]]
    for d in runs:
        plot_overview(d, all_k, args.out, args.pdf)
        plot_fingers(d, all_k, args.out, args.pdf)
        plot_joints(d, all_k, args.out, args.pdf, args.finger)
        plot_torque(d, all_k, args.out, args.pdf)
        plot_force_vs(d, all_k, args.out, args.pdf, "angle")
        if d["delta_ref"] is not None:
            plot_force_vs(d, all_k, args.out, args.pdf, "position")
        else:
            print(f"[INFO] {d['name']}: no fingertip reference penetration logged (older run), "
                  "skipping force_vs_position")
    if len({r["controller"] for r in runs}) > 1:
        plot_compare(runs, args.out, args.pdf)
        plot_compare_torque(runs, all_k, args.out, args.pdf)
    print_table(runs)


if __name__ == "__main__":
    main()
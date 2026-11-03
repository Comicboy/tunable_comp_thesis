"""Offline K-swap test: does the policy change its finger commands with K the way the admittance law does?

Runs in the "lerobot" conda env (needs the trained checkpoint, torch, h5py, matplotlib); no simulation.
Every recorded demo state is fed to the policy twice, once with log10 K = log10 150 and once with
log10 500 (one of them is the demo's own K), everything else in the state unchanged. The difference
of the two predicted actions,

    D_pred = a(s, K150) - a(s, K500),

is compared with what the admittance law would change at that state with the operator's intention
q_ref the same:

    q_cmd = q_ref + J^+ e,  |e| = min(|f| / K, 30 mm) along f at steady state (the admittance clamps |e|
    per fingertip at max_offset = 30 mm)  =>  D* = J^+ (e(K150) - e(K500))

From the demos' recorded admittance offset dq = J^+ e (at the demo's own K_rec; dq is linear in e) this
is, joint by joint, with f the smoothed fingertip force in the state (the force the admittance acts on):

    D* = dq * (min(|f|/K150, c) - min(|f|/K500, c)) / min(|f|/K_rec, c),   c = 30 mm

i.e. 0.7 dq in a K150 demo and 2.33 dq in a K500 demo while no offset is clamped, and 0 where |e| is at
the clamp for both K (|f| > 15 N, e.g. the thumb in the hold), so J^+ is not needed. Joints whose q_cmd was clamped to a joint limit in the recording are left out
(the law does not hold there; the share is printed). The relation is a steady-state one: while the
force changes, dq lags behind f / K (time scale sqrt(M / K), 24-45 ms), so the hold is the cleanest part.

    python scripts/il/kswap_test.py \\
        --policy logs/il_train/act_qcmd_v1/checkpoints/last/pretrained_model \\
        --demos logs/il_demos/lift_K150.hdf5 logs/il_demos/lift_K500.hdf5 [--out DIR] [--pdf]

What it shows (printed, plus kswap_time.png and kswap_hold.png in --out, default
logs/il_eval/analysis/<run>_kswap):

  before contact (no force yet; dq = 0): D* = 0, so any D_pred is the policy using K for something
      other than the admittance (e.g. a difference between the two demo sets). Clear-cut test.
  in contact (finger force > 0.5 N), per finger: the gain g = sum(D_pred . D*) / sum(|D*|^2), i.e. how
      much of the admittance law's change the policy's change contains (1: as the law; 0: the policy
      does not change the command with K in contact), with a 95 % bootstrap interval over the demos,
      the median cosine between D_pred and D*, and the medians of |D*| and |D_pred|. g_adj: the same
      after subtracting from each demo's D_pred its mean over the last 0.2 s before the first contact,
      i.e. the K offset the policy already had without force (a confound); g_adj separates the
      yield-like part from that offset as long as the offset stays about constant after contact.
      Caveat for reading g: in the hold the demos also satisfy the drive equilibrium q_cmd ~ q + tau/K_p
      (tau the joint torque from the fingertip force), which does not contain K. Given the measured q
      and f, that alone reproduces the recorded command, and both descriptions fit the training data
      equally. Swapping K at a fixed state asks about a state the demos never showed with the other K,
      so the answer is the network's extrapolation: g measures which description it uses, it is not a
      pass/fail like the test before contact.
  arm: the arm targets should not change with K at all (the admittance acts on the fingers only).
  fit: how close the policy's prediction at the demo's own K is to the recorded action (for scale).

The action used is the first action of the predicted chunk at that state (what the policy outputs
with no history; in the rollouts temporal ensembling also averaged older predictions).
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies import get_policy_class, make_pre_post_processors  # noqa: E402

parser = argparse.ArgumentParser(description="Offline K-swap test of a q_cmd policy on the demo states.")
parser.add_argument("--policy", required=True, help="checkpoint folder (…/checkpoints/<step>/pretrained_model)")
parser.add_argument("--demos", nargs=2, required=True, help="the two demo files (one per K)")
parser.add_argument("--out", default=None, help="folder for plots and the summary JSON")
parser.add_argument("--device", default="cuda")
parser.add_argument("--batch", type=int, default=1024, help="states per forward pass")
parser.add_argument("--hold_s", type=float, default=2.5, help="hold window: the last hold_s of each demo [s]")
parser.add_argument("--boot", type=int, default=2000, help="bootstrap resamples (over demos) for the gain intervals")
parser.add_argument("--task", default="Lift the cube 15 cm above the table and hold it there")
parser.add_argument("--pdf", action="store_true", help="also save PDFs")
args = parser.parse_args()

STATE, ENV_STATE = "observation.state", "observation.environment_state"
ENV_TERMS = ("cube_position",)  # -> environment_state, as in convert_to_lerobot.py
FINGERS = ["thumb", "index", "middle", "ring", "little"]
CONTACT_N = 0.5   # [N] smoothed fingertip force: the finger is in contact
TOUCH_N = 1e-3    # [N] summed smoothed force: first contact of the demo
PRE_N = 10        # samples (0.2 s) before the first contact: the policy's K offset there is removed for g_adj
CLAMP_TOL = 1e-4  # [rad] |q_cmd - (q_ref + dq)| above this: q_cmd was clamped to a joint limit
MAX_OFFSET = 0.03  # [m] the admittance's clamp on |e| per fingertip (AdmittanceControllerCfg.max_offset)
WINDOW = (-2.0, 5.0)  # [s] around the first contact in the time plot

# --- style (as analyse_qcmd_rollouts.py) ----------------------------------------------------------
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
K_COLOR = {150.0: "#5598e7", 500.0: "#184f95"}
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


def k_color(k):
    return K_COLOR.get(float(k), INK)


# ---------------------------------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------------------------------
def load_policy():
    """As policy_server.py: weights, config and the pre/post-processors from the checkpoint."""
    path = str(Path(args.policy))
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.pretrained_path = path
    cfg.device = args.device
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
    policy.eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=path,
                                         preprocessor_overrides={"device_processor": {"device": args.device}})
    return cfg, policy, pre, post


def predict(policy, pre, post, states: np.ndarray, envs: np.ndarray) -> np.ndarray:
    """First action of the predicted chunk for every state (no action queue, no ensembling): (N, 26)."""
    out = []
    with torch.inference_mode():
        for i in range(0, len(states), args.batch):
            batch = {STATE: torch.from_numpy(states[i:i + args.batch]),
                     ENV_STATE: torch.from_numpy(envs[i:i + args.batch]),
                     "task": [args.task] * len(states[i:i + args.batch]), "robot_type": "ur5_wuji"}
            chunk = policy.predict_action_chunk(pre(batch))  # (B, chunk, 26), normalised
            out.append(post(chunk[:, 0]).detach().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


# ---------------------------------------------------------------------------------------------------
# demos
# ---------------------------------------------------------------------------------------------------
def load_demos(path):
    with h5py.File(path, "r") as f:
        a = dict(f.attrs)
        if "q_cmd" not in str(a.get("format", "")):
            raise SystemExit(f"{path}: not a demo file with q_cmd as the action")
        terms = json.loads(a["obs_terms"])
        hand = json.loads(a["hand_joints"])
        info = {"K": float(a["stiffness"]), "dt": float(a["step_dt"]), "n_arm": len(json.loads(a["arm_joints"])),
                "arm_joints": json.loads(a["arm_joints"]),
                "finger_idx": [[i for i, n in enumerate(hand) if n.startswith(f"right_finger{k}_")] for k in range(1, 6)],
                "terms": terms, "demos": []}
        for name in sorted(f["data"].keys()):
            g = f["data"][name]
            info["demos"].append({"name": name, "obs": g["obs"][()].astype(np.float32), "act": g["actions"][()],
                                  "q_ref": g["q_ref"][()], "dq": g["dq"][()]})
    return info


def law_factor(forces: np.ndarray, k_rec: float, k_lo: float, k_hi: float) -> np.ndarray:
    """Per sample and finger: D* / dq for the admittance law with its offset clamp, (T, 5).

    At steady state |e| = min(|f| / K, MAX_OFFSET) along f, and dq = J^+ e is linear in e, so the change
    from K_hi to K_lo at the same force is dq * (|e|(K_lo) - |e|(K_hi)) / |e|(K_rec). Without the clamp
    this is K_rec (1/K_lo - 1/K_hi); with both K at the clamp it is 0. For (almost) no force the
    unclamped value is used (the ratio is independent of |f| there)."""
    e = lambda k: np.minimum(forces / k, MAX_OFFSET)  # noqa: E731
    unclamped = k_rec * (1.0 / k_lo - 1.0 / k_hi)
    with np.errstate(invalid="ignore", divide="ignore"):
        fac = (e(k_lo) - e(k_hi)) / e(k_rec)
    return np.where(forces > 1e-6, fac, unclamped)


def finger_cols(fidx):
    """Column order that maps the finger-repeated factor (finger 1 x4, finger 2 x4, ...) onto the hand joints."""
    order = np.empty(20, int)
    for f, ix in enumerate(fidx):
        for j, col in enumerate(ix):
            order[col] = 4 * f + j
    return order


def split_state(terms):
    """Indices of observation.state and observation.environment_state in the 51 numbers, and of log10 K in the state."""
    state_idx, env_idx = [], []
    for name, s, e in terms:
        (env_idx if name in ENV_TERMS else state_idx).extend(range(s, e))
    k_obs = [s for name, s, e in terms if name == "stiffness"][0]
    force_sl = [slice(s, e) for name, s, e in terms if name == "fingertip_forces"][0]
    return state_idx, env_idx, state_idx.index(k_obs), force_sl


# ---------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------
def main():
    sets = sorted((load_demos(p) for p in args.demos), key=lambda s: s["K"])
    k_lo, k_hi = sets[0]["K"], sets[1]["K"]
    if k_lo == k_hi:
        raise SystemExit("both demo files have the same K")
    cfg, policy, pre, post = load_policy()
    run = Path(args.policy).parts
    run = run[run.index("il_train") + 1] if "il_train" in run else Path(args.policy).name
    out_dir = args.out or os.path.join("logs", "il_eval", "analysis", f"{run}_kswap")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[kswap] policy {args.policy} ({cfg.type}); swapping log10 K between {k_lo:g} and {k_hi:g}")

    per_demo = []  # one dict per demo with per-step arrays
    for s in sets:
        state_idx, env_idx, k_pos, force_sl = split_state(s["terms"])
        n_arm, fidx = s["n_arm"], s["finger_idx"]
        for d in s["demos"]:
            obs, T = d["obs"], len(d["obs"])
            st, env = obs[:, state_idx].copy(), obs[:, env_idx].copy()
            st_lo, st_hi = st.copy(), st.copy()
            st_lo[:, k_pos], st_hi[:, k_pos] = np.log10(k_lo), np.log10(k_hi)
            both = predict(policy, pre, post, np.concatenate([st_lo, st_hi]), np.concatenate([env, env]))
            a_lo, a_hi = both[:T], both[T:]
            d_pred = a_lo - a_hi  # (T, 26)
            q_cmd = d["act"][:, n_arm:]
            clamped = np.abs(q_cmd - (d["q_ref"] + d["dq"])) > CLAMP_TOL  # (T, 20)
            forces = np.linalg.norm(obs[:, force_sl].reshape(T, 5, 3), axis=-1)  # (T, 5) smoothed, as the admittance sees it
            d_star = d["dq"] * np.repeat(law_factor(forces, s["K"], k_lo, k_hi), 4, axis=1)[:, finger_cols(fidx)]
            hand_pred = np.where(clamped, 0.0, d_pred[:, n_arm:])
            hand_star = np.where(clamped, 0.0, d_star)
            touch = np.nonzero((forces.sum(-1) > TOUCH_N) | (np.abs(d["dq"]).max(-1) > 0))[0]
            a_own = a_lo if s["K"] == k_lo else a_hi
            first = int(touch[0]) if touch.size else T
            # the policy's K offset just before contact (no force yet): mean D_pred over the last PRE_N samples
            off = d_pred[max(first - PRE_N, 0):first, n_arm:].mean(0) if first > 0 else np.zeros(d_pred.shape[1] - n_arm)
            hand_adj = np.where(clamped, 0.0, d_pred[:, n_arm:] - off)
            per_demo.append({
                "K": s["K"], "name": d["name"], "dt": s["dt"], "T": T, "first": first,
                "forces": forces, "clamped": clamped, "finger_idx": fidx,
                "arm_pred": d_pred[:, :n_arm],
                "pred": [hand_pred[:, ix] for ix in fidx], "star": [hand_star[:, ix] for ix in fidx],
                "pred_adj": [hand_adj[:, ix] for ix in fidx],
                "pred_raw": [d_pred[:, n_arm:][:, ix] for ix in fidx],
                "fit_arm": np.abs(a_own[:, :n_arm] - d["act"][:, :n_arm]).mean(-1),
                "fit_hand": [np.linalg.norm(a_own[:, n_arm:][:, ix] - q_cmd[:, ix], axis=-1) for ix in fidx],
            })
        print(f"[kswap] {len(s['demos'])} demos at K = {s['K']:g} done")
    arm_names = [n.replace("_joint", "") for n in sets[0]["arm_joints"]]
    summary = {"policy": args.policy, "demos": args.demos, "K": [k_lo, k_hi]}

    # --- fit ------------------------------------------------------------------------------------------
    print("\n=== fit at the demo's own K: median |prediction - recorded action| [mrad] (arm: mean over joints; "
          "fingers: norm over 4 joints)")
    print(f"{'':<14}{'arm':>9}" + "".join(f"{f:>9}" for f in FINGERS))
    for k in (k_lo, k_hi):
        ds = [d for d in per_demo if d["K"] == k]
        arm = np.median(np.concatenate([d["fit_arm"] for d in ds]))
        fing = [np.median(np.concatenate([d["fit_hand"][f] for d in ds])) for f in range(5)]
        print(f"{f'demos K{k:g}':<14}{1000 * arm:>9.1f}" + "".join(f"{1000 * v:>9.1f}" for v in fing))

    # --- before contact ---------------------------------------------------------------------------------
    pre_rows = {}
    print(f"\n=== before contact (no force, dq = 0, so the admittance law changes nothing): |D_pred| = "
          f"|a(K{k_lo:g}) - a(K{k_hi:g})| [mrad], median / 90th percentile over the samples")
    print(f"{'':<22}" + "".join(f"{f:>14}" for f in FINGERS) + f"{'arm (max joint)':>18}")
    for k in (k_lo, k_hi, None):
        ds = [d for d in per_demo if k is None or d["K"] == k]
        n = sum(d["first"] for d in ds)
        fing = [np.concatenate([np.linalg.norm(d["pred_raw"][f][:d["first"]], axis=-1) for d in ds]) for f in range(5)]
        arm = np.concatenate([np.abs(d["arm_pred"][:d["first"]]).max(-1) for d in ds])
        label = f"demos K{k:g}" if k is not None else "all demos"
        cells = [f"{1000 * np.median(v):.1f} / {1000 * np.percentile(v, 90):.1f}" for v in fing]
        print(f"{label + f' ({n})':<22}" + "".join(f"{c:>14}" for c in cells)
              + f"{f'{1000 * np.median(arm):.1f} / {1000 * np.percentile(arm, 90):.1f}':>18}")
        pre_rows[label] = {"samples": int(n), **{FINGERS[f]: [float(np.median(v)), float(np.percentile(v, 90))]
                                                 for f, v in enumerate(fing)},
                           "arm_max_joint": [float(np.median(arm)), float(np.percentile(arm, 90))]}
    summary["before_contact_rad_median_p90"] = pre_rows
    print("  per arm joint, all demos, median |D_pred| [mrad]: " + ", ".join(
        f"{n} {1000 * np.median(np.concatenate([np.abs(d['arm_pred'][:d['first'], j]) for d in per_demo])):.1f}"
        for j, n in enumerate(arm_names)))

    # --- in contact: gain against the admittance law ------------------------------------------------------
    rng = np.random.default_rng(0)
    n_hold = lambda d: int(round(args.hold_s / d["dt"]))  # noqa: E731

    def masks(d, f, which):
        c = d["forces"][:, f] > CONTACT_N
        if which == "hold":
            h = np.zeros(d["T"], bool)
            h[-n_hold(d):] = True
            c &= h
        return c

    def arm_mask(d, which):
        """Any finger in contact (and, for the hold, in the last hold_s)."""
        m = (d["forces"] > CONTACT_N).any(-1)
        if which == "hold":
            m &= np.arange(d["T"]) >= d["T"] - n_hold(d)
        return m

    def gain_table(which):
        res = {}
        print(f"\n=== {'in contact' if which == 'contact' else f'hold (last {args.hold_s:g} s), in contact'} "
              f"(finger force > {CONTACT_N} N): D_pred against the admittance law's D* (with the {1000 * MAX_OFFSET:g} mm offset clamp)")
        print(f"{'':<14}{'finger':<8}{'samples':>8}{'clamped %':>10}{'med |D*|':>10}{'med |D_pred|':>13}"
              f"{'gain g [95 %]':>25}{'g_adj [95 %]':>25}{'med cos':>9}   [mrad]")
        for k in (k_lo, k_hi, None):
            ds = [d for d in per_demo if k is None or d["K"] == k]
            label = f"demos K{k:g}" if k is not None else "all demos"
            for f in range(5):
                num = np.array([(d["pred"][f][masks(d, f, which)] * d["star"][f][masks(d, f, which)]).sum() for d in ds])
                den = np.array([(d["star"][f][masks(d, f, which)] ** 2).sum() for d in ds])
                ns = int(sum(masks(d, f, which).sum() for d in ds))
                if ns < 20 or den.sum() <= 0:
                    print(f"{label:<14}{FINGERS[f]:<8}{ns:>8}   (too few contact samples)")
                    continue
                num_adj = np.array([(d["pred_adj"][f][masks(d, f, which)] * d["star"][f][masks(d, f, which)]).sum() for d in ds])
                g, g_adj = num.sum() / den.sum(), num_adj.sum() / den.sum()
                bi = rng.integers(0, len(ds), (args.boot, len(ds)))
                with np.errstate(invalid="ignore", divide="ignore"):
                    gb = num[bi].sum(1) / den[bi].sum(1)
                    gb_adj = num_adj[bi].sum(1) / den[bi].sum(1)
                lo, hi = np.nanpercentile(gb, [2.5, 97.5])
                lo_a, hi_a = np.nanpercentile(gb_adj, [2.5, 97.5])
                ps = np.concatenate([d["pred"][f][masks(d, f, which)] for d in ds])
                ss = np.concatenate([d["star"][f][masks(d, f, which)] for d in ds])
                cl = np.concatenate([d["clamped"][:, d["finger_idx"][f]][masks(d, f, which)] for d in ds])
                n_p, n_s = np.linalg.norm(ps, axis=-1), np.linalg.norm(ss, axis=-1)
                ok = n_s > 1e-6
                cos = np.median((ps[ok] * ss[ok]).sum(-1) / (n_p[ok] * n_s[ok] + 1e-12)) if ok.any() else np.nan
                print(f"{label:<14}{FINGERS[f]:<8}{ns:>8}{100 * cl.mean():>10.1f}{1000 * np.median(n_s):>10.1f}"
                      f"{1000 * np.median(n_p):>13.1f}{f'{g:+.2f} [{lo:+.2f}, {hi:+.2f}]':>25}"
                      f"{f'{g_adj:+.2f} [{lo_a:+.2f}, {hi_a:+.2f}]':>25}{cos:>9.2f}")
                res[f"{label} {FINGERS[f]}"] = {"samples": ns, "clamped_share": float(cl.mean()),
                                                "median_abs_star": float(np.median(n_s)), "median_abs_pred": float(np.median(n_p)),
                                                "gain": [float(g), float(lo), float(hi)],
                                                "gain_offset_removed": [float(g_adj), float(lo_a), float(hi_a)],
                                                "median_cos": float(cos)}
            arm = np.concatenate([np.abs(d["arm_pred"][arm_mask(d, which)]).max(-1) for d in ds])
            print(f"{label:<14}{'arm':<8}{len(arm):>8}{'':>10}{'0':>10}{1000 * np.median(arm):>13.1f}"
                  f"{'(largest joint; should be 0)':>34}")
        return res

    summary["in_contact"] = gain_table("contact")
    summary["hold"] = gain_table("hold")
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)

    plot_time(per_demo, (k_lo, k_hi), out_dir)
    plot_hold(per_demo, (k_lo, k_hi), out_dir, n_hold)
    print(f"\n[kswap] summary -> {os.path.join(out_dir, 'summary.json')}")


# ---------------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------------
def save(fig, out_dir, name):
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    if args.pdf:
        fig.savefig(os.path.join(out_dir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


def band(rows):
    """Median and quartiles over demos (rows with NaN where a demo has no sample)."""
    m = np.array(rows)
    enough = (~np.isnan(m)).sum(0) >= 3
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return tuple(np.where(enough, np.nanpercentile(m, p, 0), np.nan) for p in (50, 25, 75))


def plot_time(per_demo, ks, out_dir):
    """|D_pred| and |D*| per finger (and the arm) around the first contact, median and IQR over the demos, per K."""
    fig, axes = plt.subplots(6, 2, figsize=(12, 14), sharex=True)
    for c, k in enumerate(ks):
        ds = [d for d in per_demo if d["K"] == k]
        dt = ds[0]["dt"]
        lag = np.arange(int(round(WINDOW[0] / dt)), int(round(WINDOW[1] / dt)) + 1)
        t = lag * dt
        for r in range(6):
            pred_rows, star_rows = [], []
            for d in ds:
                if r < 5:
                    yp, ys = np.linalg.norm(d["pred"][r], axis=-1), np.linalg.norm(d["star"][r], axis=-1)
                else:
                    yp, ys = np.abs(d["arm_pred"]).max(-1), np.zeros(d["T"])
                idx = d["first"] + lag
                ok = (idx >= 0) & (idx < d["T"])
                wp, ws = np.full(len(lag), np.nan), np.full(len(lag), np.nan)
                wp[ok], ws[ok] = yp[idx[ok]], ys[idx[ok]]
                pred_rows.append(wp)
                star_rows.append(ws)
            ax = axes[r, c]
            med, q1, q3 = band(pred_rows)
            ax.fill_between(t, 1000 * q1, 1000 * q3, color=k_color(k), alpha=0.15, lw=0)
            ax.plot(t, 1000 * med, color=k_color(k), label="policy: |a(K150) - a(K500)|")
            med_s, q1_s, q3_s = band(star_rows)
            ax.plot(t, 1000 * med_s, color=INK_2, linestyle="--", label="admittance law: |D*|")
            ax.axvline(0.0, color=AXIS, linewidth=1.0)
            name = FINGERS[r] + " (norm over 4 joints, clamped joints left out)" if r < 5 else "arm targets (largest joint)"
            ax.set_title(f"{name}, demos K = {k:g}" if r == 0 or r == 5 else name)
            ax.set_ylabel("mrad")
    for ax in axes[-1]:
        ax.set_xlabel("time relative to the demo's first contact [s]")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("K-swap on the demo states: change of the predicted command when only K is swapped, vs the "
                 "admittance law\n(median and interquartile band over the demos; before contact the law gives 0)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, out_dir, "kswap_time")


def plot_hold(per_demo, ks, out_dir, n_hold):
    """Per finger, one dot per demo: hold medians of |D*| (x) and of D_pred's component along D* (y); y = x is the law."""
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.4))
    for f, ax in enumerate(axes):
        top = 0.0
        for k in ks:
            xs, ys, ys_raw = [], [], []
            for d in (d for d in per_demo if d["K"] == k):
                m = d["forces"][:, f] > CONTACT_N
                m[:-n_hold(d)] = False
                if m.sum() < 10:
                    continue
                s = d["star"][f][m]
                ns = np.linalg.norm(s, axis=-1)
                ok = ns > 1e-6
                if not ok.any():
                    continue
                along = lambda p: np.median((p[m][ok] * s[ok]).sum(-1) / ns[ok])  # noqa: E731
                xs.append(np.median(ns[ok]))
                ys.append(along(d["pred_adj"][f]))
                ys_raw.append(along(d["pred"][f]))
            if xs:
                x = 1000 * np.array(xs)
                ax.scatter(x, 1000 * np.array(ys), s=22, color=k_color(k), alpha=0.75, linewidths=0,
                           label=f"demos K = {k:g} (n = {len(xs)}), offset before contact removed")
                ax.scatter(x, 1000 * np.array(ys_raw), s=22, facecolors="none", edgecolors=k_color(k), linewidths=0.8,
                           alpha=0.6, label=f"demos K = {k:g}, raw")
                top = max(top, float(x.max()))
        top = max(top, 1.0) * 1.05
        ax.plot([0, top], [0, top], color=INK_2, linestyle="--", linewidth=1.0, label="admittance law (y = x)")
        ax.axhline(0.0, color=AXIS, linewidth=1.0)
        ax.set_xlim(0, top)
        ax.set_title(FINGERS[f])
        ax.set_xlabel("law's change |D*| [mrad]")
        if f == 0:
            ax.set_ylabel("policy's change along D* [mrad]")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Hold (last {args.hold_s:g} s, finger in contact): one dot per demo (median over the hold); on the "
                 "dashed line the policy changes its command with K exactly as the admittance law, at 0 it ignores K\n"
                 "filled: with the policy's K offset from just before contact removed; hollow: raw",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, out_dir, "kswap_hold")


if __name__ == "__main__":
    main()
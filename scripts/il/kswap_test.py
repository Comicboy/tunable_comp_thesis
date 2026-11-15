"""Offline K-swap test: does the policy change its finger commands with K the way the admittance law does?

Runs in the "lerobot" conda env (needs the trained checkpoint, torch, h5py); no simulation. It only
computes: the figures are drawn by scripts/plots/plot_il.py --kswap <out folder> (fig6, and with
--extra kswap_time / kswap_hold) from the two files this script writes.
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
        --demos logs/il_demos/lift_K150.hdf5 logs/il_demos/lift_K500.hdf5 [--out DIR]

    # the same on the validation demos (never trained on; results go to <run>_val):
    python scripts/il/kswap_test.py --policy ... \
        --demos logs/il_demos_val/lift_K150.hdf5 logs/il_demos_val/lift_K500.hdf5

Output (--out, default logs/il_eval/kswap/<run>/, or <run>_val/ when the demo files are validation demos,
file attribute split = val): summary.json (the printed tables) and kswap.h5 (per demo and step: D_pred
and D* per finger, D_pred with the pre-contact offset removed, the arm's D_pred, the fingertip forces
and the first contact; and the absolute commands, see below; for the plots). What it shows:

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

The action used for D_pred and the tables is the first action of the predicted chunk at that state
(what the policy outputs with no history; in the rollouts temporal ensembling also averaged older
predictions).

Absolute commands (for comparing the policy's commands with the admittance's, state by state): kswap.h5
also holds, per demo and step, the policy's full action at both K in two versions,

    act_first (T, 2, 26)  first action of the chunk predicted at that state, [K_lo, K_hi]
    act_ens   (T, 2, 26)  what temporal ensembling would execute at that step if the policy had run along
                          this demo with that K from its first step: the weighted mean of the predictions
                          for step t made at the states t, t-1, ..., t-H+1 (H = chunk size), weight
                          exp(-m i) with i = 0 for the oldest (as LeRobot's ACTTemporalEnsembler,
                          m = --temporal_ensemble, 0.01 in the rollouts)

next to the demo's recorded action act_rec (T, 26) (the admittance's q_cmd for the hand), the operator's
q_ref (T, 20), the admittance offset dq (T, 20), the measured hand angles q_hand (T, 20) and the clamped
mask (T, 20). Joint orders: attrs arm_joints, hand_joints. The fit table is printed for both versions.
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np  # noqa: E402
import torch  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies import get_policy_class, make_pre_post_processors  # noqa: E402

# project policy types (LeRobot plugins in scripts/il/lerobot_plugins): importing registers them, so
# their checkpoints load like ACT ones (e.g. act_ktoken: ACT with the stiffness as its own token)
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent / "lerobot_plugins"))  # noqa: E402
import act_ktoken  # noqa: F401,E402

parser = argparse.ArgumentParser(description="Offline K-swap test of a q_cmd policy on the demo states.")
parser.add_argument("--policy", required=True, help="checkpoint folder (…/checkpoints/<step>/pretrained_model)")
parser.add_argument("--demos", nargs=2, required=True, help="the two demo files (one per K)")
parser.add_argument("--out", default=None, help="folder for plots and the summary JSON")
parser.add_argument("--device", default="cuda")
parser.add_argument("--batch", type=int, default=1024, help="states per forward pass")
parser.add_argument("--hold_s", type=float, default=3.0, help="hold window: the last hold_s of each demo [s] "
                    "(3 s = the success hold, as in plot_il.py)")
parser.add_argument("--boot", type=int, default=2000, help="bootstrap resamples (over demos) for the gain intervals")
parser.add_argument("--task", default="Lift the cube 15 cm above the table and hold it there")
parser.add_argument("--temporal_ensemble", type=float, default=0.01,
                    help="coefficient m of the ensembled commands act_ens (the rollouts used 0.01)")
args = parser.parse_args()

STATE, ENV_STATE = "observation.state", "observation.environment_state"
ENV_TERMS = ("cube_position",)  # -> environment_state, as in convert_to_lerobot.py
FINGERS = ["thumb", "index", "middle", "ring", "little"]
CONTACT_N = 0.5   # [N] smoothed fingertip force: the finger is in contact
TOUCH_N = 1e-3    # [N] summed smoothed force: first contact of the demo
PRE_N = 10        # samples (0.2 s) before the first contact: the policy's K offset there is removed for g_adj
CLAMP_TOL = 1e-4  # [rad] |q_cmd - (q_ref + dq)| above this: q_cmd was clamped to a joint limit
MAX_OFFSET = 0.03  # [m] the admittance's clamp on |e| per fingertip (AdmittanceControllerCfg.max_offset)

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
    """The whole predicted chunk for every state (no action queue, no ensembling), in joint units: (N, H, 26).

    The post-processor (un-normalisation) is applied to the chunk flattened to (B * H, 26), the shape it
    gets in the rollouts one action at a time."""
    out = []
    with torch.inference_mode():
        for i in range(0, len(states), args.batch):
            batch = {STATE: torch.from_numpy(states[i:i + args.batch]),
                     ENV_STATE: torch.from_numpy(envs[i:i + args.batch]),
                     "task": [args.task] * len(states[i:i + args.batch]), "robot_type": "ur5_wuji"}
            chunk = policy.predict_action_chunk(pre(batch))  # (B, H, 26), normalised
            b, h, n = chunk.shape
            out.append(post(chunk.reshape(b * h, n)).detach().cpu().numpy().reshape(b, h, n))
    return np.concatenate(out).astype(np.float32)


def ensemble(chunks: np.ndarray, m: float) -> np.ndarray:
    """Temporal ensembling along one trajectory, as LeRobot's ACTTemporalEnsembler: (T, H, 26) -> (T, 26).

    The action executed at step t averages the predictions for step t made at the states t - k
    (k = 0 .. min(t, H - 1); element k of that chunk), the oldest weighted most: arrival index i = 0 for
    the oldest, weight exp(-m i). Averaging in joint units equals averaging in normalised units (the
    un-normalisation is affine per dimension and the weights are normalised)."""
    T, H, _ = chunks.shape
    w = np.exp(-m * np.arange(H))
    out = np.empty((T, chunks.shape[-1]), np.float64)
    for t in range(T):
        k = np.arange(min(t + 1, H))  # age of the prediction in steps
        wk = w[len(k) - 1 - k]        # arrival index: the oldest (largest k) gets w[0]
        out[t] = (wk[:, None] * chunks[t - k, k]).sum(0) / wk.sum()
    return out


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
                "arm_joints": json.loads(a["arm_joints"]), "hand_joints": hand,
                "split": (lambda v: v.decode() if isinstance(v, bytes) else str(v))(a.get("split", "train")),
                "path": path,
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
    splits = {s["split"] for s in sets}
    if len(splits) > 1:
        raise SystemExit(f"the two demo files are from different splits: {sorted(splits)}")
    split = splits.pop()
    out_dir = args.out or os.path.join("logs", "il_eval", "kswap", run + ("_val" if split == "val" else ""))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[kswap] policy {args.policy} ({cfg.type}); swapping log10 K between {k_lo:g} and {k_hi:g}; "
          f"{'validation' if split == 'val' else 'training'} demos; ensembled commands with m = {args.temporal_ensemble:g}")

    per_demo = []  # one dict per demo with per-step arrays
    for s in sets:
        state_idx, env_idx, k_pos, force_sl = split_state(s["terms"])
        n_arm, fidx = s["n_arm"], s["finger_idx"]
        for d in s["demos"]:
            obs, T = d["obs"], len(d["obs"])
            st, env = obs[:, state_idx].copy(), obs[:, env_idx].copy()
            st_lo, st_hi = st.copy(), st.copy()
            st_lo[:, k_pos], st_hi[:, k_pos] = np.log10(k_lo), np.log10(k_hi)
            chunks = predict(policy, pre, post, np.concatenate([st_lo, st_hi]), np.concatenate([env, env]))
            a_lo, a_hi = chunks[:T, 0].astype(np.float64), chunks[T:, 0].astype(np.float64)  # first actions
            e_lo, e_hi = ensemble(chunks[:T], args.temporal_ensemble), ensemble(chunks[T:], args.temporal_ensemble)
            del chunks
            d_pred = a_lo - a_hi  # (T, 26)
            q_cmd = d["act"][:, n_arm:]
            clamped = np.abs(q_cmd - (d["q_ref"] + d["dq"])) > CLAMP_TOL  # (T, 20)
            forces = np.linalg.norm(obs[:, force_sl].reshape(T, 5, 3), axis=-1)  # (T, 5) smoothed, as the admittance sees it
            d_star = d["dq"] * np.repeat(law_factor(forces, s["K"], k_lo, k_hi), 4, axis=1)[:, finger_cols(fidx)]
            hand_pred = np.where(clamped, 0.0, d_pred[:, n_arm:])
            hand_star = np.where(clamped, 0.0, d_star)
            touch = np.nonzero((forces.sum(-1) > TOUCH_N) | (np.abs(d["dq"]).max(-1) > 0))[0]
            a_own = a_lo if s["K"] == k_lo else a_hi
            e_own = e_lo if s["K"] == k_lo else e_hi
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
                "fit_arm_ens": np.abs(e_own[:, :n_arm] - d["act"][:, :n_arm]).mean(-1),
                "fit_hand_ens": [np.linalg.norm(e_own[:, n_arm:][:, ix] - q_cmd[:, ix], axis=-1) for ix in fidx],
                # absolute commands (index 1: [K_lo, K_hi]) and the recorded quantities they are compared with
                "act_first": np.stack([a_lo, a_hi], axis=1), "act_ens": np.stack([e_lo, e_hi], axis=1),
                "act_rec": d["act"], "q_ref": d["q_ref"], "dq": d["dq"],
                "q_hand": obs[:, [i for name, s0, e0 in s["terms"] if name == "hand_joint_pos" for i in range(s0, e0)]],
            })
        print(f"[kswap] {len(s['demos'])} demos at K = {s['K']:g} done")
    arm_names = [n.replace("_joint", "") for n in sets[0]["arm_joints"]]
    summary = {"policy": args.policy, "demos": args.demos, "split": split, "K": [k_lo, k_hi],
               "temporal_ensemble": args.temporal_ensemble}

    # --- fit ------------------------------------------------------------------------------------------
    print("\n=== fit at the demo's own K: median |prediction - recorded action| [mrad] (arm: mean over joints; "
          "fingers: norm over 4 joints); first action of the chunk / ensembled")
    print(f"{'':<14}{'arm':>14}" + "".join(f"{f:>14}" for f in FINGERS))
    summary["fit_mrad_first_ens"] = {}
    for k in (k_lo, k_hi):
        ds = [d for d in per_demo if d["K"] == k]
        cells, row = [], {}
        for key, name in (("fit_arm", "arm"), *((f, FINGERS[f]) for f in range(5))):
            if key == "fit_arm":
                v1, v2 = (np.median(np.concatenate([d[x] for d in ds])) for x in ("fit_arm", "fit_arm_ens"))
            else:
                v1, v2 = (np.median(np.concatenate([d[x][key] for d in ds])) for x in ("fit_hand", "fit_hand_ens"))
            cells.append(f"{1000 * v1:.1f} / {1000 * v2:.1f}")
            row[name] = [1000 * float(v1), 1000 * float(v2)]
        print(f"{f'demos K{k:g}':<14}" + "".join(f"{c:>14}" for c in cells))
        summary["fit_mrad_first_ens"][f"demos K{k:g}"] = row

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
    save_arrays(per_demo, (k_lo, k_hi), os.path.join(out_dir, "kswap.h5"),
                {"split": split, "temporal_ensemble": args.temporal_ensemble, "chunk_size": int(cfg.chunk_size),
                 "arm_joints": json.dumps(sets[0]["arm_joints"]), "hand_joints": json.dumps(sets[0]["hand_joints"]),
                 "demo_files": json.dumps([s["path"] for s in sets])})
    print(f"\n[kswap] summary -> {os.path.join(out_dir, 'summary.json')}, arrays -> {os.path.join(out_dir, 'kswap.h5')}")


def save_arrays(per_demo, ks, path, attrs):
    """Per-demo arrays for the plots (plot_il.py): /demos/K<K>_<demo>/ with
    pred, pred_adj, star (T, 5, 4) [rad] (clamped joints set to 0), arm_pred (T, 6), forces (T, 5) [N];
    the absolute commands act_first, act_ens (T, 2, 26) [rad] (index 1: K_lo, K_hi), act_rec (T, 26),
    q_ref, dq, q_hand (T, 20), clamped (T, 20) (hand joints in the order of the attr hand_joints);
    attrs K, first (first contact sample), dt. File attrs: K_lo, K_hi, split, temporal_ensemble,
    chunk_size, arm_joints, hand_joints, demo_files, ..."""
    with h5py.File(path, "w") as f:
        f.attrs.update({"K_lo": ks[0], "K_hi": ks[1], "contact_n": CONTACT_N, "pre_n": PRE_N, "max_offset": MAX_OFFSET,
                        **attrs})
        g = f.create_group("demos")
        for d in per_demo:
            h = g.create_group(f"K{d['K']:g}_{d['name']}")
            for key in ("pred", "pred_adj", "star"):
                h.create_dataset(key, data=np.stack(d[key], axis=1).astype(np.float32))  # (T, 5, 4)
            h.create_dataset("arm_pred", data=d["arm_pred"].astype(np.float32))
            for key in ("act_first", "act_ens", "act_rec", "q_ref", "dq", "q_hand"):
                h.create_dataset(key, data=np.asarray(d[key], np.float32))
            h.create_dataset("clamped", data=d["clamped"])
            h.create_dataset("forces", data=d["forces"].astype(np.float32))
            h.attrs.update({"K": d["K"], "first": d["first"], "dt": d["dt"]})


if __name__ == "__main__":
    main()
"""How the policy responds where the perturbation demos carry their extra information: in the perturbations.

The perturbation demos (teleop_lift.py --perturb) add K-dependent information only while a perturbation
changes the fingertip force at a fixed operator intention: there the admittance's command yields by
J+ (delta f) / K, while in a steady grip the command also follows from posture and force alone (drive
relation), without K. Over all contact samples, kswap_test.py's gain g is dominated by the steady grip;
this script looks at the perturbation windows separately. It needs no policy and no simulator: it reads
kswap_test.py's output (kswap.h5: the policy's predictions at both K for every demo state) and the
demo files named in it (push_force, q_ref, dq, the state).

Two measures per finger (joint space, joints whose recorded command was clamped to a joint limit left out):

  1. K swap, split by window: the gain g = sum(D_pred . D*) / sum(|D*|^2) of kswap_test.py (D_pred =
     a(K150) - a(K500) at the same state, D* = the admittance law's change), computed separately over
       perturbation: from the onset of a perturbation to --settle_s after its release,
       steady:       the other contact samples once the cube is lifted (--lift_m above its start height),
     with 95 % bootstrap intervals over the demos. If the perturbations taught the policy the law, g is
     higher in the perturbation windows than in the steady grip.

  2. Response to the perturbation at the demo's own K: from the step before the onset (b) to the last
     step with the force (e), the change of the recorded command (d_rec = q_cmd[e] - q_cmd[b]) and of the
     policy's first predicted action at the same states (d_pol). d_rec = d_dq + d_qref: the admittance's
     response (d_dq = dq[e] - dq[b], the K-dependent part) plus whatever the operator's reference did.
     Printed: medians of |d_dq| and |d_qref| [mrad] (how large the K-informative part is next to the
     operator's own motion), and the least-squares slopes of d_pol on d_rec and on d_dq (1 = the policy
     reproduces the recorded change; the policy sees the state at e, so a policy reading the drive
     relation off the state also follows part of it — the comparison with the steady grip in 1. is the
     cleaner test).

    python scripts/il/perturb_response.py logs/il_eval/kswap/act_perturb_v1_val logs/il_eval/kswap/act_perturb_v1_seed2000_val

Prints one table per kswap folder; nothing is written. Needs h5py and numpy.
"""

import argparse
import json
import os

import h5py
import numpy as np

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("kswap", nargs="+", help="kswap_test.py output folder(s) with kswap.h5 (perturbation demos)")
parser.add_argument("--settle_s", type=float, default=0.4, help="the perturbation window ends this long after the release [s]")
parser.add_argument("--lift_m", type=float, default=0.02, help="steady window: cube at least this far above its start [m]")
parser.add_argument("--boot", type=int, default=2000, help="bootstrap resamples over the demos")
args = parser.parse_args()

FINGERS = ("thumb", "index", "middle", "ring", "little")
CONTACT_N = 0.5  # [N] as kswap_test.py


def runs_of(push: np.ndarray) -> list[tuple[int, int]]:
    """(first row, first row after) of every run of rows with a force on the cube."""
    on = np.linalg.norm(push, axis=-1) > 1e-6
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def gain_ci(nums: np.ndarray, dens: np.ndarray, rng) -> list[float]:
    """g = sum num / sum den over demos, with a bootstrap over demos."""
    if len(nums) < 3 or dens.sum() <= 0:
        return [float("nan")] * 3
    g = nums.sum() / dens.sum()
    b = []
    for _ in range(args.boot):
        i = rng.integers(0, len(nums), len(nums))
        if dens[i].sum() > 0:
            b.append(nums[i].sum() / dens[i].sum())
    return [float(g), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def slope(x: np.ndarray, y: np.ndarray) -> float:
    d = float((x * x).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")


def analyse(folder: str):
    path = os.path.join(folder, "kswap.h5")
    with h5py.File(path, "r") as f:
        k_lo, k_hi = float(f.attrs["K_lo"]), float(f.attrs["K_hi"])
        hand = json.loads(f.attrs["hand_joints"]) if isinstance(f.attrs["hand_joints"], str) else list(f.attrs["hand_joints"])
        n_arm = len(json.loads(f.attrs["arm_joints"])) if isinstance(f.attrs["arm_joints"], str) else len(f.attrs["arm_joints"])
        demo_files = json.loads(f.attrs["demo_files"])
        ks = {}
        for name in f["demos"]:
            g = f["demos"][name]
            ks[name] = {k: g[k][()] for k in ("pred", "star", "act_first", "act_rec", "clamped", "forces")}
            ks[name]["K"] = float(g.attrs["K"])
            ks[name]["dt"] = float(g.attrs.get("dt", 0.02))
    cols = [[i for i, n in enumerate(hand) if f"finger{f + 1}_" in n] for f in range(5)]  # hand columns per finger
    demos = {}  # (K, demo name) -> push_force, q_ref, dq, cube height
    for p in demo_files:
        with h5py.File(p, "r") as f:
            k = float(f.attrs["stiffness"])
            sl = {n: slice(s, e) for n, s, e in json.loads(f.attrs["obs_terms"])}
            for name in f["data"]:
                g = f["data"][name]
                if "push_force" not in g:
                    raise SystemExit(f"{p}: {name} has no push_force (not a perturbation demo file)")
                z = g["obs"][()][:, sl["cube_position"]][:, 2]
                demos[(k, name)] = {"push": g["push_force"][()], "q_ref": g["q_ref"][()], "dq": g["dq"][()], "lift": z - z[0]}

    acc = {w: [[] for _ in range(5)] for w in ("perturbation", "steady")}  # per finger: (num, den) per demo
    resp = [{"d_rec": [], "d_dq": [], "d_qref": [], "d_pol": []} for _ in range(5)]
    n_pert = 0
    for key, d in ks.items():
        k, name = d["K"], key.split("_", 1)[1]
        dm = demos[(k, name)]
        T = len(d["forces"])
        settle = int(round(args.settle_s / d["dt"]))
        runs = runs_of(dm["push"])
        n_pert += len(runs)
        pert = np.zeros(T, bool)
        for s, e in runs:
            pert[s:min(e + settle, T)] = True
        lifted = dm["lift"] >= args.lift_m
        own = 0 if np.isclose(k, k_lo) else 1
        for fi in range(5):
            contact = d["forces"][:, fi] > CONTACT_N
            for w, m in (("perturbation", contact & pert), ("steady", contact & lifted & ~pert)):
                p, s_ = d["pred"][:, fi][m], d["star"][:, fi][m]
                acc[w][fi].append(((p * s_).sum(), (s_ * s_).sum()))
            c = cols[fi]
            for s, e in runs:
                b, en = s - 1, min(e, T - 1)
                if b < 0 or not (contact[b] or contact[en]):
                    continue
                ok = ~(d["clamped"][b, c] | d["clamped"][en, c])  # joints usable in this perturbation
                if not ok.any():
                    continue
                cc = np.array(c)[ok]
                resp[fi]["d_rec"].append(d["act_rec"][en, n_arm + cc] - d["act_rec"][b, n_arm + cc])
                resp[fi]["d_pol"].append(d["act_first"][en, own, n_arm + cc] - d["act_first"][b, own, n_arm + cc])
                resp[fi]["d_dq"].append(dm["dq"][en, cc] - dm["dq"][b, cc])
                resp[fi]["d_qref"].append(dm["q_ref"][en, cc] - dm["q_ref"][b, cc])

    rng = np.random.default_rng(0)
    print(f"\n=== {folder}: {len(ks)} demos, {n_pert} perturbations; K swap {k_lo:g} <-> {k_hi:g}, joint space")
    print(f"{'finger':<8}{'g perturbation [95 %]':>26}{'g steady [95 %]':>26}{'perts':>7}{'|d_dq|':>8}{'|d_qref|':>9}"
          f"{'d_pol~d_rec':>13}{'d_pol~d_dq':>12}")
    for fi in range(5):
        cells = []
        for w in ("perturbation", "steady"):
            a = np.array(acc[w][fi]) if acc[w][fi] else np.zeros((0, 2))
            keep = a[:, 1] > 0 if len(a) else np.zeros(0, bool)
            g = gain_ci(a[keep, 0], a[keep, 1], rng)
            cells.append("n/a" if np.isnan(g[0]) else f"{g[0]:+.2f} [{g[1]:+.2f}, {g[2]:+.2f}]")
        r = resp[fi]
        n = len(r["d_rec"])
        if n:
            rec, pol = np.concatenate(r["d_rec"]), np.concatenate(r["d_pol"])
            ddq, dqr = np.concatenate(r["d_dq"]), np.concatenate(r["d_qref"])
            per = lambda v: np.median([np.linalg.norm(x) for x in v])  # noqa: E731 (norm over the finger's joints)
            tail = (f"{n:>7}{1000 * per(r['d_dq']):>8.1f}{1000 * per(r['d_qref']):>9.1f}"
                    f"{slope(rec, pol):>13.2f}{slope(ddq, pol):>12.2f}")
        else:
            tail = f"{0:>7}"
        print(f"{FINGERS[fi]:<8}{cells[0]:>26}{cells[1]:>26}{tail}")
    print("  |d_dq|, |d_qref|: median change over a perturbation of the admittance's offset and of the operator's "
          "reference [mrad, norm over the finger's joints]; d_pol~d_rec / d_pol~d_dq: slope of the policy's change "
          "on the recorded change / on the admittance's part of it")


for folder in args.kswap:
    analyse(folder)
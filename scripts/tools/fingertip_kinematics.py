"""Fingertip Jacobians of recorded states (demos, policy rollouts, scripted lift), for scripts/plots/plot_il.py.

The demo and rollout files hold joint angles, but the admittance law and the force sensor work at the
fingertips. To put both into one picture, the plots need each finger's tip Jacobian J_i (3 x 4: how the
tip moves per joint) at every recorded posture:

    fingertip side:  d_ref = J (q_ref - q),  d_cmd = J (q_cmd - q),  e = J dq       [m]
    joint side:      tau_ext = J^T f   (the torque the contact force puts on the finger's joints) [Nm]

This script puts every recorded posture into the simulation (kinematics only: the joint angles are
written and the link poses updated, no physics step, so nothing moves and nothing collides) and computes
J by central differences of a point's position on the tip link.

Which point: the admittance maps its fingertip offset with PhysX's Jacobian (dq = J_px^+ e) and logs
J_px (q_ref - q), so the plots must use the same point, or e = J dq and d_ref are off by the ratio of
the lever arms. PhysX's Jacobian refers to the tip link's centre of mass (checked 2026-11-03: finite
differences at the COM agree with it to 0.00 %, at the link frame origin, at the distal joint, they differ
by 12 % on fingers 2-5 and 6 % on the thumb; COM and PhysX reproduce the logged offset to ~0.2 mm). All
three are still computed and compared on every run; --point selects the one that is saved (default COM).

    python scripts/tools/fingertip_kinematics.py --headless \\
        logs/il_demos/lift_K150.hdf5 logs/il_demos/lift_K500.hdf5 \\
        logs/il_eval/act_qcmd_v1_te0.01_K150_seed0.hdf5 logs/il_eval/act_qcmd_v1_te0.01_K500_seed0.hdf5 \\
        logs/scripted_lift/dynamic_2026-11-03_16-33-51.npz logs/scripted_lift/static_2026-11-03_16-35-09.npz

Inputs: demo and rollout files (.hdf5: /data/<group>/obs, joint angles taken from the state) and
scripted-lift logs (.npz from scripts/experiments/scripted_lift.py: q_arm, q, one group per env,
env_000 ...). The kinematics are those of the IL env's robot, the same UR5 + Wuji hand as in the
scripted lift; the arm angles come from each file, so the different arm start poses do not matter.

Output next to each input, <name>_kin.h5 (an existing one is only replaced with --overwrite):

    /data/<demo_NNN, ep_NNN or env_NNN>/J  (T, 5, 3, 4) float32 [m/rad]  tip Jacobian of finger 1..5
                                   (thumb first) with respect to its joints 1..4, robot base frame, at
                                   the point chosen with --point (attrs["point"]); sample t is at the
                                   joint angles of obs[t] (demos, rollouts: the state the step started
                                   from) or of q[t] (scripted lift: logged after the step)
    / attrs: source, task, eps, hand_cols (5 x 4: the column of finger i, joint j in the hand joint order
             of the source file, i.e. of the state and the actions), check (JSON), created, and the drive
             properties of the hand joints in that order: kp [Nm/rad], kd [Nms/rad], effort_limit [Nm],
             pos_limits (20, 2) [rad] (the soft limits the commands are clipped to), as the simulation uses them

Checks, printed and stored: (1) the three Jacobians against each other (median relative difference
per finger); (2) during the recording the admittance logged J (q_ref - q) per fingertip (tip_ref_offset
in the demo files and the scripted-lift logs). The script recomputes it with each of the three
Jacobians and prints the error per finger. With the controller's point the error should be a few per cent of the offset (the logged value
comes from inside the env step, the angles used here from its end: the finger moved by one physics step).

Physics device: CPU by default (like the IL env); the result does not depend on it.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fingertip Jacobians of recorded demo and rollout states.")
parser.add_argument("files", nargs="+",
                    help="demo files (logs/il_demos/*.hdf5), rollout trajectory files, scripted-lift logs (.npz)")
parser.add_argument("--num_envs", type=int, default=256, help="postures evaluated at once (one per env)")
parser.add_argument("--eps", type=float, default=0.01, help="joint step of the central differences [rad]")
parser.add_argument("--point", choices=["com", "origin", "physx"], default="com",
                    help="which Jacobian is saved: finite differences at the tip link's centre of mass (default), "
                         "at its frame origin, or PhysX's own")
parser.add_argument("--overwrite", action="store_true", help="replace an existing <name>_kin.h5")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import json
import os
from datetime import datetime

import gymnasium as gym
import h5py
import numpy as np
import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse
from isaaclab_tasks.utils import parse_env_cfg

import tunable_comp_thesis.tasks  # noqa: F401  (registers the gym tasks)
from tunable_comp_thesis.tasks.manager_based.lift_compliant.config.ur5_wuji.il_env_cfg import ARM_JOINTS

TASK = "Isaac-Lift-Cube-UR5Wuji-IL-v0"
FINGERS = ["thumb", "index", "middle", "ring", "little"]
POINTS = ["origin", "com", "physx"]
CONTACT_M = 0.002  # [m] logged reference offset above this: the fingertip is pressing (for the check)


# ---------------------------------------------------------------------------------------------------
# simulation: one robot per env, used as a kinematics calculator
# ---------------------------------------------------------------------------------------------------
class Kinematics:
    def __init__(self, num_envs: int):
        env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=num_envs)
        # all robots at the origin: nothing is simulated (envs do not collide with each other anyway),
        # and small coordinates keep the float32 tip positions precise for the finite differences
        env_cfg.scene.env_spacing = 0.0
        env_cfg.terminations.success.params["show_plane"] = False
        self.env = gym.make(TASK, cfg=env_cfg)
        self.env.reset()
        uw = self.env.unwrapped
        self.sim, self.robot = uw.sim, uw.scene["robot"]
        self.view = self.robot.root_physx_view
        self.n, self.dev = num_envs, uw.device
        self.arm_ids = self.robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
        self.tip_ids = [self.robot.find_bodies(f"right_finger{f}_tip_link")[0][0] for f in range(1, 6)]
        self.q_default = self.robot.data.default_joint_pos.clone()  # (N, dofs): joints not in the files
        self.root_p = self.view.get_root_transforms()[:, :3].clone()  # fixed base: does not move
        self.root_q = self.robot.data.root_quat_w.clone()  # (N, 4) w, x, y, z
        # centre of mass of each tip link in the link's own frame (position part of (N, links, 7))
        self.com_b = self.view.get_coms()[:, self.tip_ids, :3].clone().to(self.dev)  # (N, 5, 3)

    def set_hand(self, hand_names: list[str]):
        """Joint order of a file: the articulation ids of its hand columns and of each finger's 4 joints,
        and the drive properties of these joints in that order."""
        self.hand_ids = self.robot.find_joints(hand_names, preserve_order=True)[0]
        d = self.robot.data
        self.drive = {"kp": d.joint_stiffness[0, self.hand_ids].cpu().numpy(),
                      "kd": d.joint_damping[0, self.hand_ids].cpu().numpy(),
                      "effort_limit": d.joint_effort_limits[0, self.hand_ids].cpu().numpy(),
                      "pos_limits": d.soft_joint_pos_limits[0, self.hand_ids].cpu().numpy()}
        self.hand_cols = np.array([[hand_names.index(f"right_finger{f}_joint{j}") for j in range(1, 5)]
                                   for f in range(1, 6)])  # (5, 4)
        self.finger_ids = torch.tensor(np.asarray(self.hand_ids)[self.hand_cols], device=self.dev)  # (5, 4)

    def _write(self, q: torch.Tensor) -> torch.Tensor:
        """Write all joint angles (N, dofs), update the link poses; returns the angles PhysX holds now."""
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        self.sim.forward()  # link poses from the joint angles, no physics step
        return self.view.get_dof_positions().clone()

    def _tips(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Tip link origin and tip link centre of mass relative to the robot base, base frame, (N, 5, 3) each."""
        lt = self.view.get_link_transforms()[:, self.tip_ids]  # (N, 5, 7): position, quaternion x, y, z, w
        q_link = torch.cat([lt[..., 6:7], lt[..., 3:6]], dim=-1).reshape(-1, 4)  # -> w, x, y, z
        origin = lt[..., :3]
        com = origin + quat_apply(q_link, self.com_b.reshape(-1, 3)).reshape(self.n, 5, 3)
        rq = self.root_q[:, None].expand(-1, 5, -1).reshape(-1, 4)
        to_base = lambda p: quat_apply_inverse(rq, (p - self.root_p[:, None]).reshape(-1, 3)).reshape(self.n, 5, 3)
        return to_base(origin), to_base(com)

    def jacobians(self, q_arm: torch.Tensor, q_hand: torch.Tensor) -> dict:
        """{"origin", "com", "physx"}: J (N, 5, 3, 4) at the given arm (N, 6) and hand (N, 20) angles.

        Finite differences: joint j of all five fingers is stepped together (every tip depends only on
        its own finger's joints and the arm, which stays put). The step actually taken is read back, so
        a joint that PhysX holds at its limit is differentiated over the step it really made."""
        q = self.q_default.clone()
        q[:, self.arm_ids] = q_arm
        q[:, self.hand_ids] = q_hand
        J = {p: torch.zeros(self.n, 5, 3, 4, device=self.dev) for p in POINTS}
        for j in range(4):
            ids = self.finger_ids[:, j]  # (5,) the j-th joint of every finger
            qp, qm = q.clone(), q.clone()
            qp[:, ids] += args_cli.eps
            qm[:, ids] -= args_cli.eps
            q_p = self._write(qp)[:, ids]
            o_p, c_p = self._tips()
            q_m = self._write(qm)[:, ids]
            o_m, c_m = self._tips()
            step = (q_p - q_m).clamp(min=1e-6)[..., None]  # (N, 5, 1)
            J["origin"][..., j] = (o_p - o_m) / step
            J["com"][..., j] = (c_p - c_m) / step
        # PhysX's analytic Jacobian at q (the one the admittance uses), world frame -> base frame
        self._write(q)
        jac = self.view.get_jacobians()  # (N, links - 1, 6, dofs): fixed base, so no row for the root
        rows = [t - 1 for t in self.tip_ids]
        J_px = torch.stack([jac[:, rows[i], :3][:, :, self.finger_ids[i]] for i in range(5)], dim=1)  # (N, 5, 3, 4)
        rq = self.root_q[:, None, None].expand(-1, 5, 4, -1).reshape(-1, 4)
        J_px = quat_apply_inverse(rq, J_px.transpose(2, 3).reshape(-1, 3)).reshape(self.n, 5, 4, 3).transpose(2, 3)
        J["physx"] = J_px.to(self.dev)
        return J


def rel_diff(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """|A - B| / |B| per sample and finger (Frobenius norms of the 3 x 4 blocks): (S, 5)."""
    return np.linalg.norm(A - B, axis=(2, 3)) / np.maximum(np.linalg.norm(B, axis=(2, 3)), 1e-9)


# ---------------------------------------------------------------------------------------------------
# one file
# ---------------------------------------------------------------------------------------------------
def load(path: str) -> tuple[list[str], list[str], dict]:
    """Hand joint names, group names and per group (q_arm (T, 6), q_hand (T, 20), check), where check is
    None or (logged reference offset (T, 5, 3) world frame, q_ref (T, 20), lag): the logged offset of
    sample t is compared with J[t + lag] (q_ref[t] - q[t + lag]), i.e. with the angles at the end of step t."""
    if path.endswith(".npz"):  # scripted lift: arrays (T, N, ...), everything logged after the step
        d = np.load(path, allow_pickle=True)
        hand_names = [str(n) for n in d["joint_names"]]
        groups = [f"env_{i:03d}" for i in range(d["q"].shape[1])]
        data = {g: (d["q_arm"][:, i], d["q"][:, i], (d["tip_ref_offset"][:, i], d["q_ref"][:, i], 0))
                for i, g in enumerate(groups)}
        return hand_names, groups, data
    with h5py.File(path, "r") as f:  # demos and rollouts: the state obs[t] is from before step t
        a = f.attrs
        sl = {n: slice(s, e) for n, s, e in json.loads(a["obs_terms"])}
        hand_names = json.loads(a["hand_joints"])
        groups = sorted(f["data"])
        data = {}
        for g in groups:
            G = f["data"][g]
            obs = G["obs"][()]
            chk = (G["tip_ref_offset"][()], G["q_ref"][()], 1) if "tip_ref_offset" in G else None
            data[g] = (obs[:, sl["arm_joint_pos"]], obs[:, sl["hand_joint_pos"]], chk)
    return hand_names, groups, data


def process(kin: Kinematics, path: str):
    out = os.path.splitext(path)[0] + "_kin.h5"
    if os.path.exists(out) and not args_cli.overwrite:
        print(f"[kin] {out} exists, skipped (--overwrite replaces it)")
        return
    hand_names, groups, data = load(path)
    kin.set_hand(hand_names)
    if not getattr(kin, "_printed", False):
        print_drives(kin, hand_names)
        kin._printed = True

    # every posture of the file in one list, evaluated N at a time
    q_arm = np.concatenate([data[g][0] for g in groups]).astype(np.float32)
    q_hand = np.concatenate([data[g][1] for g in groups]).astype(np.float32)
    total = len(q_arm)
    J_all = {p: np.zeros((total, 5, 3, 4), np.float32) for p in POINTS}
    for start in range(0, total, kin.n):
        idx = np.arange(start, start + kin.n) % total  # the last batch is padded with the first samples
        J = kin.jacobians(torch.tensor(q_arm[idx], device=kin.dev), torch.tensor(q_hand[idx], device=kin.dev))
        m = min(kin.n, total - start)
        for p in POINTS:
            J_all[p][start:start + m] = J[p][:m].cpu().numpy()
    print(f"[kin] {path}: {len(groups)} episodes / envs, {total} states")
    check = {"jacobians_rel_diff_median": {}, "logged_offset": {}}
    print("      the three Jacobians against each other, median relative difference per finger:")
    for a_, b_ in (("origin", "physx"), ("com", "physx"), ("origin", "com")):
        r = np.median(rel_diff(J_all[a_], J_all[b_]), axis=0)
        check["jacobians_rel_diff_median"][f"{a_} vs {b_}"] = {FINGERS[i]: float(r[i]) for i in range(5)}
        print(f"      {a_ + ' vs ' + b_:<18}" + "  ".join(f"{FINGERS[i]} {100 * r[i]:6.2f} %" for i in range(5)))

    # split per episode
    lens = [len(data[g][0]) for g in groups]
    ends = np.cumsum(lens)
    J_ep = {p: {g: J_all[p][e - n:e] for g, e, n in zip(groups, ends, lens)} for p in POINTS}
    rec = {g: data[g][2] for g in groups if data[g][2] is not None}

    # check against the recorded reference offset (demos and scripted lift)
    if rec:
        rq = kin.root_q[0].cpu()
        print(f"      check against the logged reference offset J (q_ref - q), samples with |offset| > "
              f"{1000 * CONTACT_M:.0f} mm: median |error| [mm] (error / offset)")
        print(f"      {'finger':<8}{'samples':>9}{'|offset| [mm]':>15}" + "".join(f"{'J at ' + p:>22}" for p in POINTS))
        rows = {i: {"err": {p: [] for p in POINTS}, "size": []} for i in range(5)}
        for g, (tro_w, q_ref, lag) in rec.items():
            tro_w = np.asarray(tro_w, np.float32)
            tro = quat_apply_inverse(rq.expand(len(tro_w) * 5, -1),
                                     torch.tensor(tro_w).reshape(-1, 3)).reshape(-1, 5, 3).numpy()
            qh = data[g][1]
            T = len(qh) - lag
            # logged in step t (q from inside the step) vs recomputed from the angles at the end of step t
            tro = tro[:T]
            dq_ref = (q_ref[:T] - qh[lag:lag + T])[:, kin.hand_cols]  # (T, 5, 4)
            on = np.linalg.norm(tro, axis=-1) > CONTACT_M  # (T, 5)
            for p in POINTS:
                d = np.einsum("tfij,tfj->tfi", J_ep[p][g][lag:lag + T], dq_ref)
                for i in range(5):
                    rows[i]["err"][p].append(np.linalg.norm(d[on[:, i], i] - tro[on[:, i], i], axis=-1))
            for i in range(5):
                rows[i]["size"].append(np.linalg.norm(tro[on[:, i], i], axis=-1))
        for i in range(5):
            size = np.concatenate(rows[i]["size"])
            if len(size) == 0:
                print(f"      {FINGERS[i]:<8}{0:>9}")
                continue
            line, entry = f"      {FINGERS[i]:<8}{len(size):>9}{1000 * np.median(size):>15.2f}", {}
            for p in POINTS:
                e_ = np.concatenate(rows[i]["err"][p])
                rel = float(np.median(e_ / size))
                entry[p] = {"median_error_mm": float(1000 * np.median(e_)), "median_relative": rel}
                line += f"{f'{1000 * np.median(e_):.2f} ({100 * rel:.1f} %)':>22}"
            check["logged_offset"][FINGERS[i]] = {"samples": int(len(size)),
                                                  "median_offset_mm": float(1000 * np.median(size)), **entry}
            print(line)

    with h5py.File(out, "w") as f:
        d = f.create_group("data")
        for g in groups:
            d.create_group(g).create_dataset("J", data=J_ep[args_cli.point][g], compression="gzip")
        f.attrs.update({
            "source": os.path.abspath(path), "task": TASK, "point": args_cli.point, "eps": float(args_cli.eps),
            "base_frame": True, "hand_cols": kin.hand_cols, "fingers": json.dumps(FINGERS),
            "check": json.dumps(check), "created": datetime.now().isoformat(timespec="seconds"),
            **{k: np.asarray(v, np.float64) for k, v in kin.drive.items()},
        })
    print(f"[kin] -> {out} (J at the {args_cli.point} point)\n")


def print_drives(kin: Kinematics, hand_names: list[str]):
    """The hand's drive gains, effort limits and joint limits, finger by finger."""
    print("\n[kin] hand drives (as simulated): kp [Nm/rad], kd [Nms/rad], effort limit [Nm], soft joint limits [rad]")
    for f in range(5):
        for j in range(4):
            c = kin.hand_cols[f, j]
            lo, hi = kin.drive["pos_limits"][c]
            print(f"      {FINGERS[f]:<7} joint{j + 1}  kp {kin.drive['kp'][c]:6.3f}  kd {kin.drive['kd'][c]:6.4f}  "
                  f"effort {kin.drive['effort_limit'][c]:7.3f}  limits [{lo:+.3f}, {hi:+.3f}]")
    print()


def main():
    kin = Kinematics(args_cli.num_envs)
    with torch.inference_mode():
        for path in args_cli.files:
            process(kin, path)
    kin.env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
"""Full teleoperation of the IL env: the Quest 3 drives the arm (wrist) and the hand (fingers).

Runs on Isaac-Lift-Cube-UR5Wuji-IL-v0 (fingers-forward start pose, cube randomised in front of the
hand, all actions absolute joint positions). Without --record it is for practising; with --record
it saves the successful demos to an HDF5 file (see "Recording" below).

Data path (as for teleop_hand.py):

    Quest 3 --WebSocket--> stream_joints.py (conda env "quest") --UDP 127.0.0.1:9870--> this script

What happens every env step (50 Hz):

  hand: the newest packet's 20 angles -> q_ref (by joint name), as in teleop_hand.py
  arm:  1. when the clutch engages, remember the pose of the robot's tracked point (the anchor);
        2. target pose = anchor moved by the packet's ee_delta (your wrist's motion since the clutch
           engaged, already in the robot base frame): p* = p0 + dp, R* = dR R0;
        3. one damped-least-squares IK step from the current arm angles q:
               dq = J^T (J J^T + lambda^2 I)^-1 e,   e = (p* - p, rotation error as axis-angle),
           J = the 6x6 Jacobian of the tracked point, base frame;
        4. arm action = q + dq (each joint at most --max_joint_step per step, inside the soft limits).

  The tracked point (--ik_point) decides what your wrist's motion is applied to:
    wrist (default): the palm link's origin, the robot's counterpart of your wrist. The robot hand
           then moves as a rigid copy of your hand: rotating your hand about your wrist rotates the
           robot hand about its wrist, and the fingers swing around it exactly as yours do.
    grasp: the grasp point (EE_OFFSET in front of the palm). Rotations then turn the hand about the
           point between the fingers (useful to rotate a held object in place), which your own hand
           does not do.
  frozen / tracking lost / stream silent: arm and hand hold their last targets.

The IK runs here, outside the env, so the env's action (and later the recorded action) stays the
26 joint positions [arm | hand] decided for imitation learning.

Run (two terminals):

    # terminal 1, conda env "quest", in .../quest_teleop/quest_isaac_bridge
    python stream_joints.py --hand-repo ~/Documents/balazs_folder/quest_teleop/wuji-retargeting --hand-side right
    # headset: Quest Browser -> http://128.178.169.140:8080 -> Start Teleop

    # terminal 2, conda env "env_isaaclab", in the repo root
    python scripts/teleop/teleop_lift.py                     # admittance off (K = inf)
    python scripts/teleop/teleop_lift.py --stiffness 150     # with the admittance
    python scripts/teleop/teleop_lift.py --no_rotation       # hand keeps its orientation (position only)
    python scripts/teleop/teleop_lift.py --stiffness 150 --record    # record demos at K = 150

Operating it: stand as if behind the robot (looking along its +x). Right hand: wrist moves the hand,
fingers move the fingers. LEFT-hand pinch freezes everything: move your own arm back to a comfortable
place, release, carry on (the robot continues from where it is). In this terminal: "r" + Enter resets
the episode (new cube position), "d" + Enter deletes the last saved demo, "q" + Enter quits.

Start of an episode: after every reset (and at the start) the robot holds its start pose until you
pinch with the left hand and release; the episode starts at the release (status line: "pinch +
release to start"). So you can put your hands in a comfortable place first.

Recording (--record): a demo starts at that release. From then on every env step is stored, including later freezes (the
robot then holds still, which ends up in the demo as a pause; so avoid re-clutching during a demo).
The episode ends by itself on success (cube bottom >= 15 cm above the table for 3 s; the green grid
turns dark while the 3 s run). Only successful demos are kept:

    success                              -> saved (unless the run was slower than real time, see below)
    "r" reset, time limit, cube dropped  -> discarded
    "d" + Enter                          -> deletes the last demo saved in this session

One sample per env step (50 Hz): s_t = the IL state (51 numbers, ILPolicyCfg) the step started
from, a_t = the 26 joint commands of that step: the arm targets and the finger commands q_cmd the
admittance produced from the operator's q_ref, i.e. what a policy that performs the admittance
itself sees and outputs.
A demo whose real-time factor (simulated / wall time) is below --min_rtf is not saved: you moved at
a slower pace than the simulated time suggests, which would distort the speeds in the data
(claude/notes_imitation_learning.md, section 6).

File layout (one file per stiffness, default logs/il_demos/lift_K<K>.hdf5; new demos are appended;
a file whose "format" attribute differs, e.g. demos recorded before q_cmd was the action, is refused):

    /                     attrs: format, task, stiffness, controller, device, physics_dt, decimation,
                                 step_dt, lp_alpha, IK settings, success condition, obs_terms (name,
                                 start, end), arm_joints, hand_joints, created
    /data/demo_000/obs      (T, 51) float32
    /data/demo_000/actions  (T, 26) float32   the training action: [arm targets, ARM_JOINTS order |
                                              hand q_cmd, hand_joints order]. q_cmd = the command the
                                              admittance sent to the finger drives in step t, q_ref +
                                              J^+ e clamped to the joint limits (the last of the step's
                                              2 physics steps): the policy performs the admittance itself
    /data/demo_000/q_ref    (T, 20) float32   the operator's finger references (what the Quest sent;
                                              the input of the admittance during recording)
    /data/demo_000/dq       (T, 20) float32   the admittance part J^+ e = q_cmd - q_ref before clamping [rad]
    /data/demo_000/tip_ref_offset (T, 5, 3)   J (q_ref - q) per fingertip [m] (reference penetration)
    /data/demo_000/tip_force_raw  (T, 5, 3)   unsmoothed contact force from the cube per fingertip [N],
                                              world frame (the state holds the smoothed one)
    /data/demo_000          attrs: num_samples, duration_s, wall_s, rtf, late_steps, idle_s (time from
                                   the reset to the clutch), recorded (date and time)

Check a file with scripts/tools/inspect_demos.py.

Status line (once a second): besides position, cube height and forces it shows why the arm may
not follow your hand (which feels like lag): the largest IK error of the last second, how often a
joint wanted to move faster than --max_joint_step ("step-limited"), which joints hit their limits,
and the force on the fingertips from anything but the cube ("other", e.g. the table; contacts of
the palm or finger segments are not measured).

Physics on the CPU and the GUI frame limiter off are defaults, as in teleop_hand.py (see there and
claude/notes_imitation_learning.md, section 6).
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Quest 3 -> UR5 + Wuji hand teleoperation (IL env).")
parser.add_argument("--stiffness", type=float, default=math.inf, help="K [N/m] of the fingertip admittance; inf = off")
parser.add_argument("--controller", choices=["dynamic", "static"], default="dynamic")
parser.add_argument("--udp", default="127.0.0.1:9870", help="HOST:PORT the stream is sent to (stream_joints.py --to)")
parser.add_argument("--no_rotation", action="store_true", help="ignore the wrist rotation (hand keeps its orientation)")
parser.add_argument("--ik_point", choices=["wrist", "grasp"], default="wrist",
                    help="robot point that follows your wrist's motion: the palm origin (wrist) or the grasp point")
parser.add_argument("--max_joint_step", type=float, default=0.05,
                    help="largest arm joint change per env step [rad] (0.05 = 2.5 rad/s); guards against IK jumps")
parser.add_argument("--ik_damping", type=float, default=0.05, help="lambda of the damped least-squares IK")
parser.add_argument("--cam_eye", type=float, nargs=3, default=[0.30, 0.45, 0.25],
                    help="camera position relative to the palm [m] (the camera follows the palm)")
parser.add_argument("--cam_lookat", type=float, nargs=3, default=[0.05, 0.0, -0.10],
                    help="point the camera looks at, relative to the palm [m]")
parser.add_argument("--episode_s", type=float, default=300.0,
                    help="time limit of an episode [s] (the env's default is 60 s; longer for practising)")
parser.add_argument("--print_every", type=int, default=50, help="status line every N env steps (50 = 1 s)")
parser.add_argument("--record", action="store_true", help="save successful demos to --dataset")
parser.add_argument("--dataset", default=None,
                    help="HDF5 file for the demos (default logs/il_demos/lift_K<stiffness>.hdf5); appended to")
parser.add_argument("--min_rtf", type=float, default=0.95,
                    help="demos recorded slower than this real-time factor are not saved")
parser.add_argument("--lp_alpha", type=float, default=0.2,
                    help="the bridge's finger filter (stream_joints.py --lp-alpha); only stored in the file")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")  # one env: CPU physics is much faster than the GPU
args_cli = parser.parse_args()
FRAME_LIMIT_OFF = "--/app/runLoops/main/rateLimitEnabled=false --/app/vsync=false"
args_cli.kit_args = f"{args_cli.kit_args or ''} {FRAME_LIMIT_OFF}".strip()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import datetime
import json
import os
import queue
import socket
import sys
import threading
import time

import gymnasium as gym
import h5py
import numpy as np
import torch

from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    matrix_from_quat,
    quat_from_angle_axis,
    quat_mul,
    skew_symmetric_matrix,
    subtract_frame_transforms,
)
from isaaclab_tasks.utils import parse_env_cfg

import tunable_comp_thesis.tasks  # noqa: F401  (registers the gym tasks)
from tunable_comp_thesis.controllers import AdmittanceControllerCfg, StaticAdmittanceControllerCfg
from tunable_comp_thesis.tasks.manager_based.lift_compliant.config.ur5_wuji.il_env_cfg import (
    ARM_JOINTS,
    EE_OFFSET,
    STIFFNESS_VALUES,
)

TASK = "Isaac-Lift-Cube-UR5Wuji-IL-v0"
STALE_S = 0.3  # [s] a packet older than this counts as "no stream"
PALM = "right_palm_link"
CUBE_REST = 0.021  # [m] rest height of the cube centre (for the "lifted by" display)


# ------------------------------------------------------------------------------------------------
# joint mapping and UDP receiver (as in teleop_hand.py)
# ------------------------------------------------------------------------------------------------

def bridge_index(joint_name: str) -> int:
    """right_finger{f}_joint{j} -> entry 4 (f - 1) + (j - 1) of the bridge's finger-major hand_q."""
    finger = int(joint_name.split("finger")[1][0])
    joint = int(joint_name[-1])
    return 4 * (finger - 1) + (joint - 1)


class QuestReceiver:
    """Non-blocking UDP receiver for stream_joints.py; keeps only the newest packet."""

    def __init__(self, address: str):
        host, _, port = address.rpartition(":")
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.bind((host, int(port)))
        except OSError as exc:
            raise SystemExit(f"[teleop] cannot listen on {address} ({exc}). Is another receiver still running?") from exc
        self._sock.setblocking(False)
        self.packet: dict | None = None
        self.count = 0
        self.gaps = 0
        self._t_recv = -math.inf
        self._last_seq: int | None = None

    def poll(self) -> dict | None:
        while True:
            try:
                data, _ = self._sock.recvfrom(65535)
            except BlockingIOError:
                break
            try:
                packet = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            seq = packet.get("seq")
            if self._last_seq is not None and seq is not None and seq != self._last_seq + 1:
                self.gaps += 1
            self._last_seq = seq
            self.packet, self._t_recv = packet, time.monotonic()
            self.count += 1
        return self.packet

    @property
    def age(self) -> float:
        return time.monotonic() - self._t_recv

    def close(self):
        self._sock.close()


def stdin_commands() -> queue.Queue:
    """Lines typed in this terminal ("r", "q"), read in a background thread so the loop never waits."""
    q: queue.Queue = queue.Queue()

    def reader():
        for line in sys.stdin:
            q.put(line.strip().lower())

    threading.Thread(target=reader, daemon=True).start()
    return q


# ------------------------------------------------------------------------------------------------
# arm: grasp-point pose, Jacobian and one IK step
# ------------------------------------------------------------------------------------------------

class ArmIK:
    """Damped-least-squares IK for a point fixed to the palm (palm origin + offset), robot base frame."""

    def __init__(self, robot, arm_ids: list[int], damping: float, max_step: float, offset=(0.0, 0.0, 0.0)):
        self.robot, self.arm_ids = robot, arm_ids
        self.palm_id = robot.find_bodies(PALM)[0][0]
        # a fixed-base Jacobian has no row for the root body, so all body rows shift by one
        self.jac_row = self.palm_id - 1 if robot.is_fixed_base else self.palm_id
        self.lam2 = damping**2
        self.max_step = max_step
        dev = robot.device
        self.offset = torch.tensor([offset], dtype=torch.float, device=dev)  # tracked point in the palm frame
        self.unit_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev)
        self.limits = robot.data.soft_joint_pos_limits[:, arm_ids]  # (1, 6, 2)
        # diagnostics of the last step, for the status line
        self.step_limited = False  # some joint wanted to move more than max_step
        self.at_limit = torch.zeros(len(arm_ids), dtype=torch.bool, device=dev)  # joints pressed against a limit

    def pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Tracked point: position (1, 3) and orientation (1, 4, wxyz) in the base frame."""
        d = self.robot.data
        p_palm, q_palm = subtract_frame_transforms(d.root_pos_w, d.root_quat_w,
                                                   d.body_pos_w[:, self.palm_id], d.body_quat_w[:, self.palm_id])
        return combine_frame_transforms(p_palm, q_palm, self.offset, self.unit_quat)

    def jacobian(self) -> torch.Tensor:
        """6 x 6 Jacobian of the tracked point w.r.t. the arm joints, base frame (linear rows first)."""
        d = self.robot.data
        J = self.robot.root_physx_view.get_jacobians()[:, self.jac_row, :, :][:, :, self.arm_ids].clone()  # world
        # the palm Jacobian is for the palm origin; the tracked point is r = R_palm offset away from it:
        # v_point = v_palm + w x r  ->  J_lin += -[r]_x J_ang   (r = 0 for the wrist: nothing changes)
        r_w = (matrix_from_quat(d.body_quat_w[:, self.palm_id]) @ self.offset.unsqueeze(-1)).squeeze(-1)
        J[:, 0:3] += torch.bmm(-skew_symmetric_matrix(r_w), J[:, 3:6])
        # world -> base frame (the UR5 base is not rotated here, but do it properly)
        R_bw = matrix_from_quat(d.root_quat_w).transpose(1, 2)
        J[:, 0:3] = torch.bmm(R_bw, J[:, 0:3])
        J[:, 3:6] = torch.bmm(R_bw, J[:, 3:6])
        return J

    def step(self, p_des: torch.Tensor, q_des: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        """One IK step towards the pose (p_des, q_des). Returns arm joint targets (1, 6) and the
        remaining error before the step [m], [rad] (for the status line)."""
        p, q = self.pose()
        pos_err, rot_err = compute_pose_error(p, q, p_des, q_des, rot_error_type="axis_angle")
        e = torch.cat([pos_err, rot_err], dim=-1).unsqueeze(-1)  # (1, 6, 1)
        J = self.jacobian()
        A = J @ J.transpose(1, 2) + self.lam2 * torch.eye(6, device=J.device)
        dq = (J.transpose(1, 2) @ torch.linalg.solve(A, e)).squeeze(-1)  # (1, 6)
        self.step_limited = bool((dq.abs() > self.max_step).any())
        dq = dq.clamp(-self.max_step, self.max_step)
        q_now = self.robot.data.joint_pos[:, self.arm_ids]
        q_want = q_now + dq
        q_new = torch.clamp(q_want, self.limits[..., 0], self.limits[..., 1])
        self.at_limit = (q_want != q_new)[0]
        return q_new, pos_err.norm().item(), rot_err.norm().item()


def rotvec_to_quat(rv: torch.Tensor) -> torch.Tensor:
    """Rotation vector (1, 3) -> quaternion (1, 4, wxyz); zero vector -> identity."""
    angle = rv.norm(dim=-1)
    return quat_from_angle_axis(angle, rv)  # the axis is normalised inside (safe for angle 0)


# ------------------------------------------------------------------------------------------------
# recording: one HDF5 file, one group per successful demo
# ------------------------------------------------------------------------------------------------

class DemoFile:
    """Appends demos to an HDF5 file (layout in the module docstring).

    The file is opened only to write or delete a demo and closed straight after, so a crash or a
    killed process never leaves it half-written. If the file exists, its settings must match this
    run's (same stiffness, state and action sizes, step time); demos of different setups are not mixed.
    """

    MUST_MATCH = ("format", "task", "stiffness", "obs_dim", "action_dim", "step_dt")

    def __init__(self, path: str, attrs: dict):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with h5py.File(path, "a") as f:
            if "data" not in f:  # new file
                f.create_group("data")
                for k, v in attrs.items():
                    f.attrs[k] = v
            else:
                for k in self.MUST_MATCH:
                    old, new = f.attrs.get(k), attrs[k]
                    same = np.isclose(old, new) if isinstance(new, float) else old == new
                    if not same:
                        raise SystemExit(f"[record] {path} was recorded with {k} = {old}, this run has {new}. "
                                         "Use another --dataset.")
            self.count = len(f["data"])
        self.saved: list[str] = []  # demos saved in this session, for "d" (delete the last one)

    def save(self, obs: np.ndarray, actions: np.ndarray, attrs: dict, extra: dict | None = None) -> str:
        with h5py.File(self.path, "a") as f:
            data = f["data"]
            index = 1 + max((int(n.split("_")[1]) for n in data), default=-1)  # never reuse a name
            name = f"demo_{index:03d}"
            g = data.create_group(name)
            g.create_dataset("obs", data=obs.astype(np.float32))
            g.create_dataset("actions", data=actions.astype(np.float32))
            for k, v in (extra or {}).items():  # q_cmd, dq, ...
                g.create_dataset(k, data=v.astype(np.float32))
            for k, v in attrs.items():
                g.attrs[k] = v
            self.count = len(data)
        self.saved.append(name)  # (h5py objects such as g are invalid once the file is closed)
        return name

    def delete_last(self) -> str | None:
        if not self.saved:
            return None
        name = self.saved.pop()
        with h5py.File(self.path, "a") as f:
            del f["data"][name]
            self.count = len(f["data"])
        return name


# ------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------

def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    env_cfg.actions.hand_action.stiffness_values = [args_cli.stiffness]
    env_cfg.actions.hand_action.controller = (
        AdmittanceControllerCfg() if args_cli.controller == "dynamic" else StaticAdmittanceControllerCfg()
    )
    # teleoperation: the operator sends references q_ref, the admittance turns them into the joint
    # commands q_cmd (the recorded action). The IL env has the law off by default (for the policy).
    env_cfg.actions.hand_action.admittance_enabled = True
    env_cfg.viewer.origin_type = "asset_body"  # the camera follows the palm
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.body_name = PALM
    env_cfg.viewer.eye = tuple(args_cli.cam_eye)
    env_cfg.viewer.lookat = tuple(args_cli.cam_lookat)
    env_cfg.episode_length_s = args_cli.episode_s

    env = gym.make(TASK, cfg=env_cfg)
    uw = env.unwrapped
    robot, cube = uw.scene["robot"], uw.scene["object"]
    hand = uw.action_manager.get_term("hand_action")
    dev, dt = uw.device, uw.step_dt
    success_cfg = uw.termination_manager.get_term_cfg("success")
    success_term = success_cfg.func  # CubeHeldAboveHeight: hold timer for the status line
    hold_s = success_cfg.params["hold_s"]

    arm_ids = robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
    hand_ids = robot.find_joints(hand.joint_names, preserve_order=True)[0]
    perm = torch.tensor([bridge_index(n) for n in hand.joint_names], device=dev)
    assert sorted(perm.tolist()) == list(range(20)), "the hand mapping does not use every bridge angle once"
    ik = ArmIK(robot, arm_ids, args_cli.ik_damping, args_cli.max_joint_step,
               offset=(0.0, 0.0, 0.0) if args_cli.ik_point == "wrist" else EE_OFFSET)
    grasp = ArmIK(robot, arm_ids, args_cli.ik_damping, args_cli.max_joint_step, offset=EE_OFFSET)  # display only

    def start_targets():
        """Arm and hand targets that hold the start pose (default angles)."""
        q0 = robot.data.default_joint_pos
        return q0[:, arm_ids].clone(), q0[:, hand_ids].clone()

    obs, _ = env.reset()
    obs = obs["policy"]  # (1, 51): the IL state
    q_arm, q_ref = start_targets()
    p0, R0 = grasp.pose()
    k_name = "inf (admittance off)" if math.isinf(args_cli.stiffness) else f"{args_cli.stiffness:g} N/m"
    print(f"\n[teleop] {TASK}: K = {k_name}, {args_cli.controller} controller, device {dev}, "
          f"rotation {'off' if args_cli.no_rotation else 'on'}, IK point: {args_cli.ik_point}")
    print(f"[teleop] grasp point at the start (base frame): {[round(v, 3) for v in p0[0].tolist()]} m")
    print(f"[teleop] IK: damping {args_cli.ik_damping}, at most {args_cli.max_joint_step} rad per step; "
          f"episodes end after {args_cli.episode_s:g} s")
    tip_sensor = uw.scene["fingertip_contacts"]  # unfiltered: everything touching the fingertips

    # --- recording setup ---------------------------------------------------------------------------
    demos: DemoFile | None = None
    if args_cli.record:
        if args_cli.stiffness not in STIFFNESS_VALUES:
            print(f"[record] WARNING: K = {args_cli.stiffness:g} is not one of the demo stiffnesses {STIFFNESS_VALUES}")
        path = args_cli.dataset or os.path.join("logs", "il_demos", f"lift_K{args_cli.stiffness:g}.hdf5")
        om = uw.observation_manager
        names, dims = om.active_terms["policy"], [d[0] for d in om.group_obs_term_dim["policy"]]
        ends = np.cumsum(dims).tolist()
        attrs = {
            "task": TASK,
            "stiffness": float(args_cli.stiffness),
            "controller": args_cli.controller,
            "device": str(dev),
            "physics_dt": float(uw.physics_dt),
            "decimation": int(uw.cfg.decimation),
            "step_dt": float(dt),
            "lp_alpha": float(args_cli.lp_alpha),
            "ik_point": args_cli.ik_point,
            "ik_damping": float(args_cli.ik_damping),
            "max_joint_step": float(args_cli.max_joint_step),
            "no_rotation": bool(args_cli.no_rotation),
            "success": json.dumps({k: v for k, v in success_cfg.params.items() if k in ("height", "half_size", "hold_s", "tolerance")}),
            "obs_dim": int(sum(dims)),
            "action_dim": int(uw.action_manager.total_action_dim),
            "obs_terms": json.dumps([[n, e - d, e] for n, d, e in zip(names, dims, ends)]),
            "arm_joints": json.dumps(ARM_JOINTS),
            "hand_joints": json.dumps(list(hand.joint_names)),
            "action_layout": "[arm joint targets (arm_joints) | hand q_cmd (hand_joints)], absolute, rad",
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "format": "actions = [arm targets | hand q_cmd]; per demo also q_ref, dq, tip_ref_offset, tip_force_raw",
        }
        demos = DemoFile(path, attrs)
        print(f"[record] recording successful demos to {path} ({demos.count} already in it); "
              f"a demo starts when the clutch engages, demos below x{args_cli.min_rtf} real time are not saved")

    rec: dict | None = None  # the demo being recorded: lists of obs and actions, start times
    episode_step = 0  # env steps since the last reset (for the idle time before the clutch)

    def discard(why: str):
        nonlocal rec
        if rec is not None:
            print(f"[record] demo discarded ({why}, {len(rec['obs']) * dt:.1f} s)")
        rec = None

    rx = QuestReceiver(args_cli.udp)
    cmds = stdin_commands()
    print(f"[teleop] listening on {args_cli.udp}; type r + Enter = reset episode, "
          f"{'d + Enter = delete the last saved demo, ' if demos else ''}q + Enter = quit\n")

    anchor: tuple[torch.Tensor, torch.Tensor] | None = None  # grasp pose when the clutch engaged
    was_live, step, count_prev, late = False, 0, 0, 0
    # start gate: after every reset (and at the start) the robot holds its start pose until the
    # operator pinches and releases, so every episode (and demo) starts from a fresh clutch engagement
    wait_clutch, seen_pinch = True, False
    pos_err = rot_err = 0.0
    # per status interval: why the arm may not follow (shown as "lag" to the operator)
    diag = {"live": 0, "limited": 0, "pos": 0.0, "rot": 0.0, "other": 0.0}
    limit_hits = torch.zeros(len(arm_ids), dtype=torch.bool, device=dev)
    t_next = t_print = time.perf_counter()
    with torch.inference_mode():
        while simulation_app.is_running():
            # --- terminal commands ---------------------------------------------------------------
            reset_now = False
            while not cmds.empty():
                c = cmds.get()
                if c == "q":
                    discard("quit")
                    print("[teleop] quit")
                    rx.close()
                    env.close()
                    return
                if c == "r":
                    reset_now = True
                if c == "d" and demos is not None:
                    name = demos.delete_last()
                    print(f"[record] deleted {name}, {demos.count} demos in the file" if name
                          else "[record] nothing to delete (no demo saved in this session)")
            if reset_now:
                discard("reset")
                obs, _ = env.reset()
                obs = obs["policy"]
                q_arm, q_ref = start_targets()
                anchor, was_live, episode_step = None, False, 0
                wait_clutch, seen_pinch = True, False
                print("[teleop] episode reset (new cube position): pinch + release to start")

            # --- what the operator does this step ------------------------------------------------
            packet = rx.poll()
            if packet is None:
                live, reason = False, "no stream yet (is stream_joints.py running?)"
            elif rx.age > STALE_S:
                live, reason = False, f"stream silent for {rx.age:.1f} s"
            elif not packet.get("clutch", False) or packet.get("ee_delta") is None:
                live, reason = False, packet.get("reason") or "frozen"
            elif packet.get("hand_q") is None or len(packet["hand_q"]) != 20:
                live, reason = False, "no hand_q (stream_joints.py without --hand-repo?)"
            else:
                live, reason = True, ""

            # start gate: ignore the stream until a pinch (a fresh packet with the clutch off) has
            # been followed by a release; the bridge re-anchors at the release, so ee_delta starts at 0
            if wait_clutch:
                if live and seen_pinch:
                    wait_clutch = False  # released after a pinch: go (this step is the first live one)
                else:
                    if packet is not None and rx.age <= STALE_S and not packet.get("clutch", False):
                        seen_pinch = True
                    if packet is not None and rx.age <= STALE_S:  # keep the stream problems' reasons
                        reason = "pinching: release to start" if seen_pinch else "pinch + release to start"
                    live = False

            if live:
                # the bridge re-anchors whenever its clutch engages; re-anchor here at the same moment
                if not was_live or anchor is None:
                    anchor = ik.pose()
                p_a, q_a = anchor
                delta = torch.tensor([packet["ee_delta"]], dtype=torch.float, device=dev)  # (1, 6)
                p_des = p_a + delta[:, 0:3]
                q_des = q_a if args_cli.no_rotation else quat_mul(rotvec_to_quat(delta[:, 3:6]), q_a)
                q_arm, pos_err, rot_err = ik.step(p_des, q_des)
                diag["live"] += 1
                diag["limited"] += int(ik.step_limited)
                diag["pos"], diag["rot"] = max(diag["pos"], pos_err), max(diag["rot"], rot_err)
                limit_hits |= ik.at_limit
                q_ref = torch.tensor(packet["hand_q"], dtype=torch.float, device=dev)[perm].unsqueeze(0)
            # not live: q_arm and q_ref keep their last values (hold, do not snap anywhere)
            was_live = live
            action = torch.cat([q_arm, q_ref], dim=-1)  # 26 absolute joint positions [arm | hand]

            # --- recording: starts at the first live step after a reset, i.e. at the release -------
            if demos is not None and rec is None and live:
                rec = {"obs": [], "act": [], "q_ref": [], "dq": [], "tip_ref_offset": [], "tip_force_raw": [],
                       "t0": time.perf_counter(), "late0": late, "idle_s": episode_step * dt}
                print(f"[record] demo started (clutch engaged after {rec['idle_s']:.1f} s)")
            if rec is not None:
                rec["obs"].append(obs[0].cpu().numpy().copy())  # s_t: the state this step starts from
                rec["q_ref"].append(q_ref[0].cpu().numpy().copy())  # the operator's finger references

            # --- env step ------------------------------------------------------------------------
            obs, _, terminated, truncated, _ = env.step(action)
            obs = obs["policy"]
            if rec is not None:  # what the admittance did in this step (these buffers survive an env reset)
                # a_t = [arm targets sent | finger command the admittance produced in this step]
                rec["act"].append(np.concatenate([q_arm[0].cpu().numpy(), hand.joint_command[0].cpu().numpy()]))
                rec["dq"].append(hand.joint_offsets[0].cpu().numpy().copy())
                rec["tip_ref_offset"].append(hand.tip_reference_offsets[0].cpu().numpy().copy())
                rec["tip_force_raw"].append(hand.tip_forces[0].cpu().numpy().copy())
            step += 1
            episode_step += 1
            # fingertip contact with anything but the cube (table, other fingers): net force minus cube force
            net = tip_sensor.data.net_forces_w[0, hand._sensor_rows]  # (5, 3), finger1..5
            diag["other"] = max(diag["other"], (net - hand.tip_forces[0]).norm(dim=-1).sum().item())
            if bool(terminated[0]) or bool(truncated[0]):  # the env has reset itself; obs is the new episode's
                success = bool(terminated[0]) and bool(uw.termination_manager.get_term("success")[0])
                if success:
                    print(f"[teleop] SUCCESS: cube held above the target for {hold_s:g} s")
                    if rec is not None:
                        n = len(rec["obs"])
                        wall = time.perf_counter() - rec["t0"]
                        rtf = n * dt / max(wall, 1e-6)
                        if rtf < args_cli.min_rtf:
                            print(f"[record] demo NOT saved: ran at x{rtf:.2f} real time (< {args_cli.min_rtf})")
                        else:
                            name = demos.save(np.stack(rec["obs"]), np.stack(rec["act"]), {
                                "num_samples": n,
                                "duration_s": n * dt,
                                "wall_s": wall,
                                "rtf": rtf,
                                "late_steps": late - rec["late0"],
                                "idle_s": rec["idle_s"],
                                "recorded": datetime.datetime.now().isoformat(timespec="seconds"),
                            }, extra={k_: np.stack(rec[k_]) for k_ in ("q_ref", "dq", "tip_ref_offset", "tip_force_raw")})
                            print(f"[record] saved {name}: {n} samples, {n * dt:.1f} s, x{rtf:.2f} real time "
                                  f"-> {demos.count} demos in {demos.path}")
                        rec = None
                else:
                    why = "time limit" if bool(truncated[0]) else "cube off the table"
                    discard(why)
                    print(f"[teleop] episode ended by the env ({why}), reset")
                q_arm, q_ref = start_targets()
                anchor, was_live, episode_step = None, False, 0
                wait_clutch, seen_pinch = True, False
                print("[teleop] pinch + release to start the next episode")

            # --- status line, about once a second ------------------------------------------------
            if args_cli.print_every and step % args_cli.print_every == 0:
                now = time.perf_counter()
                wall = max(now - t_print, 1e-6)
                rate, rtf = (rx.count - count_prev) / wall, args_cli.print_every * dt / wall
                count_prev, t_print = rx.count, now
                p, _ = grasp.pose()
                cube_p, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, cube.data.root_pos_w)
                f_sum = hand.tip_forces[0].norm(dim=-1).sum().item()
                state = "LIVE" if live else f"held: {reason}"
                limited = 100 * diag["limited"] / max(diag["live"], 1)
                at_limit = [ARM_JOINTS[k].replace("_joint", "") for k in torch.nonzero(limit_hits).flatten().tolist()]
                hold = f" | hold {success_term.hold_time[0].item():3.1f}/{hold_s:g} s" if bool(success_term.above[0]) else ""
                rec_state = ""
                if demos is not None:
                    rec_state = f"REC {len(rec['obs']) * dt:5.1f} s | " if rec is not None else "rec: wait clutch | "
                print(f"[{step * dt:7.1f} s] {rec_state}{state:<34.34} | grasp pt {' '.join(f'{v:+.3f}' for v in p[0].tolist())} m"
                      f" | IK err max {1000 * diag['pos']:4.0f} mm {math.degrees(diag['rot']):5.1f} deg, "
                      f"step-limited {limited:3.0f} %{', at limit: ' + ' '.join(at_limit) if at_limit else ''}"
                      f" | cube lifted {100 * (cube_p[0, 2].item() - CUBE_REST):5.1f} cm{hold} | sum|f| cube {f_sum:4.1f} N,"
                      f" other {diag['other']:4.1f} N | stream {rate:4.1f} Hz | real time x{rtf:.2f}, late {late}")
                diag = {"live": 0, "limited": 0, "pos": 0.0, "rot": 0.0, "other": 0.0}
                limit_hits[:] = False

            # --- real time ------------------------------------------------------------------------
            t_next += dt
            sleep = t_next - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                late += 1
                t_next = time.perf_counter()

    discard("app closed")
    rx.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
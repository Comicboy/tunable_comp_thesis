"""Hand-only teleoperation: the Quest 3 drives the 20 finger joints of the simulated Wuji hand.

First teleop step. The arm holds a fixed pose; only the fingers follow the operator. Nothing is
recorded. The point is to check, by eye, that the retargeted angles land on the right joints with
the right signs and that the hand follows smoothly through the fingertip admittance.

Data path:

    Quest 3 --WebSocket--> stream_joints.py (conda env "quest") --UDP 127.0.0.1:9870--> this script

This script only *receives*: the hand tracking, the clutch and the retargeting all happen in
stream_joints.py (the colleague's quest_isaac_bridge), so nothing of it has to be installed in the
Isaac Lab env. Each env step the newest packet's 20 angles become the hand reference q_ref of the
FingertipAdmittanceAction, exactly as a policy's output will later.

Run (two terminals):

    # terminal 1, conda env "quest", in .../quest_teleop/quest_isaac_bridge
    python stream_joints.py --hand-repo ~/Documents/balazs_folder/quest_teleop/wuji-retargeting --hand-side right
    # headset: Quest Browser -> http://128.178.169.140:8080 -> Start Teleop

    # terminal 2, conda env "env_isaaclab", in the repo root
    python scripts/teleop/teleop_hand.py                    # admittance off (K = inf)
    python scripts/teleop/teleop_hand.py --stiffness 150    # with the admittance

Operating it: right hand drives the fingers, a LEFT-hand pinch freezes them (they hold the last
reference). If the stream stops or the right hand is lost, the fingers also hold.
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Quest 3 -> Wuji hand teleoperation, arm fixed.")
parser.add_argument("--stiffness", type=float, default=math.inf, help="K [N/m] of the fingertip admittance; inf = off")
parser.add_argument("--controller", choices=["dynamic", "static"], default="dynamic")
parser.add_argument("--udp", default="127.0.0.1:9870", help="HOST:PORT the stream is sent to (stream_joints.py --to)")
parser.add_argument("--ee_pos", type=float, nargs=3, default=None,
                    help="grasp point position to hold, robot base frame [m] (default: the start pose)")
parser.add_argument("--cam_eye", type=float, nargs=3, default=[0.30, -0.30, 0.15],
                    help="camera position relative to the palm [m] (the camera follows the palm)")
parser.add_argument("--cam_lookat", type=float, nargs=3, default=[0.0, 0.0, -0.05],
                    help="point the camera looks at, relative to the palm [m]")
parser.add_argument("--print_every", type=int, default=50, help="status line every N env steps (50 = 1 s)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import socket
import time

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import tunable_comp_thesis.tasks  # noqa: F401  (registers the gym tasks)
from tunable_comp_thesis.controllers import AdmittanceControllerCfg, StaticAdmittanceControllerCfg

TASK = "Isaac-Lift-Cube-UR5Wuji-Scripted-v0"
STALE_S = 0.3  # [s] a packet older than this counts as "no stream" (the bridge uses the same value for lost hands)
BRIDGE_FINGERS = ("thumb", "index", "middle", "ring", "pinky")


# ------------------------------------------------------------------------------------------------
# joint mapping: bridge layout -> this robot's hand joints
# ------------------------------------------------------------------------------------------------

def bridge_index(joint_name: str) -> int:
    """Position of a joint in the bridge's hand_q.

    hand_q is 20 floats, finger-major: [thumb J0..J3, index J0..J3, middle, ring, pinky], with J0 the
    knuckle, J1 abduction (thumb: opposition), J2 the middle and J3 the distal joint. Our joints are
    named right_finger{f}_joint{j} with finger 1 = thumb and joint 2 = abduction, so finger f, joint j
    is simply entry 4 (f - 1) + (j - 1). Done by name, so the action's own joint order (breadth-first)
    cannot get mixed up with the bridge's.
    """
    finger = int(joint_name.split("finger")[1][0])  # "right_finger3_joint2" -> 3
    joint = int(joint_name[-1])  # -> 2
    return 4 * (finger - 1) + (joint - 1)


# ------------------------------------------------------------------------------------------------
# UDP receiver
# ------------------------------------------------------------------------------------------------

class QuestReceiver:
    """Non-blocking UDP receiver for stream_joints.py. Keeps only the newest packet.

    Polled once per env step: everything that arrived since the last poll is read and all but the
    newest packet is dropped (for teleoperation only the latest sample matters; the stream runs at
    40 Hz, the env at 50 Hz, so most steps see one new packet or none).
    """

    def __init__(self, address: str):
        host, _, port = address.rpartition(":")
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.bind((host, int(port)))
        except OSError as exc:
            raise SystemExit(f"[teleop] cannot listen on {address} ({exc}). Is receive_example.py still running? "
                             "Only one program can listen on the port.") from exc
        self._sock.setblocking(False)
        self.packet: dict | None = None
        self.count = 0  # packets received
        self.gaps = 0  # jumps in the sequence number (dropped packets)
        self._t_recv = -math.inf
        self._last_seq: int | None = None

    def poll(self) -> dict | None:
        """Read everything waiting in the socket, keep the newest packet, return it (or None)."""
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
            self.packet = packet
            self._t_recv = time.monotonic()
            self.count += 1
        return self.packet

    @property
    def age(self) -> float:
        """Seconds since the newest packet arrived (inf before the first one)."""
        return time.monotonic() - self._t_recv

    def close(self):
        self._sock.close()


# ------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------

def main():
    # --- environment: the scripted-lift env with one env (DiffIK arm, absolute hand references) ---------
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    env_cfg.actions.hand_action.stiffness_values = [args_cli.stiffness]
    env_cfg.actions.hand_action.controller = (
        AdmittanceControllerCfg() if args_cli.controller == "dynamic" else StaticAdmittanceControllerCfg()
    )
    env_cfg.episode_length_s = 3600.0  # the scripted env times out after 20 s; teleop runs until closed
    # the camera follows the palm, so the hand stays in view wherever the arm is
    env_cfg.viewer.origin_type = "asset_body"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.body_name = "right_palm_link"
    env_cfg.viewer.eye = tuple(args_cli.cam_eye)
    env_cfg.viewer.lookat = tuple(args_cli.cam_lookat)

    env = gym.make(TASK, cfg=env_cfg)
    uw = env.unwrapped
    arm = uw.action_manager.get_term("arm_action")
    hand = uw.action_manager.get_term("hand_action")
    robot = uw.scene["robot"]
    dev, dt = uw.device, uw.step_dt

    # --- joint mapping, printed so it can be checked once -----------------------------------------------
    perm = torch.tensor([bridge_index(n) for n in hand.joint_names], device=dev)
    assert sorted(perm.tolist()) == list(range(20)), "the mapping does not use every bridge angle exactly once"
    print("\n[teleop] joint mapping (action order <- bridge hand_q):")
    for k, name in enumerate(hand.joint_names):
        b = perm[k].item()
        print(f"    {k:2d}  {name:<22} <- hand_q[{b:2d}]  {BRIDGE_FINGERS[b // 4]:<6} J{b % 4}")

    # --- start: hold the arm where it is, hand at its default pose until the first live packet -----------
    env.reset()
    p_hold, quat_hold = arm._compute_frame_pose()  # grasp point pose in the base frame, (1, 3), (1, 4)
    print(f"\n[teleop] grasp point at the start (base frame): {[round(v, 3) for v in p_hold[0].tolist()]} m")
    if args_cli.ee_pos is not None:
        p_hold = torch.tensor([args_cli.ee_pos], dtype=torch.float, device=dev)
        print(f"[teleop] holding it at --ee_pos {args_cli.ee_pos} instead")
    hand_ids = robot.find_joints(hand.joint_names, preserve_order=True)[0]
    q_ref = robot.data.default_joint_pos[:, hand_ids].clone()  # (1, 20), absolute [rad]

    k_name = "inf (admittance off)" if math.isinf(args_cli.stiffness) else f"{args_cli.stiffness:g} N/m"
    print(f"[teleop] K = {k_name}, {args_cli.controller} controller, env step {1000 * dt:.0f} ms")

    rx = QuestReceiver(args_cli.udp)
    print(f"[teleop] listening on {args_cli.udp} for stream_joints.py; Ctrl-C or close the window to stop\n")

    # --- loop -------------------------------------------------------------------------------------------
    step, count_prev, warned_no_q = 0, 0, False
    t_next = t_print = time.perf_counter()
    late = 0  # steps that could not keep up with real time
    with torch.inference_mode():
        while simulation_app.is_running():
            packet = rx.poll()

            # live = the operator drives the fingers this step; otherwise hold the last reference
            if packet is None:
                live, reason = False, "no stream yet (is stream_joints.py running?)"
            elif rx.age > STALE_S:
                live, reason = False, f"stream silent for {rx.age:.1f} s"
            elif not packet.get("clutch", False):
                live, reason = False, packet.get("reason") or "frozen"
            elif packet.get("hand_q") is None or len(packet["hand_q"]) != 20:
                live, reason = False, "no hand_q in the stream (stream_joints.py without --hand-repo?)"
                if not warned_no_q:
                    print("[teleop] WARNING: packets arrive but carry no hand angles; start stream_joints.py "
                          "with --hand-repo <wuji-retargeting>")
                    warned_no_q = True
            else:
                live, reason = True, ""
                q_bridge = torch.tensor(packet["hand_q"], dtype=torch.float, device=dev)
                q_ref = q_bridge[perm].unsqueeze(0)  # (1, 20) in the action's joint order

            # arm: same pose every step (DiffIK holds it); hand: the reference
            env.step(torch.cat([p_hold, quat_hold, q_ref], dim=-1))
            step += 1

            # status line, about once a second
            if args_cli.print_every and step % args_cli.print_every == 0:
                now = time.perf_counter()
                rate = (rx.count - count_prev) / max(now - t_print, 1e-6)
                count_prev, t_print = rx.count, now
                forces = hand.tip_forces[0].norm(dim=-1).tolist()  # |f| per fingertip [N], thumb first
                thumb = [q_ref[0, k].item() for k, n in enumerate(hand.joint_names) if "finger1_" in n]
                state = "LIVE" if live else f"held: {reason}"
                print(f"[{step * dt:7.1f} s] stream {rate:4.1f} Hz, {rx.gaps} gaps | {state:<40.40} | "
                      f"thumb q_ref {' '.join(f'{v:+.2f}' for v in thumb)} | "
                      f"|f| tips {' '.join(f'{v:4.1f}' for v in forces)} N | late steps {late}")

            # real time: one env step per dt of wall-clock time, so the hand moves with yours
            t_next += dt
            sleep = t_next - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                late += 1
                t_next = time.perf_counter()  # fell behind: do not try to catch up in a burst

    rx.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
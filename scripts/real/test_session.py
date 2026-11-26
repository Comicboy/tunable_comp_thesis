#!/usr/bin/env python3
"""Offline run of RealTeleopSession inside the supervisor's real TeleopSession loop.

    python scripts/real/test_session.py

No robot, hand, glove or headset: a fake arm (the TCP follows the servo command), the Quest's
DualHandInput fed with made-up hands, a fake keyboard and HUD, the glove and hand-stream fakes of
test_tactile_admittance.py. His session loop, safety limiter, wrist retargeter and recorder, and
our hand filter, success check and recorders are the real ones; h5py writes to a temp folder.
A scripted operator goes through whole episodes:

  1. at home: SPACE ignored, pinch alone does not start, pinch + release starts;
  2. grasp + 17 cm lift -> saved by itself after 3 s; the file holds the demo with our keys;
  3. d deletes it; h homes; r mid-episode discards and homes; s is ignored;
  4. the time limit and a tracking loss discard and home;
  5. two more successes reach one demo per K -> finished; q quits;
  6. K never appears on the status line.
Prints PASS / FAIL per check; exit code 1 if any fails.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

from quest_teleop2.config import RobotCfg, TaskCfg, TeleopCfg
from quest_teleop2.quest.quest_input import DualHandInput
from quest_teleop2.robot.rtde_io import ENABLE_SERVO
from quest_teleop2.tactile.contact import FingerRegions
from quest_teleop2.tactile.glove import TactileFrame
from tunable_comp_real.admittance import AdmittanceParams
from tunable_comp_real.episodes import (FORMAT, DeletableRecorder, StiffnessRecorders, SuccessDetector,
                                        SuccessParams, demo_path)
from tunable_comp_real.hand_driver import MeasuredWujiHand2
from tunable_comp_real.hand_model import HandModel
from tunable_comp_real.session import DONE, FINISHED, RECORDING, WAITING, RealTeleopSession
from tunable_comp_real.tactile_admittance import TactileFingertipAdmittance

DT = 0.008
HOME = np.array([0.40, 0.0, 0.25, 0.0, 0.0, 0.0])    # TCP pose inside his default workspace box
results: list[tuple[bool, str]] = []


def check(ok: bool, text: str) -> None:
    results.append((bool(ok), text))


# ------------------------------------------------------------------------------------------------ #
# fakes
# ------------------------------------------------------------------------------------------------ #

class Clock:
    t = 0.0

    def __call__(self) -> float:
        return self.t


class State:
    """The fields of RobotState the session and the limiter read."""

    def __init__(self, tcp: np.ndarray, t: float) -> None:
        self.actual_q = np.zeros(6)
        self.actual_qd = np.zeros(6)
        self.target_q = np.zeros(6)
        self.actual_tcp_pose = tcp.copy()
        self.actual_tcp_speed = np.zeros(6)
        self.timestamp = t
        self.is_faulted = False
        self.robot_mode = 7
        self.mode_name = "RUNNING"
        self.safety_flags = []


class Operator:
    """What the scripted operator does this tick."""

    def __init__(self) -> None:
        self.pinch = False          # left hand
        self.press = False          # something presses the thumb and index fingertips
        self.lift_y = 0.0           # right wrist height above its start, XR y = base z
        self.tracked = True         # right hand tracked
        self.keys: list[str] = []


def hand_joints(wrist: np.ndarray, pinch: bool) -> np.ndarray:
    j = np.zeros((25, 3)) + wrist
    j[9] = j[4] + np.array([0.01 if pinch else 0.10, 0.0, 0.0])   # thumb tip (4) to index tip (9)
    return j


class FakeArm:
    """TCP follows every servo pose command; receive() advances the clock and the operator."""

    def __init__(self, clock: Clock, quest: DualHandInput, op: Operator) -> None:
        self.clock, self.quest, self.op = clock, quest, op
        self.tcp = HOME.copy()
        self.rehomes = 0
        self.script = None
        self._servo_running = True

    def receive(self) -> State:
        self.clock.t += DT
        if self.script is not None:
            try:
                next(self.script)
            except StopIteration:
                self.script = None
                self.op.keys.append("q")
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        if self.op.tracked:
            self.quest.update_hand("right", hand_joints(np.array([0.0, self.op.lift_y, 0.0]), False), quat)
        else:
            self.quest.clear_hand("right")
            self.op.tracked = True
        self.quest.update_hand("left", hand_joints(np.array([0.3, 0.0, 0.0]), self.op.pinch), quat)
        return State(self.tcp, self.clock.t)

    def command(self, pose, enable) -> None:
        if enable == ENABLE_SERVO:
            self.tcp = np.asarray(pose, dtype=float).copy()

    def hold(self) -> None:
        pass

    def rehome(self) -> State:
        self.tcp = HOME.copy()
        self.rehomes += 1
        return State(self.tcp, self.clock.t)

    def stop_servo(self) -> None:
        self._servo_running = False

    def start_servo(self) -> None:
        self._servo_running = True


class FakeServer:
    def __init__(self) -> None:
        self.banners: list[str] = []

    def banner(self, text, kind="info") -> None:
        self.banners.append(text)

    def update_hud(self, **kw) -> None:
        pass

    def haptic(self, *a) -> None:
        pass


class FakeKeyboard:
    def __init__(self, op: Operator) -> None:
        self.op = op

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def drain(self) -> list[str]:
        return []

    def poll(self) -> list[str]:
        keys, self.op.keys = self.op.keys, []
        return keys


class FakeGlove:
    """A new frame every call; `press` adds pressure on the thumb and index fingertips."""

    def __init__(self, tips: FingerRegions, op: Operator) -> None:
        self.info = {"rate_hz": 117.0, "serial": "FAKE"}
        self.lost = False
        self.op = op
        self.rng = np.random.default_rng(0)
        self.base = self.rng.uniform(0.05, 0.2, (24, 32)).astype(np.float32)
        self.grip = np.zeros((24, 32), np.float32)
        for name in ("thumb", "index"):
            self.grip[tips.masks[tips.names.index(name)]] = 0.15
        self.seq = 0

    def latest(self) -> TactileFrame:
        self.seq += 1
        p = self.base + (self.grip if self.op.press else 0) + self.rng.normal(0, 0.0005, (24, 32)).astype(np.float32)
        return TactileFrame(p, self.seq, 0, time.monotonic())


class FakeStreamHand(MeasuredWujiHand2):
    """Dry-run driver whose measured angles are the last command (a perfect hand)."""

    def __init__(self, limits) -> None:
        super().__init__(limits, dry_run=True)

    def _read_positions(self):
        return np.asarray(self.to_radians(self._value)).reshape(5, 4)


# ------------------------------------------------------------------------------------------------ #
# the scripted operator
# ------------------------------------------------------------------------------------------------ #

def operator(sess: RealTeleopSession, op: Operator, arm: FakeArm, files):
    def wait(cond, limit=3000):
        for _ in range(limit):
            if cond():
                return
            yield
        raise TimeoutError("operator script waited too long")

    def ticks(n):
        for _ in range(n):
            yield

    def gate():
        yield from wait(lambda: sess.phase == WAITING and sess.hand_filter.ready)
        op.pinch = True
        yield from ticks(25)
        op.pinch = False
        yield from ticks(5)

    def lift():
        op.press = True
        yield from ticks(25)
        for i in range(300):                          # wrist up 17 cm over 2.4 s
            op.lift_y = 0.17 * (i + 1) / 300
            yield
        yield from wait(lambda: sess.phase != RECORDING, limit=1500)

    def put_down():
        op.press = False
        for i in range(100):
            op.lift_y = 0.17 * (1 - (i + 1) / 100)
            yield

    # 1. at home
    yield from ticks(20)
    check(sess.phase == WAITING and not sess.engaged and not sess.hand_filter.ready,
          "at home: waiting, disengaged, glove learning its rest")
    op.keys.append(" ")
    yield from ticks(10)
    check(not sess.engaged, "SPACE while waiting is ignored")
    yield from wait(lambda: sess.hand_filter.ready)
    neutral = np.asarray(sess.gripper.neutral_pose)
    check(np.allclose(sess.gripper._value, neutral, atol=1e-6), "at home the hand holds the neutral pose")
    op.pinch = True
    yield from ticks(40)
    check(sess.phase == WAITING and not sess.recording, "pinch held: not started")
    op.pinch = False
    yield from ticks(5)
    check(sess.phase == RECORDING and sess.recording and sess.engaged, "pinch + release: engaged, recording")

    # 2. a successful lift
    yield from lift()
    check(sess.phase == DONE and sess.recorder.n_saved == 1, "grasp + lift 17 cm: saved by itself")
    k1 = sess.k
    with h5py.File(demo_path(files, k1), "r") as f:
        d = f["data/demo_0"]
        T = int(d.attrs["num_samples"])
        ok = (d["actions"].shape == (T, 26) and d["obs/fingertip_force"].shape == (T, 5)
              and d["extra/hand_q_cmd_ticks"].shape == (T, 5, 20)
              and float(d.attrs["max_lift_m"]) >= 0.15 and np.allclose(d.attrs["stiffness"], k1)
              and abs(float(d.attrs["duration_s"]) - T * 0.04) < 0.2)
        dur = float(d.attrs["duration_s"])
    check(ok, f"the demo: {T} samples ({dur:.1f} s), actions (T, 26), per-tick extra, K and max lift stored")
    yield from put_down()

    # 3. d, h, r, s
    op.keys.append("d")
    yield from ticks(5)
    check(sess.recorder.n_saved == 0, "d: the demo deleted")
    rehomes = arm.rehomes
    op.keys.append("h")
    yield from ticks(5)
    check(arm.rehomes == rehomes + 1 and sess.phase == WAITING, "h: homed, waiting for the next episode")
    yield from gate()
    op.keys.append("s")
    yield from ticks(10)
    check(sess.phase == RECORDING and sess.recorder.n_saved == 0, "s during an episode is ignored")
    op.keys.append("r")
    yield from ticks(5)
    check(sess.phase == WAITING and arm.rehomes == rehomes + 2 and sess.recorder.n_saved == 0
          and not sess.recording, "r: discarded and homed")

    # 4. time limit, tracking loss
    yield from gate()
    yield from wait(lambda: sess.phase != RECORDING, limit=int(sess.time_limit_s / DT) + 50)
    check(sess.phase == WAITING and arm.rehomes == rehomes + 3 and sess.recorder.n_saved == 0,
          f"time limit ({sess.time_limit_s:g} s): discarded and homed")
    yield from gate()
    yield from ticks(20)
    op.tracked = False
    yield from ticks(3)
    check(sess.phase == WAITING and arm.rehomes == rehomes + 4 and sess.recorder.n_saved == 0,
          "right hand's tracking lost: discarded and homed")

    # 5. two successes: one demo per K
    yield from gate()
    yield from lift()
    yield from put_down()
    op.keys.append("h")
    yield from ticks(5)
    yield from gate()
    yield from lift()
    check(sess.phase == FINISHED and sess.recorder.counts == {150.0: 1, 500.0: 1},
          f"one demo per K: finished ({sess.recorder.counts})")
    yield from put_down()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regions", type=Path, default=Path("config/real/glove_tip_regions.yaml"))
    ap.add_argument("--urdf", type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)
    model = HandModel(args.urdf)
    regions = args.regions
    tips = FingerRegions.load(regions)
    op = Operator()
    clock = Clock()
    quest = DualHandInput(TeleopCfg().gripper, pinch_freezes=True, drive_side="right")
    arm = FakeArm(clock, quest, op)
    gripper = FakeStreamHand((model.lower, model.upper))
    filt = TactileFingertipAdmittance(FakeGlove(tips, op), gripper, model, tips, AdmittanceParams(),
                                      regions_path=regions, ticks_per_record=5)
    task = TaskCfg(name="real_lift", starts=["home"], goals=["lift"])
    u_ref = model.to_normalised(np.tile([0.4, 0.0, 0.4, 0.4], 5))
    out = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp:
        files = Path(tmp)
        rec = StiffnessRecorders({k: DeletableRecorder(
            demo_path(files, k), task, {"format": FORMAT, "split": "train", "stiffness": k, "task": "real_lift"},
            action_hz=25.0, more_obs_keys=filt.OBS_KEYS, more_extra_keys=filt.EXTRA_KEYS)
            for k in (150.0, 500.0)}, target_per_k=1)
        sess = RealTeleopSession(
            arm=arm, quest=quest, server=FakeServer(), robot_cfg=RobotCfg(), teleop_cfg=TeleopCfg(),
            task_cfg=task, gripper=gripper, recorder=rec, hand_retargeter=lambda joints: u_ref,
            hand_filter=filt, k_values=[150.0, 500.0], success=SuccessDetector(SuccessParams(0.5)),
            rng=np.random.default_rng(0), time_limit_s=8.0, park_s=0.02,
            episode_attrs={"cube_spot": "test"}, clock=clock)
        sess.keyboard = FakeKeyboard(op)
        arm.script = operator(sess, op, arm, files)
        sess.prepare()
        with contextlib.redirect_stdout(out):
            sess.run()
    text = out.getvalue()
    status_lines = [s.split("\n")[0] for s in text.split("\r") if s.startswith("[")]
    check(len(status_lines) > 0 and not any("150" in s or "500" in s or "K =" in s for s in status_lines),
          f"K never on the status line ({len(status_lines)} lines checked)")
    print("RealTeleopSession in his TeleopSession loop, scripted operator:")
    for ok, msg in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    failures = sum(not ok for ok, _ in results)
    print(f"\n{'all checks passed' if failures == 0 else f'{failures} check(s) FAILED'}")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
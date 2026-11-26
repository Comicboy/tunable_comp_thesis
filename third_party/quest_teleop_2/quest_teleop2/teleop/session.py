"""The teleoperation control loop.

Paced by RTDE: `arm.receive()` blocks until the controller's next packet, so the
loop runs at exactly `control.frequency_hz` without sleeping. The Quest streams
at ~60 Hz into a lock-protected buffer, and the loop reads the latest sample --
so the two rates are decoupled and neither blocks the other.

`arm.command(...)` is called on EVERY tick, including holds. That is not
optional: it feeds the controller-side RTDE watchdog, which stops the servo
program if the host goes quiet.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:  # importing at runtime would pull cv2 into every teleop script
    from ..vision.camera import CameraRig

from ..config import RobotCfg, TaskCfg, TeleopCfg
from ..keyboard import KeyboardReader
from ..quest.quest_input import Event, PinchDetector, QuestInput, QuestSnapshot
from ..quest.webxr_server import WebXRServer
from ..robot.gripper import Gripper
from ..robot.rtde_io import RobotState
from ..robot.safety import SafetyAbort, SafetyLimiter
from ..robot.ur5e_arm import UR5eArm
from .retarget import WristRetargeter

_log = logging.getLogger(__name__)


class Recorder(Protocol):
    """What the session needs from a recorder. See data/recorder.py."""

    @property
    def is_recording(self) -> bool: ...

    @property
    def n_saved(self) -> int: ...

    @property
    def n_frames(self) -> int: ...

    def start_episode(self, start_idx: int, goal_idx: int) -> None: ...

    def add(self, obs: dict[str, np.ndarray], action: np.ndarray,
            extra: dict[str, np.ndarray],
            images: list[np.ndarray] | None = None) -> None: ...

    def save_episode(self, success: bool = True) -> int: ...

    def discard_episode(self) -> None: ...


@dataclass
class TaskSelection:
    """Which (start, goal) the operator is about to demonstrate."""

    task: TaskCfg
    start_idx: int = 0
    goal_idx: int = 0

    def cycle_start(self) -> None:
        self.start_idx = (self.start_idx + 1) % self.task.n_starts

    def cycle_goal(self) -> None:
        self.goal_idx = (self.goal_idx + 1) % self.task.n_goals

    def label(self) -> str:
        return f"{self.task.starts[self.start_idx]} -> {self.task.goals[self.goal_idx]}"


@dataclass
class _Pending:
    """One tick's observation, waiting for the action it produced.

    RTDE `target_q` read at tick t reflects the servoj issued at t-1, so the
    action that answers obs_t shows up in the *next* sample. Holding the
    observation back by one tick keeps (obs, action) pairs properly aligned
    instead of off by 8 ms.
    """

    obs: dict[str, np.ndarray]
    extra: dict[str, np.ndarray]
    gripper_cmd: np.ndarray  # (GRIPPER_DIM,) commanded end-effector target
    #: Camera frames grabbed at the SAME instant as `obs`, in rig order. They
    #: travel with the observation rather than being grabbed at flush time,
    #: which is one decimation window (40 ms) later -- long enough for the arm
    #: to have moved visibly, and the resulting offset would be systematic.
    images: list[np.ndarray] | None = None


class TeleopSession:
    """Runs teleoperation, optionally recording demonstrations."""

    #: Seconds to blend the fingers in when the clutch engages.
    ENGAGE_EASE_S = 0.8

    def __init__(
        self,
        arm: UR5eArm,
        quest: QuestInput,
        server: WebXRServer,
        robot_cfg: RobotCfg,
        teleop_cfg: TeleopCfg,
        task_cfg: TaskCfg,
        gripper: Gripper,
        recorder: Recorder | None = None,
        event_hook: Callable[[Event, RobotState], bool] | None = None,
        key_hook: Callable[[str, RobotState], bool] | None = None,
        home_before_episode: bool = False,
        hand_retargeter: Callable[[np.ndarray], np.ndarray] | None = None,
        cameras: "CameraRig | None" = None,
        hand_filter=None,
    ) -> None:
        self.arm = arm
        self.quest = quest
        self.server = server
        self.robot_cfg = robot_cfg
        self.teleop_cfg = teleop_cfg
        self.task = task_cfg
        self.gripper = gripper
        #: Maps the right hand's 25 WebXR joints to a normalised command of
        #: width `gripper.command_dim`. When set, the full finger pose drives
        #: the end-effector instead of the single pinch scalar, and that full
        #: command is what gets recorded as the action's gripper channel.
        self.hand_retargeter = hand_retargeter
        #: Already started by the caller. The rig holds only the newest frame
        #: per camera, so grabbing never blocks the control loop -- a tick takes
        #: whatever is latest and records how stale it was.
        self.cameras = cameras
        self.recorder = recorder
        #: Between the operator's hand reference and the hand: steps every tick, frozen or
        #: not, and adds its own observations (control/admittance.TactileAdmittance).
        self.hand_filter = hand_filter
        # Let a caller repurpose the operator's inputs without reimplementing
        # the control loop -- used by record_demos.py --teach. Returning True
        # consumes the event/key. Both exist because episode control moved to
        # the keyboard when the controller turned out to be unavailable.
        self.event_hook = event_hook
        self.key_hook = key_hook
        #: Return to the home pose automatically when an episode starts, so
        #: every demonstration begins from an identical configuration.
        self.home_before_episode = home_before_episode

        self.retargeter = WristRetargeter(teleop_cfg.retarget)
        self.pinch = PinchDetector(teleop_cfg.gripper)
        #: Fallback pose when hand tracking drops. `open_pose` is asymmetric --
        #: the joint limits are not centred on zero -- so it is asked of the
        #: gripper rather than assumed to be zeros.
        self._hand_open_pose = np.asarray(
            getattr(gripper, "open_pose", np.zeros(gripper.command_dim, dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1)
        #: Last end-effector target actually issued, and the reference it came from, which
        #: is what a freeze holds.
        self._last_gripper_cmd = self._hand_open_pose.copy()
        self._last_reference = self._hand_open_pose.copy()
        #: Countdown for `ease_gripper`. The hand maps the operator's fingers
        #: ABSOLUTELY and has no anchor -- unlike the arm, which anchors on
        #: clutch engagement and moves relatively -- so the first tick after a
        #: blocking move commands the operator's whole pose at once.
        self._ease_left = 0
        self._ease_total = 0
        self._ease_from = self._hand_open_pose.copy()
        #: Clutch last tick: the ease starts on the edge, not every tick.
        self._prev_clutch = False
        self.limiter = SafetyLimiter(robot_cfg.safety, robot_cfg.control.dt)
        self.selection = TaskSelection(task_cfg)

        self._pending: _Pending | None = None
        #: Servo ticks since the loop started; drives record decimation.
        self._tick = 0
        #: Set when a blocking call (homing) invalidated this tick's state.
        self._stale_state = False
        self._last_hud = 0.0
        self._last_print = 0.0
        self._running = False

        self.dual_hand = teleop_cfg.input.mode == "hands"
        # With a release-style clutch the arm would follow the instant tracking
        # appears, so motion stays off until the operator presses SPACE. In
        # controller mode the grip button is itself the gate, so there is
        # nothing to arm.
        self.engaged = bool(teleop_cfg.input.start_engaged) or not self.dual_hand
        self.keyboard = KeyboardReader()

    KEYS = {
        " ": "engage",
        "s": "save",
        "f": "save_failure",
        "r": "discard",
        "t": "start",
        "[": "cycle_start",
        "]": "cycle_goal",
        "a": "reanchor",
        "h": "home",
        "g": "glove_rest",
        "q": "quit",
    }

    def go_home(self) -> None:
        """Return the arm to the configured home pose and re-anchor.

        Every demonstration starting from the same configuration is what makes
        this dataset learnable: the policy only sees joint state and a task
        one-hot, so varying start poses would be an extra hidden variable it has
        to infer from nothing.
        """
        print("\n  homing...", end="", flush=True)
        self.server.update_hud(robot="homing", robot_kind="warn")
        self.server.banner("homing", "idle")
        # rehome() blocks for a second or two. Anything the loop already read
        # this tick is stale by the time it returns -- the arm is somewhere
        # else entirely -- so the tick must not be recorded.
        self._stale_state = True
        state = self.arm.rehome()
        # The TCP has moved a long way and the operator's hand has not, so drop
        # the anchor -- it is recaptured on the next enabled frame with zero
        # delta, which is what stops the arm lurching to meet the hand.
        self.retargeter.clear()
        self.limiter.reset(state.actual_tcp_pose)
        self.pinch.reset()
        print(" at home.")
        self.server.banner("at home", "ok")

    def ease_gripper(self, seconds: float) -> None:
        """Blend the next `seconds` of finger commands in from the current pose.

        Call after any blocking move that left the hand somewhere the operator's
        own hand is not. Without it the first tick of the resumed loop commands
        the full tracked pose, which the hand's 200 ms low-pass turns into a
        snap; with it the same motion is spread over a chosen time.

        The blended value is what gets recorded, because it is what was
        commanded -- so a replay reproduces the ease rather than the snap.
        """
        self._ease_total = max(1, int(seconds / self.robot_cfg.control.dt))
        self._ease_left = self._ease_total
        # From where the hand measurably is: a parking ramp moves it outside
        # this loop, so the last command is not where it sits.
        try:
            self._ease_from = np.asarray(
                self.gripper.state(), dtype=np.float64).reshape(-1).copy()
        except Exception:  # noqa: BLE001 -- a state read must never stop teleop
            self._ease_from = np.asarray(self._last_gripper_cmd, dtype=np.float64).copy()
        if self.hand_filter is not None:
            # The ease runs on the reference, before the filter: start from the reference that
            # gives the current command, or the filter's deviation would count twice.
            self._ease_from = self._ease_from - (self._last_gripper_cmd - self._last_reference)

    def _handle_keys(self, state: RobotState) -> None:
        for key in self.keyboard.poll():
            action = self.KEYS.get(key)
            if action is None:
                continue
            if self.key_hook is not None and self.key_hook(action, state):
                continue
            if action == "engage":
                self.engaged = not self.engaged
                state = "ENGAGED" if self.engaged else "disengaged"
                self.server.banner(state, "ok" if self.engaged else "idle")
                _log.info("Motion %s", state)
            elif action == "quit":
                _log.info("Quit requested.")
                self._running = False
            elif action == "reanchor":
                self.retargeter.clear()
                self.server.banner("re-anchored")
            elif action == "home":
                self.go_home()
            elif action == "glove_rest":
                if self.hand_filter is None:
                    continue
                if self.recorder is not None and self.recorder.is_recording:
                    self.server.banner("stop episode first", "bad")
                    continue
                self.hand_filter.learn_rest()
                self.server.banner("glove: learning rest, touch nothing")
            elif action == "start":
                if self.home_before_episode:
                    self.go_home()
                self._start_episode()
            elif action == "save":
                self._save_episode()
            elif action == "save_failure":
                # A failed attempt is data, not waste: an advantage-weighted
                # update needs negatives to have anything to contrast a good
                # action against. Discard is for a spoiled take -- tracking
                # dropped, wrong object, operator error -- not for a genuine
                # miss.
                self._save_episode(success=False)
            elif action == "discard":
                self._discard_episode()
            elif action == "cycle_start":
                if self.recorder is not None and self.recorder.is_recording:
                    self.server.banner("stop episode first", "bad")
                    continue
                self.selection.cycle_start()
                self.server.banner(self.selection.label())
            elif action == "cycle_goal":
                if self.recorder is not None and self.recorder.is_recording:
                    self.server.banner("stop episode first", "bad")
                    continue
                self.selection.cycle_goal()
                self.server.banner(self.selection.label())

    # -- events ----------------------------------------------------------- #

    def _handle_events(self, snap: QuestSnapshot, state: RobotState) -> None:
        for event in snap.events:
            if self.event_hook is not None and self.event_hook(event, state):
                continue
            if event is Event.CYCLE_START:
                if self.recorder is not None and self.recorder.is_recording:
                    self.server.banner("stop episode first", "bad")
                    continue
                self.selection.cycle_start()
                self.server.banner(self.selection.label())
            elif event is Event.CYCLE_GOAL:
                if self.recorder is not None and self.recorder.is_recording:
                    self.server.banner("stop episode first", "bad")
                    continue
                self.selection.cycle_goal()
                self.server.banner(self.selection.label())
            elif event is Event.REANCHOR:
                self.retargeter.clear()
                self.server.banner("re-anchored")
            elif event is Event.START_EPISODE:
                self._start_episode()
            elif event is Event.SAVE:
                self._save_episode()
            elif event is Event.DISCARD:
                self._discard_episode()

    def _start_episode(self) -> None:
        if self.recorder is None:
            self.server.banner("not recording", "bad")
            return
        if self.recorder.is_recording:
            self.server.banner("already recording", "bad")
            return
        if self.hand_filter is not None and not self.hand_filter.ready:
            self.server.banner("glove not ready", "bad")
            _log.error("Refusing to start: the glove is%s", self.hand_filter.status())
            return
        # Prove the cameras are delivering BEFORE the operator performs the
        # demonstration. Finding out at save time means the demo is already
        # done and wasted, and the headset gives no other way to notice.
        if self.cameras is not None:
            try:
                self.cameras.grab()
            except RuntimeError as exc:
                self.server.banner("camera not ready", "bad")
                _log.error("Refusing to start: %s", exc)
                return
        self._pending = None
        # Start every episode on a recorded tick, so the first action lands one
        # full period after the first observation like all the others.
        self._tick = 0
        self.recorder.start_episode(self.selection.start_idx, self.selection.goal_idx)
        if self.hand_filter is not None:
            self.recorder.set_episode_attrs(**self.hand_filter.attrs())
        self.server.banner(f"REC {self.selection.label()}", "ok")
        self.server.haptic(0.5)
        _log.info("Episode started: %s", self.selection.label())

    def _save_episode(self, success: bool = True) -> None:
        if self.recorder is None or not self.recorder.is_recording:
            self.server.banner("no episode", "bad")
            return
        frames = self.recorder.n_frames
        if frames < 10:
            self.server.banner("too short - discarded", "bad")
            self.recorder.discard_episode()
            _log.warning("Episode discarded: only %d frames.", frames)
        else:
            index = self.recorder.save_episode(success)
            label = "SAVED" if success else "SAVED (failure)"
            self.server.banner(f"{label} #{index}", "ok" if success else "warn")
            self.server.haptic(0.9 if success else 0.5)
            _log.info("Episode %d saved as %s (%d frames).", index,
                      "success" if success else "FAILURE", frames)
        self._pending = None

    def _discard_episode(self) -> None:
        if self.recorder is None or not self.recorder.is_recording:
            self.server.banner("no episode", "bad")
            return
        self.recorder.discard_episode()
        self.server.banner("discarded", "bad")
        self.server.haptic(0.3)
        _log.info("Episode discarded.")
        self._pending = None

    # -- recording -------------------------------------------------------- #

    def _observe(self, state: RobotState, snap: QuestSnapshot,
                 gripper_cmd: np.ndarray) -> _Pending:
        obs = {
            "joint_pos": state.actual_q.astype(np.float32),
            "joint_vel": state.actual_qd.astype(np.float32),
            # MEASURED, not commanded. On real hardware the fingers lag and
            # stall against objects; recording the command here instead would
            # teach a policy that its commands are always achieved.
            "gripper": np.asarray(self.gripper.state(), dtype=np.float32).reshape(-1),
            "start_onehot": self.task.one_hot("start", self.selection.start_idx),
            "goal_onehot": self.task.one_hot("goal", self.selection.goal_idx),
        }
        hand = snap.hand
        extra = {
            "tcp_pose": state.actual_tcp_pose.astype(np.float32),
            "tcp_speed": state.actual_tcp_speed.astype(np.float32),
            "timestamp": np.array([state.timestamp], dtype=np.float64),
            "pinch_dist": np.array(
                [hand.pinch_distance if hand is not None else np.nan], dtype=np.float32
            ),
            "clutch": np.array([1.0 if snap.clutch else 0.0], dtype=np.float32),
            "wrist_pose": (
                np.concatenate([hand.wrist_pos, hand.wrist_quat]).astype(np.float32)
                if hand is not None
                else np.full(7, np.nan, dtype=np.float32)
            ),
            "hand_joints": (
                hand.joints.astype(np.float32)
                if hand is not None
                else np.full((25, 3), np.nan, dtype=np.float32)
            ),
        }
        if self.hand_filter is not None:
            obs.update(self.hand_filter.observation())
            extra.update(self.hand_filter.extra())
        images = None
        if self.cameras is not None:
            # Grabbed HERE, with the observation, not at flush time one
            # decimation window later. `grab` returns the newest frame each
            # camera holds and never blocks.
            images, worst_age = self.cameras.grab()
            extra["image_age"] = np.array([worst_age], dtype=np.float32)
        return _Pending(obs=obs, extra=extra, gripper_cmd=gripper_cmd, images=images)

    def _flush_pending(self, state: RobotState) -> None:
        """Pair the previous tick's observation with the action it produced."""
        if self._pending is None or self.recorder is None or not self.recorder.is_recording:
            return
        action = np.concatenate([
            state.target_q.astype(np.float32),
            np.asarray(self._pending.gripper_cmd, dtype=np.float32).reshape(-1),
        ])
        self._pending.extra["target_tcp_pose"] = (
            self.limiter.last_command
            if self.limiter.last_command is not None
            else state.actual_tcp_pose
        ).astype(np.float32)
        self.recorder.add(self._pending.obs, action, self._pending.extra,
                          images=self._pending.images)

    # -- HUD -------------------------------------------------------------- #

    def _update_hud(self, snap: QuestSnapshot, verdict_reason: str) -> None:
        now = time.monotonic()
        if now - self._last_hud < 0.2:  # 5 Hz is plenty for text
            return
        self._last_hud = now
        recording = self.recorder is not None and self.recorder.is_recording
        if recording:
            episode, kind = f"REC {self.recorder.n_frames} frames", "ok"
        elif self.recorder is not None:
            episode, kind = "idle (trigger to start)", "idle"
        else:
            episode, kind = "not recording", "idle"
        self.server.update_hud(
            task=self.selection.label(),
            episode=episode,
            episode_kind=kind,
            robot=verdict_reason or "tracking",
            robot_kind="warn" if verdict_reason else "ok",
            saved=str(self.recorder.n_saved) if self.recorder is not None else "-",
        )

    def _print_status(self, snap: QuestSnapshot, verdict_reason: str) -> None:
        now = time.monotonic()
        if now - self._last_print < 0.5:
            return
        self._last_print = now
        hand = "ok" if snap.hand is not None else "LOST"
        if self.dual_hand:
            clutch = "MOVE" if snap.clutch else ("FRZN" if snap.gate_pinching else "off ")
        else:
            clutch = "HELD" if snap.clutch else "----"
        rec = (
            f"REC {self.recorder.n_frames:5d}"
            if self.recorder is not None and self.recorder.is_recording
            else "        idle"
        )
        line = (f"[{self.selection.label():>12}] hand:{hand:4} clutch:{clutch} {rec}  "
                f"{verdict_reason:<44}{self._amps()}"
                f"{self.hand_filter.status() if self.hand_filter is not None else ''}")
        # One line only: a status that wraps cannot be overwritten and scrolls instead.
        print("\r" + line[:shutil.get_terminal_size((120, 24)).columns - 1], end="", flush=True)

    def _amps(self) -> str:
        """Peak measured current per finger against the limit, for the status line."""
        read = getattr(self.gripper, "currents", None)
        if not callable(read):
            return ""
        if not (self.recorder is not None and self.recorder.is_recording):
            self.gripper.state()        # a recording reads the hand itself every tick
        amps = read()
        if amps is None:
            return ""
        limit = getattr(self.gripper, "current_limit", None)
        peaks = " ".join(f"{n}{v:4.2f}" for n, v in zip("TIMRP", np.nanmax(np.abs(amps), axis=1)))
        return f" A {peaks}" + (f" /{limit:.1f}" if limit else "")

    # -- main loop -------------------------------------------------------- #

    def run(self) -> None:
        """Block until Ctrl-C or a safety abort."""
        self._running = True
        dt = self.robot_cfg.control.dt
        self.keyboard.start()
        self.keyboard.drain()
        _log.info("Teleop loop running at %.0f Hz.", 1.0 / dt)
        if self.dual_hand and not self.engaged:
            _log.info("Motion is DISENGAGED. Press SPACE to engage.")
        try:
            while self._running:
                state = self.arm.receive()
                self.limiter.check_robot(state)

                # Actions are recorded every Nth servo tick: the loop stays at
                # frequency_hz, the dataset gets a uniform record_hz.
                record_tick = self._tick % self.robot_cfg.control.record_decimation == 0
                self._tick += 1

                # Pair the previous observation with the action it produced,
                # before overwriting anything with this tick's data.
                if record_tick:
                    self._flush_pending(state)

                snap = self.quest.snapshot(engaged=self.engaged)
                self._handle_keys(state)
                self._handle_events(snap, state)
                if not self._running:
                    break

                # A blocking move ran inside this tick, so `state` is stale and
                # must not reach the retargeter: it anchors on actual_tcp_pose,
                # and a stale anchor drives the arm back where it started.
                # hold() rather than `continue`, so the heartbeat is not skipped.
                if self._stale_state:
                    self._stale_state = False
                    self._pending = None
                    self.arm.hold()
                    continue

                # Anchor on clutch engagement, and re-anchor after any tracking
                # dropout: the hand may have moved while it was invisible, so
                # resuming from the old anchor would command a jump.
                if snap.clutch and snap.hand is not None:
                    if not self.retargeter.anchored:
                        self.retargeter.anchor(
                            snap.hand.wrist_pos, snap.hand.wrist_quat, state.actual_tcp_pose
                        )
                        _log.debug("Anchored at TCP %s", np.round(state.actual_tcp_pose, 3))
                    desired = self.retargeter.target(snap.hand.wrist_pos, snap.hand.wrist_quat)
                else:
                    self.retargeter.clear()
                    desired = None

                verdict = self.limiter.filter(
                    desired, state, motion_enabled=snap.clutch, data_age=snap.hand_age
                )
                self.arm.command(verdict.command, verdict.enable)

                # Pinch is still tracked -- it is recorded in extra/pinch_dist
                # and remains a useful scalar summary of grasp intent -- but it
                # is no longer the gripper command when a hand is attached.
                closed = self.pinch.update(
                    snap.hand.pinch_distance if snap.hand is not None else float("inf")
                )

                # Which branch issues a command is deliberately unchanged from
                # the arm+hand teleop path that was validated on hardware. The
                # only addition is remembering the value, so the recorder can
                # log what is currently commanded during a freeze -- when no
                # new command is sent, and the hand is holding its last target.
                send = True
                if self.hand_retargeter is None:
                    # No hand: broadcast the pinch across the channel so the
                    # recorded width still matches the schema. Crude, and only
                    # meaningful as a placeholder for "no end-effector".
                    gripper_cmd = np.full(
                        self.gripper.command_dim, 1.0 if closed else 0.0, dtype=np.float64
                    )
                elif snap.hand is not None and snap.clutch:
                    # Gated on the same clutch as the arm. The fingers map
                    # absolutely with no anchor, so ease on every rising edge.
                    if not self._prev_clutch:
                        self.ease_gripper(self.ENGAGE_EASE_S)
                    gripper_cmd = np.asarray(
                        self.hand_retargeter(snap.hand.joints), dtype=np.float64
                    ).reshape(-1)
                    if self._ease_left > 0:
                        blend = 1.0 - self._ease_left / self._ease_total
                        gripper_cmd = self._ease_from * (1.0 - blend) + gripper_cmd * blend
                        self._ease_left -= 1
                elif snap.hand is None:
                    # Tracking lost. Open rather than hold: a hand left gripping
                    # something it cannot see is worse than one that lets go.
                    gripper_cmd = np.asarray(self._hand_open_pose, dtype=np.float64).reshape(-1)
                else:
                    # Frozen with tracking good: the reference holds, and without a
                    # hand filter no command is issued, so the hand holds its last target.
                    gripper_cmd = self._last_reference
                    send = False

                self._last_reference = gripper_cmd
                if self.hand_filter is not None:
                    gripper_cmd = self.hand_filter.step(gripper_cmd)
                    send = True
                if send:
                    self.gripper.command(gripper_cmd)
                self._last_gripper_cmd = gripper_cmd
                self._prev_clutch = bool(snap.clutch)

                # `_stale_state` is handled at the top of the tick now, which
                # both skips the corrupt first frame this used to guard against
                # AND keeps the stale pose away from the retargeter.
                if record_tick and self.recorder is not None and self.recorder.is_recording:
                    self._pending = self._observe(state, snap, gripper_cmd)
                elif record_tick:
                    self._pending = None

                self._update_hud(snap, verdict.reason)
                self._print_status(snap, verdict.reason)

        except KeyboardInterrupt:
            print()
            _log.info("Interrupted by operator.")
        except SafetyAbort as exc:
            print()
            _log.error("SAFETY ABORT: %s", exc)
            raise
        finally:
            self._running = False
            # Restore the terminal before anything else prints, or the shell is
            # left in cbreak mode after an abort.
            self.keyboard.stop()
            print()
            if self.recorder is not None and self.recorder.is_recording:
                _log.warning("Session ended mid-episode; discarding it.")
                self.recorder.discard_episode()
            _log.info("Safety: %s", self.limiter.stats.summary())

    def stop(self) -> None:
        self._running = False

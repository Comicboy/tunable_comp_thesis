"""Quest input state, shared between the WebSocket thread and the control loop.

Threading model is the same as _TeleopState in the Isaac quest3_device.py: the
aiohttp thread writes, the control loop reads, a lock guards the handoff, and
every sample carries a `time.monotonic()` stamp so the reader can tell how stale
it is. Staleness matters more here than in sim -- hand tracking drops out
routinely, and the safety layer keys off exactly this number.

Button *presses* are edge-detected on the writer side and queued, because the
control loop polls at 125 Hz while the headset sends at ~60 Hz: a level-only
API would let a short press be missed or counted twice.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..config import ControllerCfg, GripperCfg  # noqa: F401

# WebXR hand joints, in the order defined by the spec. The client indexes
# XRHand by these names rather than relying on iteration order, so these indices
# are guaranteed rather than assumed.
JOINT_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb-metacarpal",
    "thumb-phalanx-proximal",
    "thumb-phalanx-distal",
    "thumb-tip",
    "index-finger-metacarpal",
    "index-finger-phalanx-proximal",
    "index-finger-phalanx-intermediate",
    "index-finger-phalanx-distal",
    "index-finger-tip",
    "middle-finger-metacarpal",
    "middle-finger-phalanx-proximal",
    "middle-finger-phalanx-intermediate",
    "middle-finger-phalanx-distal",
    "middle-finger-tip",
    "ring-finger-metacarpal",
    "ring-finger-phalanx-proximal",
    "ring-finger-phalanx-intermediate",
    "ring-finger-phalanx-distal",
    "ring-finger-tip",
    "pinky-finger-metacarpal",
    "pinky-finger-phalanx-proximal",
    "pinky-finger-phalanx-intermediate",
    "pinky-finger-phalanx-distal",
    "pinky-finger-tip",
)
N_JOINTS = len(JOINT_NAMES)
WRIST = JOINT_NAMES.index("wrist")
THUMB_TIP = JOINT_NAMES.index("thumb-tip")
INDEX_TIP = JOINT_NAMES.index("index-finger-tip")


class Event(str, Enum):
    """Semantic operator actions, decoded from controller buttons."""

    SAVE = "save"
    DISCARD = "discard"
    START_EPISODE = "start_episode"
    REANCHOR = "reanchor"
    CYCLE_START = "cycle_start"
    CYCLE_GOAL = "cycle_goal"


@dataclass
class HandSample:
    """One right-hand frame: all 25 joints valid, plus the wrist orientation."""

    joints: np.ndarray  # (25, 3) metres, in the XR reference space
    wrist_quat: np.ndarray  # (4,) xyzw

    #: Per-joint orientations, (25, 4) xyzw, in the same space as `joints`.
    #: Optional because the offline fakes and older clients only send positions.
    #: Carries strictly more information than `joints`: the spec fixes each joint
    #: space as -Z along the bone and -Y out of the palm, so a relative rotation
    #: between two joints is a signed anatomical angle. The thumb's saddle CMC
    #: needs this -- its two DoF are not recoverable from positions alone.
    quats: np.ndarray | None = None

    @property
    def wrist_pos(self) -> np.ndarray:
        return self.joints[WRIST]

    @property
    def pinch_distance(self) -> float:
        """Thumb-tip to index-tip distance, metres."""
        return float(np.linalg.norm(self.joints[THUMB_TIP] - self.joints[INDEX_TIP]))


@dataclass
class ControllerSample:
    """One left-controller frame."""

    position: np.ndarray  # (3,)
    quat: np.ndarray  # (4,) xyzw
    pressed: np.ndarray  # (n_buttons,) bool
    values: np.ndarray  # (n_buttons,) float 0..1
    axes: np.ndarray  # (n_axes,) float -1..1

    def button(self, index: int) -> bool:
        return bool(self.pressed[index]) if 0 <= index < len(self.pressed) else False

    def value(self, index: int) -> float:
        return float(self.values[index]) if 0 <= index < len(self.values) else 0.0

    def axis(self, index: int) -> float:
        return float(self.axes[index]) if 0 <= index < len(self.axes) else 0.0


@dataclass
class QuestSnapshot:
    """A consistent view for one control tick."""

    hand: HandSample | None
    controller: ControllerSample | None
    hand_age: float
    controller_age: float
    events: list[Event] = field(default_factory=list)

    #: True when the arm should follow the tracked hand this tick. Computed by
    #: the input backend, and false whenever the gating input is missing or
    #: stale -- losing an input must never leave the arm enabled.
    clutch: bool = False

    #: Populated in dual-hand mode only.
    #: The hand that gates motion -- the one you are NOT driving with.
    gate_hand: HandSample | None = None
    gate_pinching: bool = False
    #: Why motion is gated off, for the HUD and the terminal status line.
    gate_reason: str = ""


class PinchDetector:
    """Thumb-index pinch with hysteresis.

    A single threshold chatters when the measured distance sits on the boundary.
    That is harmless in simulation and is not on real hardware, so closing and
    opening use different thresholds and the state latches between them.
    """

    def __init__(self, cfg: GripperCfg) -> None:
        self._close = cfg.pinch_close_m
        self._open = cfg.pinch_open_m
        self._closed = False

    def update(self, distance: float) -> bool:
        if self._closed:
            if distance > self._open:
                self._closed = False
        elif distance < self._close:
            self._closed = True
        return self._closed

    @property
    def closed(self) -> bool:
        return self._closed

    def reset(self) -> None:
        self._closed = False


class DualHandInput:
    """Both hands tracked, no controller.

    This is the configuration Quest Browser actually offers: two 'oculus-hand'
    input sources and no Touch controller in the same WebXR session.

    One hand DRIVES -- its wrist pose moves the TCP and its fingers drive the
    robot hand. The other GATES -- its pinch starts or stops motion. Which is
    which is set by `drive_side`, so a left Wuji hand can be driven by your left
    hand with your right doing the gating.
    """

    def __init__(self, gripper_cfg: GripperCfg, pinch_freezes: bool = True,
                 drive_side: str = "right") -> None:
        if drive_side not in ("left", "right"):
            raise ValueError(f"drive_side must be 'left' or 'right', got {drive_side!r}")
        self._lock = threading.Lock()
        self.drive_side = drive_side
        self.gate_side = "left" if drive_side == "right" else "right"
        self._drive: HandSample | None = None
        self._gate: HandSample | None = None
        self._drive_stamp = 0.0
        self._gate_stamp = 0.0
        self._gate_pinch = PinchDetector(gripper_cfg)
        self._pinch_freezes = pinch_freezes
        self.frames_drive = 0
        self.frames_gate = 0

    # -- writer side (WebSocket thread) ----------------------------------- #

    def update_hand(
        self,
        handedness: str,
        joints: np.ndarray,
        wrist_quat: np.ndarray,
        quats: np.ndarray | None = None,
    ) -> None:
        sample = HandSample(
            joints=np.asarray(joints, dtype=np.float64).reshape(N_JOINTS, 3),
            wrist_quat=np.asarray(wrist_quat, dtype=np.float64).reshape(4),
            quats=(None if quats is None
                   else np.asarray(quats, dtype=np.float64).reshape(N_JOINTS, 4)),
        )
        with self._lock:
            if handedness == self.drive_side:
                self._drive = sample
                self._drive_stamp = time.monotonic()
                self.frames_drive += 1
            else:
                self._gate = sample
                self._gate_stamp = time.monotonic()
                self.frames_gate += 1
                # Hysteresis lives here so a pinch hovering at the threshold
                # cannot make the arm stutter between moving and frozen.
                self._gate_pinch.update(sample.pinch_distance)

    def clear_hand(self, handedness: str | None = None) -> None:
        with self._lock:
            if handedness in (None, self.drive_side):
                self._drive = None
            if handedness in (None, self.gate_side):
                self._gate = None
                self._gate_pinch.reset()

    # -- reader side (control loop) --------------------------------------- #

    def snapshot(self, engaged: bool = True) -> QuestSnapshot:
        now = time.monotonic()
        with self._lock:
            drive, gate = self._drive, self._gate
            drive_age = now - self._drive_stamp if self._drive_stamp else float("inf")
            gate_age = now - self._gate_stamp if self._gate_stamp else float("inf")
            gate_pinching = self._gate_pinch.closed
            freezes = self._pinch_freezes
            side = self.gate_side

        # A stale gate hand must not silently mean "not pinching" when a pinch
        # is what stops the arm -- fail toward frozen, not toward moving.
        gate_fresh = gate is not None and gate_age < 0.3
        if freezes:
            frozen = gate_pinching or not gate_fresh
            moving = engaged and not frozen
            reason = f"reposition ({side} pinch)" if gate_pinching else (
                "" if gate_fresh else f"{side} hand not tracked"
            )
        else:
            moving = engaged and gate_fresh and gate_pinching
            reason = "" if moving else f"hold {side} pinch to move"

        snap = QuestSnapshot(
            hand=drive,
            controller=None,
            hand_age=drive_age,
            controller_age=gate_age,
            events=[],
        )
        snap.clutch = bool(moving)
        snap.gate_hand = gate
        snap.gate_pinching = bool(gate_pinching)
        snap.gate_reason = "" if moving else (reason or "disengaged")
        return snap


class QuestInput:
    """Lock-protected latest-sample buffer with an edge-detected event queue."""

    #: Thumbstick deflection that counts as a flick, and the value it must
    #: return within before another flick is accepted.
    FLICK_ON = 0.7
    FLICK_OFF = 0.3

    def __init__(self, cfg: ControllerCfg) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        self._hand: HandSample | None = None
        self._controller: ControllerSample | None = None
        self._hand_stamp = 0.0
        self._controller_stamp = 0.0
        self._events: deque[Event] = deque(maxlen=64)
        self._prev_pressed: np.ndarray | None = None
        self._stick_latched_x = False
        self._stick_latched_y = False
        self.frames_hand = 0
        self.frames_controller = 0

    # -- writer side (WebSocket thread) ----------------------------------- #

    def update_hand(
        self,
        joints: np.ndarray,
        wrist_quat: np.ndarray,
        quats: np.ndarray | None = None,
    ) -> None:
        with self._lock:
            self._hand = HandSample(
                joints=np.asarray(joints, dtype=np.float64).reshape(N_JOINTS, 3),
                wrist_quat=np.asarray(wrist_quat, dtype=np.float64).reshape(4),
                quats=(None if quats is None
                       else np.asarray(quats, dtype=np.float64).reshape(N_JOINTS, 4)),
            )
            self._hand_stamp = time.monotonic()
            self.frames_hand += 1

    def update_controller(
        self,
        position: np.ndarray,
        quat: np.ndarray,
        pressed: np.ndarray,
        values: np.ndarray,
        axes: np.ndarray,
    ) -> None:
        sample = ControllerSample(
            position=np.asarray(position, dtype=np.float64).reshape(3),
            quat=np.asarray(quat, dtype=np.float64).reshape(4),
            pressed=np.asarray(pressed, dtype=bool),
            values=np.asarray(values, dtype=np.float64),
            axes=np.asarray(axes, dtype=np.float64),
        )
        with self._lock:
            self._decode_events(sample)
            self._controller = sample
            self._controller_stamp = time.monotonic()
            self.frames_controller += 1

    def _decode_events(self, sample: ControllerSample) -> None:
        """Rising-edge detection. Caller must hold the lock."""
        b = self._cfg.buttons
        prev = self._prev_pressed
        current = sample.pressed.copy()

        if prev is not None and len(prev) == len(current):
            rising = current & ~prev
            for index, event in (
                (b.a, Event.SAVE),
                (b.b, Event.DISCARD),
                (b.trigger, Event.START_EPISODE),
                (b.thumbstick, Event.REANCHOR),
            ):
                if 0 <= index < len(rising) and rising[index]:
                    self._events.append(event)
        self._prev_pressed = current

        # Thumbstick flicks cycle the task selection. Latched so that holding
        # the stick over produces one step, not a stream of them.
        x = sample.axis(self._cfg.axes.thumbstick_x)
        y = sample.axis(self._cfg.axes.thumbstick_y)
        if abs(x) > self.FLICK_ON and not self._stick_latched_x:
            self._events.append(Event.CYCLE_START)
            self._stick_latched_x = True
        elif abs(x) < self.FLICK_OFF:
            self._stick_latched_x = False
        if abs(y) > self.FLICK_ON and not self._stick_latched_y:
            self._events.append(Event.CYCLE_GOAL)
            self._stick_latched_y = True
        elif abs(y) < self.FLICK_OFF:
            self._stick_latched_y = False

    def clear_hand(self) -> None:
        """Called when the client reports the hand is not fully tracked."""
        with self._lock:
            self._hand = None

    # -- reader side (control loop) --------------------------------------- #

    def snapshot(self, engaged: bool = True) -> QuestSnapshot:
        """Consistent view for one tick. Drains the event queue.

        `engaged` keeps the signature identical to DualHandInput so the control
        loop never has to know which backend it is talking to.
        """
        now = time.monotonic()
        with self._lock:
            events = list(self._events)
            self._events.clear()
            controller = self._controller
            snap = QuestSnapshot(
                hand=self._hand,
                controller=controller,
                hand_age=now - self._hand_stamp if self._hand_stamp else float("inf"),
                controller_age=(
                    now - self._controller_stamp if self._controller_stamp else float("inf")
                ),
                events=events,
            )
        # The clutch is a level, not an event, and must read False whenever the
        # controller sample is missing or stale.
        held = bool(
            controller is not None
            and snap.controller_age < 0.2
            and controller.value(self._cfg.buttons.squeeze) >= self._cfg.press_threshold
        )
        snap.clutch = held and engaged
        snap.gate_reason = "" if snap.clutch else ("disengaged" if held else "clutch released")
        return snap

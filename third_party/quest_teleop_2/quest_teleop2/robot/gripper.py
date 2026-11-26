"""End-effector abstraction.

No gripper is mounted yet, so `NullGripper` is what actually runs. The pinch
command is still computed, recorded and plotted throughout -- so the dataset
schema is already final, and the operator's grasp intent is captured even though
nothing acts on it.

That matters for the Wuji hand later. The WebXR client already streams all 25
right-hand joints, and wuji_direct_retargeting.py in the Isaac repo already maps
exactly those joints to 20 Wuji joint targets. Adding the hand means writing a
`WujiHand2Gripper` here and widening `command_dim` from 1 to 20 -- the client,
the clutch, the safety layer and the episode controls do not change.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

import numpy as np

from ..data.schema import GRIPPER_DIM

_log = logging.getLogger(__name__)


class Gripper(ABC):
    """A commandable end-effector."""

    #: Width of this gripper's slice of the action vector.
    command_dim: int = 1

    @abstractmethod
    def command(self, value: np.ndarray) -> None:
        """Send a command. `value` has length `command_dim`, normalised 0=open, 1=closed."""

    @abstractmethod
    def state(self) -> np.ndarray:
        """Current state, length `command_dim`, same normalisation as `command`."""

    def connect(self) -> None:
        """Open any connection this gripper needs. Default: nothing to do."""

    def disconnect(self) -> None:
        """Release resources. Default: nothing to do."""

    def __enter__(self) -> Gripper:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.disconnect()


class NullGripper(Gripper):
    """No hardware. Records the commanded value and reports it straight back.

    Because `state()` echoes the command, recorded observations look exactly as
    they will once real hardware is attached, which keeps the dataset schema and
    the training code identical across that change.

    Its width is `GRIPPER_DIM` -- the schema's, not 1 -- for exactly that
    reason: the flat observation layout is a static contract that the recorder,
    the torch Dataset and the policy all compile against, so a stand-in that
    silently narrowed it would produce datasets the models cannot load.
    """

    command_dim = GRIPPER_DIM

    def __init__(self) -> None:
        self._value = np.zeros(self.command_dim, dtype=np.float64)
        self._warned = False

    def command(self, value: np.ndarray) -> None:
        v = np.asarray(value, dtype=np.float64).reshape(self.command_dim)
        self._value = np.clip(v, 0.0, 1.0)
        if not self._warned:
            _log.info("NullGripper: pinch is being recorded but no hardware is driven.")
            self._warned = True

    def state(self) -> np.ndarray:
        return self._value.copy()

    def __str__(self) -> str:
        return "NullGripper (no end-effector mounted; pinch recorded only)"


#: Abduction joint within each finger's four. 0 rad is mid-range, not
#: normalised 0: (-0.698, +0.698) on the fingers, (-1.484, +0.698) on the
#: thumb, where this index is opposition instead.
ABDUCTION_JOINT = 1

#: Flexion angle for `neutral_pose`, radians per joint. ~11 degrees at each of
#: the three flexion joints, so ~34 degrees of curl along a finger.
NEUTRAL_FLEXION_RAD = 0.20

#: Hard lower bound in radians, applied by raising `self._lower` so a backward
#: angle is unrepresentable in the 0..1 contract every caller speaks. The
#: shipped model allows -1.047 rad of hyperextension on every flexion joint,
#: which is a simulation range, not a mechanical limit.
#:
#: `-inf` keeps the model's range, for the axes where negative is ordinary
#: travel: all five abductions, and the thumb's saddle flexion.
_NO_FLOOR = -np.inf
MIN_JOINT_RAD = np.array([
    # J0 flex     J1 abd    J2 pip    J3 dip
    [_NO_FLOOR, _NO_FLOOR,      0.0,      0.0],   # thumb (J0/J1 are the saddle)
    [      0.0, _NO_FLOOR,      0.0,      0.0],   # index
    [      0.0, _NO_FLOOR,      0.0,      0.0],   # middle
    [      0.0, _NO_FLOOR,      0.0,      0.0],   # ring
    [      0.0, _NO_FLOOR,      0.0,      0.0],   # pinky
])


class NormalisedHand(Gripper):
    """Shared 0..1 <-> radians contract for both Wuji Hand generations.

    Commands are normalised per joint (0 = the joint's lower limit, 1 = its
    upper), not radians. That keeps every gripper in this codebase on the same
    0..1 contract, keeps recorded actions bounded for training, and confines
    knowledge of the URDF limits to these classes.

    The limits are NOT shared between generations: all 20 joints differ, by up
    to 1.32 rad, so the same normalised number is a different angle on each
    hand. A dataset recorded on one cannot be replayed on the other without
    going through radians. `urdf_limits` reads whichever URDF applies.
    """

    command_dim = 20

    def __init__(self, limits: tuple[np.ndarray, np.ndarray]) -> None:
        model_lower, self._upper = (np.asarray(x, dtype=np.float64).reshape(5, 4)
                                    for x in limits)
        # Raise the floor BEFORE deriving the span, so a backward angle cannot
        # be expressed as a normalised command at all. See `MIN_JOINT_RAD`.
        self._model_lower = model_lower
        self._lower = np.maximum(model_lower, MIN_JOINT_RAD)
        if np.any(self._lower > self._upper):
            bad = np.argwhere(self._lower > self._upper)
            raise ValueError(
                f"straight (0 rad) is above the upper limit for joint(s) "
                f"{bad.tolist()}; these limits are not a right-way-up hand model"
            )
        self._span = np.where(self._upper - self._lower > 1e-9,
                              self._upper - self._lower, 1.0)
        #: Where the hand is, normalised. Seeded to the flat-open pose, NOT to
        #: normalised zero -- with the floor applied they coincide, but a
        #: subclass given an unfloored model must still never default to "every
        #: joint at its lower limit", which is a fully back-bent hand.
        self._value = self.open_pose
        #: Commands whose radians had to be clamped at the floor. Counted, not
        #: silent: steady growth means something upstream is asking for
        #: hyperextension every frame.
        self.floored_frames = 0

    # -- unit conversion -------------------------------------------------- #

    def to_radians(self, value: np.ndarray) -> np.ndarray:
        """(20,) or (5,4) normalised -> (5, 4) radians, clipped to the limits."""
        v = np.asarray(value, dtype=np.float64).reshape(5, 4)
        return np.clip(self._lower + v * self._span, self._lower, self._upper)

    def to_normalised(self, radians: np.ndarray) -> np.ndarray:
        """(5, 4) radians -> (20,) in 0..1."""
        r = np.asarray(radians, dtype=np.float64).reshape(5, 4)
        return np.clip((r - self._lower) / self._span, 0.0, 1.0).reshape(-1)

    def enforce_floor(self, radians: np.ndarray) -> np.ndarray:
        """Final gate before anything is published: never past straight.

        A command asking a finger to bend backwards is not executed as asked --
        the joint is held at its floor instead. Redundant with the raised
        `_lower` for anything that arrived through `to_radians`, and that is the
        point: this also covers a caller that builds radians some other way, or
        a subclass constructed from an unfloored model. The guard that only
        exists on the normalised path stops guarding the moment someone adds a
        path that is not normalised.
        """
        r = np.asarray(radians, dtype=np.float64).reshape(5, 4)
        held = np.maximum(r, self._lower)
        if not np.allclose(held, r):
            self.floored_frames += 1
            if self.floored_frames == 1 or self.floored_frames % 250 == 0:
                worst = float(np.min(r - self._lower))
                _log.warning(
                    "hyperextension blocked: a command reached %.3f rad past a "
                    "joint's straight limit and was held there (%d frame(s) so far)",
                    -worst, self.floored_frames)
        return held

    @property
    def open_pose(self) -> np.ndarray:
        """Normalised command for a flat open hand -- the safe fallback pose.

        Zero radians. With the hyperextension floor applied this is also
        normalised zero on every floored joint, because straight IS the bottom
        of their range -- but it is written in radians deliberately, so the
        abduction axes (which keep their symmetric model range, and whose
        normalised zero is fully splayed to one side) still land at 0 rad.
        """
        return self.to_normalised(np.zeros((5, 4)))

    @property
    def neutral_pose(self) -> np.ndarray:
        """Normalised command for a relaxed hand -- parking, not grasping.

        `open_pose` is flat: every joint at zero, fingers straight and splayed
        wherever the abduction joints happen to sit. That maximises the hand's
        silhouette, which is the wrong shape to leave it in near a table edge or
        an object.

        This curls each finger slightly and squares the abduction:

        * **Abduction (joint 1) to 0 rad**, so the fingers sit parallel rather
          than fanned. Note 0 *radians*: for the four fingers that is normalised
          **0.5**, because the range is symmetric (-0.698, +0.698). Normalised
          zero would be fully splayed to one side, the opposite of what is
          wanted. On the thumb the same index is the opposition axis, so 0 rad
          lands at normalised 0.68 and lays the thumb alongside the fingers.
        * **Flexion (joints 0, 2, 3) to +0.20 rad** each, about 11 degrees, so
          roughly 34 degrees of curl along a finger. Fingertips tucked in, and
          nowhere near a fist -- MCP flexion runs to +1.57 and PIP to +2.094.
          Curled, so it is also safely clear of the straight floor rather than
          sitting on it.

        Uniform across all five fingers on purpose. The pose recorded at the
        start of a demonstration is *not* a good template for this: it is the
        operator's own relaxed hand, so it cascades from index to pinky
        (measured 0.18 -> 0.71 rad) and carries whatever splay they held. Fine
        as a policy seed, wrong as a parking pose.
        """
        pose = np.full((5, 4), NEUTRAL_FLEXION_RAD)
        pose[:, ABDUCTION_JOINT] = 0.0
        return self.to_normalised(pose)

    def ramp_to(self, target: np.ndarray, seconds: float = 1.0,
                hz: float = 50.0) -> None:
        """Travel to `target` over `seconds`, issuing intermediate commands.

        The hand resolves a single step faster than it looks like a motion, so
        from a clenched hand a one-shot command reads as a snap. Stepping the
        command is the only way to choose the travel time.

        BLOCKING, and it must not be called while the arm is servoing: the RTDE
        watchdog times out after 0.1 s, so holding the control loop for longer
        than that turns a smooth hand motion into a protective stop. Call it
        with servoing stopped -- `UR5eArm.approach(before_move=...)` provides
        exactly that window.
        """
        start = self.state()
        goal = np.asarray(target, dtype=np.float64).reshape(self.command_dim)
        steps = max(1, int(seconds * hz))
        for i in range(1, steps + 1):
            self.command(start + (goal - start) * (i / steps))
            self.state()  # drains the ~1 kHz state stream, whose buffer holds ~1 s
            time.sleep(1.0 / hz)

    def hold(self, seconds: float, hz: float = 50.0) -> None:
        """Keep the last command for `seconds`, still reading the hand. BLOCKING, as `ramp_to`.

        Unlike a ramp to the same pose, which restarts from the measured fingers, this keeps
        the full command against whatever is stopping them.
        """
        for _ in range(max(1, int(round(seconds * hz)))):
            self.command(self._value)
            self.state()
            time.sleep(1.0 / hz)


class WujiHand2Gripper(NormalisedHand):
    """The Wuji Hand 2, over Ethernet via `wuji_sdk`.

    Same 0..1 contract as the first-generation class, but almost nothing else
    carries over:

    * **Transport.** UDP on the local subnet, discovered by `SdkManager.scan()`,
      not USB enumeration. Serial numbers start "WH"; gloves are "WG".
    * **Control.** MIT impedance (position + kp/kd), which is the firmware
      default -- the older SDK's explicit `control_mode().set("mit")` no longer
      exists. Commands are a 20-element `JointCommand` frame per publish.
    * **No built-in smoothing.** This is the big one. The first-generation
      `realtime_controller` did 100 Hz -> 1 kHz interpolation AND a low-pass
      inside the SDK; `joint_command().publish()` is a raw passthrough at
      whatever rate the caller manages. The low-pass is therefore reimplemented
      here, matching the vendor's 5 Hz default, so the hand tracks a smoothed
      target rather than a stepped one.
    * **Limits.** `effort_limit` IS the per-joint current limit in amps in SDK
      2026.9.22 ("Kt=1.0 placeholder so effort(N.m)==current(A)"); the separate
      `sys_current_limit` resource the older SDK had is gone.

    Defaults for kp / kd / current limit are the vendor's own, from
    `wuji-retargeting/example/teleop_real.py`'s argparse.
    """

    #: Impedance gains from teleop_real.py, the vendor's own teleoperation
    #: path. kd is 0.1 there and 0.05 in the wuji-sdk example; the firmware
    #: itself boots at kp 5.0 / kd 0.01, which is stiffer with almost no
    #: damping. Taking the teleoperation values deliberately.
    DEFAULT_KP = 3.0
    DEFAULT_KD = 0.1

    #: Per-joint current ceiling, amps; `None` keeps the firmware's own value.
    #: Firmware boots at 1.0 A and both vendor samples raise it to 1.5. Two
    #: first-generation hands failed at the thumb drive train, so raising it
    #: is an explicit choice rather than a default.
    DEFAULT_EFFORT_LIMIT_A = 0.5
    #: Matches the first generation's LowPass, so the feel carries over.
    DEFAULT_CUTOFF_HZ = 5.0

    #: Assumed interval for the first filtered command, when there is no
    #: previous call to measure against. The control loop runs at 125 Hz.
    _NOMINAL_DT_S = 1.0 / 125.0

    #: `status_word.ext_state`: 0=Init 1=Ready 2=Enabled 3=Stopped.
    _ENABLED_STATE = 2
    _ENABLE_TIMEOUT_S = 5.0

    #: How long to wait at connect for the first measured joint frame. The
    #: hand streams at 125 Hz and the first frame was measured at 5.1 ms, so
    #: this is three orders of magnitude of slack, not a tuning parameter.
    _STATE_TIMEOUT_S = 2.0

    def __init__(
        self,
        limits: tuple[np.ndarray, np.ndarray],
        serial_number: str = "",
        handedness: str = "",
        cutoff_hz: float = DEFAULT_CUTOFF_HZ,
        kp: float = DEFAULT_KP,
        kd: float = DEFAULT_KD,
        effort_limit_a: float | None = DEFAULT_EFFORT_LIMIT_A,
        dry_run: bool = False,
    ) -> None:
        super().__init__(limits)
        self._serial = serial_number
        self._handedness = (handedness or "").lower()
        self._cutoff = cutoff_hz
        self._kp, self._kd = kp, kd
        self._effort_limit_a = effort_limit_a
        #: Compute every command, publish none, leave the motors cold.
        self._dry_run = bool(dry_run)
        #: Frames `command()` produced, radians. Dry run only.
        self.commanded: list[np.ndarray] = []
        self._manager = None
        self._hand = None
        self._publisher = None
        self._states = None
        self._sdk = None
        self._filtered = None       # low-pass state, normalised units
        self._last_cmd_t = None
        self._effort = None         # (5, 4) amps, from the last complete state frame
        self.product_sn = ""

    # -- discovery -------------------------------------------------------- #

    def _discover(self, manager):
        """Return the connected hand, selecting by serial or handedness.

        Mirrors `teleop_real.py:WujiHand2Backend`: filter scan results to hands
        ("WH" prefix -- gloves are "WG"), then disambiguate. `connect(
        handedness=...)` is not used because it also matches the same-side
        glove, and DiscoveredDevice carries no handedness, so several hands
        have to be probed by opening a session and reading it back.
        """
        hands = [d for d in manager.scan() if str(d.sn).upper().startswith("WH")]
        if not hands:
            raise RuntimeError(
                "no Wuji Hand 2 found on the network. Check it is powered and "
                "that its switch port has link -- `ping 192.168.1.111`."
            )
        if self._serial:
            match = [d for d in hands if str(d.sn) == self._serial]
            if not match:
                found = ", ".join(str(d.sn) for d in hands)
                raise RuntimeError(f"serial {self._serial!r} not among [{found}]")
            return manager.connect(address=match[0].address, device_name="wuji_hand_2")
        if len(hands) == 1:
            return manager.connect(address=hands[0].address, device_name="wuji_hand_2")
        if self._handedness not in ("left", "right"):
            found = ", ".join(f"{d.sn}@{d.address}" for d in hands)
            raise RuntimeError(
                f"{len(hands)} hands online ({found}); pass a serial or a side."
            )
        for d in hands:
            hand = manager.connect(address=d.address, device_name="wuji_hand_2")
            try:
                side = str(hand.handedness().get()).lower()
            except Exception:  # noqa: BLE001 -- a probe failure is not fatal
                side = "?"
            if side == self._handedness:
                return hand
            hand.disconnect()
            time.sleep(0.2)  # let the device_name bridge release before reusing it
        found = ", ".join(str(d.sn) for d in hands)
        raise RuntimeError(f"no {self._handedness} hand among [{found}]")

    # -- Gripper interface ------------------------------------------------ #

    def connect(self) -> None:
        import wuji_sdk  # noqa: PLC0415 -- deferred so --help works without it

        self._sdk = wuji_sdk
        self._manager = wuji_sdk.SdkManager.instance()
        self._hand = self._discover(self._manager)
        # From here the session is open on the device. Any failure must tear it
        # down, or the next run fails with "Session already exists (0x0013)".
        try:
            n_online = int(self._hand.online_joints_count().get())
            if n_online != self.command_dim:
                raise RuntimeError(
                    f"hand reports {n_online}/{self.command_dim} joints online. The "
                    "dataset action width and the retargeting config both assume 20 DoF."
                )
            self.product_sn = str(self._hand.serial_number)
            try:
                self.handedness = str(self._hand.handedness().get()).lower()
            except Exception:  # noqa: BLE001
                self.handedness = ""

            # Read first so the log records what the hand actually shipped
            # with, whether or not we then change it.
            try:
                firmware_limit = float(min(self._hand.effort_limit().get()))
            except Exception:  # noqa: BLE001 -- informational only
                firmware_limit = float("nan")
            if self._effort_limit_a is None:
                self._effort_limit_a = firmware_limit
            elif self._dry_run:
                pass        # a dry run writes nothing to the hand, not even a limit
            elif self._effort_limit_a > firmware_limit:
                _log.warning(
                    "raising the hand's current ceiling from its firmware "
                    "default %.2f A to %.2f A", firmware_limit, self._effort_limit_a)
                self._hand.effort_limit().set(self._effort_limit_a)
            else:
                self._hand.effort_limit().set(self._effort_limit_a)
            # Measure before energising. `_read_positions` is non-blocking and
            # a fresh subscription returns None for ~5 ms; treating that as "no
            # measurement" once seeded the filter at normalised zero, which on
            # this model is every joint 60 deg back-bent.
            self._states = self._hand.joint_states().subscribe()
            measured = self._await_positions()
            self._value = self.to_normalised(measured)
            self._filtered = self._value.copy()

            # Enabling is what energises the motors, so a dry run stops short
            # of it. Everything above this line is a read.
            if not self._dry_run:
                self._hand.mit_params().set((self._kp, self._kd))
                # Publisher first, so the hold frame goes out the instant the
                # motors are live: until it does, the hand is energised against
                # the firmware's own undocumented target.
                self._publisher = self._hand.joint_command().publish()
                self._hand.enable()
                self._await_enabled()
                self._hold(measured)
            _log.info(
                "WujiHand2Gripper connected: SN %s%s, %d joints, %.2f A limit, "
                "kp %.1f kd %.2f, LowPass %.1f Hz",
                self.product_sn, f" ({self.handedness})" if self.handedness else "",
                n_online, self._effort_limit_a, self._kp, self._kd, self._cutoff,
            )
        except BaseException:
            self._teardown()
            raise

    def _hold(self, measured: np.ndarray) -> None:
        """Publish "stay exactly where you are" as the first frame after enable.

        The firmware holds some internal position target from the moment the
        motors energise, and it is not ours. Commanding the measured pose
        straight away replaces it with one that asks for no motion at all, so
        whatever the hand was about to do, it stops doing it within one frame
        instead of within however long the caller takes to send its first
        command.
        """
        held = self.enforce_floor(measured).reshape(-1)
        self._publisher.send([self._sdk.JointCommand(float(q), 0.0, 0.0)
                              for q in held])

    def _await_enabled(self) -> None:
        """Block until every live joint reports Enabled, or raise.

        `enable()` is an action, not a request -- publishing before the motors
        have actually enabled races the command stream ahead of the hand.

        Polled every 5 ms rather than every 200 ms. The motors are already
        live throughout this wait, holding the firmware's own target, so the
        poll interval is the floor on how long that goes unopposed: it is
        dead time on an energised hand, not a politeness interval.
        """
        deadline = time.monotonic() + self._ENABLE_TIMEOUT_S
        diag = self._hand.joint_diagnostics().subscribe()
        last = None
        try:
            while time.monotonic() < deadline:
                time.sleep(0.005)
                frame = diag.recv()
                if frame is None or not frame.joints:
                    continue
                last = frame
                if all(e.status_word.ext_state == self._ENABLED_STATE
                       for e in frame.joints):
                    return
        finally:
            diag.close()
        states = ([e.status_word.ext_state for e in last.joints] if last else [])
        raise RuntimeError(
            f"hand did not enable within {self._ENABLE_TIMEOUT_S:.0f}s; "
            f"per-joint ext_state {states} (want {self._ENABLED_STATE})"
        )

    def _await_positions(self) -> np.ndarray:
        """Block until the hand reports where it actually is, or raise.

        Only used at connect, where it is worth waiting: every later caller
        gets the non-blocking `_read_positions` and tolerates a dropped frame,
        because by then `_value` holds a real measurement to fall back on.
        """
        deadline = time.monotonic() + self._STATE_TIMEOUT_S
        while time.monotonic() < deadline:
            measured = self._read_positions()
            if measured is not None:
                return measured
            time.sleep(0.005)
        raise RuntimeError(
            f"hand reported no joint state within {self._STATE_TIMEOUT_S:.1f}s. "
            "Refusing to command it: without a measured starting pose the "
            "first command would step from an assumed one, and no assumed pose "
            "is safe on a hand with no emergency stop."
        )

    def _read_positions(self) -> np.ndarray | None:
        """Newest (5, 4) measured position in radians, or None if no frame yet.

        Drains the subscription to the most recent frame rather than taking the
        oldest queued one: this runs once per control tick and a backlog would
        otherwise feed the loop steadily staler positions.
        """
        if self._states is None:
            return None
        frame = None
        while True:
            newer = self._states.recv()
            if newer is None:
                break
            frame = newer
        if frame is None or len(frame.joints) != self.command_dim:
            return None
        out = np.zeros(self.command_dim, dtype=np.float64)
        effort = np.full(self.command_dim, np.nan, dtype=np.float64)
        seen = 0
        for entry in frame.joints:
            try:
                idx = self._hand.nid_to_joint_index(entry.nid)
            except Exception:  # noqa: BLE001 -- a tactile nid is not a joint
                return None
            if entry.position is None:
                return None
            out[idx] = float(entry.position)
            if getattr(entry, "effort", None) is not None:
                effort[idx] = float(entry.effort)
            seen += 1
        if seen != self.command_dim:
            return None
        self._effort = effort.reshape(5, 4)
        return out.reshape(5, 4)

    def currents(self) -> np.ndarray | None:
        """Measured current per joint, amps (5, 4), as of the last `state()`; None before one.

        The SDK reports effort with a torque constant of 1, so its unit is the amp.
        """
        return None if self._effort is None else self._effort.copy()

    @property
    def current_limit(self) -> float | None:
        return self._effort_limit_a

    def command(self, value: np.ndarray) -> None:
        """Low-pass the target, then publish one 20-joint MIT frame.

        The filter replaces the one the first-generation SDK applied internally.
        Its coefficient uses the measured interval between calls, so the cutoff
        stays at `cutoff_hz` whether the control loop runs at 125 Hz or stalls.
        """
        v = np.clip(np.asarray(value, dtype=np.float64).reshape(self.command_dim),
                    0.0, 1.0)
        self._value = v
        if self._publisher is None and not self._dry_run:
            return
        now = time.monotonic()
        if self._filtered is None:
            # Nothing measured to ease away from, so start where we are told.
            self._filtered = v.copy()
        elif self._cutoff > 0.0:
            # The first command must be filtered too: with no previous call to
            # time against, assume the nominal control period.
            dt = (self._NOMINAL_DT_S if self._last_cmd_t is None
                  else min(max(now - self._last_cmd_t, 1e-4), 0.5))
            rc = 1.0 / (2.0 * np.pi * self._cutoff)
            alpha = dt / (dt + rc)
            self._filtered += alpha * (v - self._filtered)
        else:
            self._filtered = v.copy()
        self._last_cmd_t = now
        radians = self.enforce_floor(self.to_radians(self._filtered)).reshape(-1)
        if self._dry_run:
            self.commanded.append(radians.copy())
            return
        # MIT: position target, zero velocity and zero effort feedforward.
        self._publisher.send([self._sdk.JointCommand(float(q), 0.0, 0.0)
                              for q in radians])

    def state(self) -> np.ndarray:
        """Measured joint positions, normalised. Must stay cheap."""
        measured = None
        try:
            measured = self._read_positions()
        except Exception as exc:  # must never kill the control loop
            _log.warning("hand state read failed, echoing last command: %s", exc)
        if measured is None:
            return self._value.copy()
        return self.to_normalised(measured)

    def _teardown(self) -> None:
        """Release publisher, motors and session. Each step independent."""
        for close, what in ((lambda: self._publisher.close(), "publisher"),
                            (lambda: self._states.close(), "state stream"),
                            (lambda: self._hand.disable(), "motors"),
                            (lambda: self._manager.disconnect_all(), "session")):
            try:
                close()
            except Exception as exc:  # noqa: BLE001
                _log.debug("hand %s teardown: %s", what, exc)
        self._publisher = self._states = self._hand = self._manager = None

    def disconnect(self) -> None:
        # The hand has no emergency stop, so de-energising must happen even if
        # the publisher or the session is already wedged.
        self._teardown()
        _log.info("WujiHand2Gripper disabled")

    def __str__(self) -> str:
        amps = ("firmware default" if self._effort_limit_a is None
                else f"{self._effort_limit_a:.2f} A")
        dry = " \033[33mDRY RUN -- motors never energised\033[0m" if self._dry_run else ""
        return (f"WujiHand2Gripper (20 DoF, {amps}, "
                f"kp {self._kp:.1f}, LowPass {self._cutoff:.1f} Hz){dry}")


def make_gripper(kind: str = "none", **kwargs: object) -> Gripper:
    """Factory, so scripts can take `--gripper` without importing implementations."""
    if kind in ("none", "null"):
        return NullGripper()
    if kind == "wuji":
        limits = kwargs.get("limits")
        if limits is None:
            raise ValueError(f"the {kind!r} gripper needs limits=(lower, upper) from the URDF")
        return WujiHand2Gripper(limits=limits,
                                **{k: v for k, v in kwargs.items()
                                   if k != "limits"})  # type: ignore[arg-type]
    raise ValueError(
        f"Unknown gripper {kind!r}. Choose 'none' or 'wuji' (the Wuji Hand 2, "
        "over Ethernet); add a Gripper subclass here when other hardware is "
        "mounted."
    )

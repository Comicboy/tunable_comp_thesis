"""RTDE state types, the arm command vocabulary, and a read-only state reader.

State comes from `ur_rtde` (SDU Robotics), which replaced ~450 lines of
hand-written URScript and register plumbing. That plumbing existed for one
reason -- `ur_rtde` shipped no macOS wheels -- and this project is Linux-only
now.

Two things live here rather than in ur5e_arm.py:

**`RobotState`**, because `safety.py`, `session.py` and the tests all consume it
without ever touching the transport.

**The ENABLE_* vocabulary**, because it is not a URScript detail. It is how
`SafetyLimiter` says "move to this pose", "move to these joints" or "hold", and
`safety.py`, `tests/fakes.py` and `test_pipeline.py` are all written against it.
The numbering is inherited from the register protocol it used to travel over;
the values no longer go on the wire, but keeping them stable kept every caller
and every test unchanged through the migration.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from ..config import RobotCfg

_log = logging.getLogger(__name__)

# --- command vocabulary ---------------------------------------------------- #
#
# What a SafetyVerdict is asking the arm to do. Formerly the value written to
# input_int_register_24; now interpreted host-side by UR5eArm.command().
ENABLE_HOLD = 0
ENABLE_SERVO = 1  # command is a TCP pose  [x,y,z,rx,ry,rz]
ENABLE_EXIT = 2  # kept for API compatibility; stop_servo() handles this now
ENABLE_SERVO_JOINT = 3  # command is joint targets [q0..q5]

#: Largest jump between consecutive IK solutions, radians, L2 over six joints.
#: An elbow or wrist branch flip near a singularity shows up as a step far
#: larger than this. Enforced host-side in UR5eArm.command() -- it used to be a
#: check inside the URScript.
#:
#: NOTE the norm: L2 here, matching the original URScript. `JointLimiter` in
#: safety.py uses L-infinity for a different job (per-joint rate limiting).
MAX_JOINT_STEP = 0.05

#: The host owes the controller a command roughly this often while servoing.
#: RTDE's port on the controller. Probed before handing off to ur_rtde, whose
#: constructor retries forever against an unreachable host -- an indefinite
#: hang with nothing printed, which reads as a broken script rather than a
#: network that is not up.
RTDE_PORT = 30004

#: Seconds to wait for that probe. Long enough for a switch, short enough that
#: a wrong network profile is obvious rather than something you wait out.
REACHABLE_TIMEOUT_S = 3.0


def _require_reachable(host: str, port: int = RTDE_PORT,
                       timeout: float = REACHABLE_TIMEOUT_S) -> None:
    """Fail fast, and name the usual cause.

    On this cell an unreachable controller is almost always the ur5e network
    profile being down rather than a dead robot: the Franka profile
    autoconnects and this one does not, so a reboot leaves the wrong subnet up.
    """
    import socket  # noqa: PLC0415

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return
    except OSError as exc:
        raise ConnectionError(
            f"No RTDE at {host}:{port} after {timeout:.0f} s ({exc}).\n"
            "  Usually the network profile, which does not autoconnect:\n"
            "      nmcli con up ur5e\n"
            f"  Then check with:  ping -c3 {host}\n"
            "  If that works, the controller may be off or still booting."
        ) from exc


#: ur_rtde's control script keeps its own watchdog, exactly as the old
#: hand-written one did; this is the budget a stall has to stay inside, and
#: run_policy.py warns when the gap before the first tick eats into it.
WATCHDOG_HZ = 10
WATCHDOG_TIMEOUT_S = 1.0 / WATCHDOG_HZ

# --- controller enums ------------------------------------------------------ #

ROBOT_MODE = {
    -1: "NO_CONTROLLER",
    0: "DISCONNECTED",
    1: "CONFIRM_SAFETY",
    2: "BOOTING",
    3: "POWER_OFF",
    4: "POWER_ON",
    5: "IDLE",
    6: "BACKDRIVE",
    7: "RUNNING",
    8: "UPDATING_FIRMWARE",
}

RUNTIME_STATE = {
    0: "STOPPING",
    1: "STOPPED",
    2: "PLAYING",
    3: "PAUSED",
    4: "RESUMING",
    5: "RETRACTING",
}

# safety_status_bits, bit index -> meaning.
_SAFETY_BITS = {
    0: "NORMAL",
    1: "REDUCED",
    2: "PROTECTIVE_STOP",
    3: "RECOVERY",
    4: "SAFEGUARD_STOP",
    5: "SYSTEM_EMERGENCY_STOP",
    6: "ROBOT_EMERGENCY_STOP",
    7: "EMERGENCY_STOP",
    8: "VIOLATION",
    9: "FAULT",
    10: "STOPPED_DUE_TO_SAFETY",
}

# Any of these means the arm is not going to move and we should stop pretending
# it will. NORMAL and REDUCED are the two healthy states.
_FAULT_BITS = {2, 4, 5, 6, 7, 8, 9, 10}


@dataclass
class RobotState:
    """One RTDE sample."""

    timestamp: float
    actual_q: np.ndarray  # (6,) rad
    actual_qd: np.ndarray  # (6,) rad/s
    actual_tcp_pose: np.ndarray  # (6,) [x,y,z,rx,ry,rz], m + axis-angle
    actual_tcp_speed: np.ndarray  # (6,)
    target_q: np.ndarray  # (6,) commanded joint positions
    robot_mode: int
    safety_status_bits: int
    runtime_state: int

    @property
    def mode_name(self) -> str:
        return ROBOT_MODE.get(self.robot_mode, f"UNKNOWN({self.robot_mode})")

    @property
    def runtime_name(self) -> str:
        return RUNTIME_STATE.get(self.runtime_state, f"UNKNOWN({self.runtime_state})")

    @property
    def safety_flags(self) -> list[str]:
        return [
            name
            for bit, name in _SAFETY_BITS.items()
            if self.safety_status_bits & (1 << bit)
        ]

    @property
    def is_faulted(self) -> bool:
        """True on protective stop, safeguard stop, e-stop, violation or fault."""
        return any(self.safety_status_bits & (1 << b) for b in _FAULT_BITS)

    @property
    def is_program_running(self) -> bool:
        return self.runtime_state == 2  # PLAYING

    def describe(self) -> str:
        flags = ", ".join(self.safety_flags) or "none"
        return (
            f"mode={self.mode_name} runtime={self.runtime_name} safety=[{flags}]"
        )


def build_state(rtde_r) -> RobotState:
    """Assemble a RobotState from an `ur_rtde` RTDEReceiveInterface.

    Shared by UR5eArm and by RTDEInterface below so there is exactly one place
    that decides what a sample contains.

    `target_q` is deliberate and load-bearing: it is what the controller was
    COMMANDED to, not what it achieved, and it is what gets recorded as the
    action. Recording the achieved state instead makes a replay or a policy see
    "already there" and under-actuate.
    """
    return RobotState(
        timestamp=rtde_r.getTimestamp(),
        actual_q=np.asarray(rtde_r.getActualQ(), dtype=np.float64),
        actual_qd=np.asarray(rtde_r.getActualQd(), dtype=np.float64),
        actual_tcp_pose=np.asarray(rtde_r.getActualTCPPose(), dtype=np.float64),
        actual_tcp_speed=np.asarray(rtde_r.getActualTCPSpeed(), dtype=np.float64),
        target_q=np.asarray(rtde_r.getTargetQ(), dtype=np.float64),
        robot_mode=int(rtde_r.getRobotMode()),
        safety_status_bits=int(rtde_r.getSafetyStatusBits()),
        runtime_state=int(rtde_r.getRuntimeState()),
    )


class RTDEInterface:
    """Read-only state stream. Cannot move the robot.

    For monitors and diagnostics -- `stream_state.py`, `set_home.py`,
    `setup/preflight.py`. Deliberately receive-only: it opens no control
    interface, so it needs neither Remote Control on the pendant nor a script
    upload, and it cannot command motion even by mistake. Control lives in
    `UR5eArm`.
    """

    def __init__(self, cfg: RobotCfg) -> None:
        self.cfg = cfg
        self._rtde_r = None
        self._period = 1.0 / cfg.control.frequency_hz
        self._next: float | None = None

    def connect(self) -> None:
        import rtde_receive  # noqa: PLC0415 -- keep import cost off module load

        _require_reachable(self.cfg.connection.host)
        self._rtde_r = rtde_receive.RTDEReceiveInterface(
            self.cfg.connection.host, self.cfg.control.frequency_hz
        )
        self._next = None
        _log.info("RTDE receive connected to %s", self.cfg.connection.host)

    def disconnect(self) -> None:
        if self._rtde_r is not None:
            self._rtde_r.disconnect()
            self._rtde_r = None

    def is_connected(self) -> bool:
        return self._rtde_r is not None and self._rtde_r.isConnected()

    def __enter__(self) -> RTDEInterface:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.disconnect()

    def receive(self) -> RobotState | None:
        """Next sample, paced to `frequency_hz`. None once the stream ends.

        `ur_rtde`'s getters do NOT block -- they return the latest cached value
        -- so the pacing that the old blocking RTDE read gave us for free has to
        be done explicitly. A deadline clock is used rather than a flat sleep so
        the time spent building the sample is subtracted rather than added.

        The control loop does NOT come through here: `UR5eArm.receive()` paces
        with the control interface's own initPeriod/waitPeriod, which is the
        mechanism ur_rtde intends for that job.
        """
        if self._rtde_r is None or not self._rtde_r.isConnected():
            return None
        now = time.monotonic()
        if self._next is not None and now < self._next:
            time.sleep(self._next - now)
        self._next = max(now, self._next or now) + self._period
        return build_state(self._rtde_r)

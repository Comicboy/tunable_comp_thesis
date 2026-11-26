"""High-level UR5e interface, built on `ur_rtde` (SDU Robotics).

Replaces a hand-written URScript servo program and RTDE register protocol.
That code existed because `ur_rtde` shipped no macOS wheels; the project is
Linux-only now, so it is gone.

**Pacing.** The old `receive()` blocked on the RTDE socket, so a loop of
receive -> compute -> command ran at exactly `control.frequency_hz` with no
sleep. `ur_rtde`'s getters do NOT block -- they return the latest cached sample
-- so the clock is now the control interface's own `initPeriod()`/`waitPeriod()`
pair, absorbed inside `receive()`. Callers are unchanged: call `receive()` at
the top of the tick and you are paced. This is a sleep-based clock on a
non-realtime kernel, so tick-period jitter is worth watching; `run_policy.py`
already reports it.

**The latched hold moved here.** The old URScript held a target on the
controller; `hold()` reproduces it host-side.

**The IK branch-flip guard did NOT move here, and could not.** Solving IK
host-side costs 16 ms -- two full control cycles of request/reply latency, not
computation -- against an 8 ms tick. The pose path therefore uses `servoL` and
lets the controller solve IK inside the move, exactly as the old URScript did.
The guard degrades from prevention to after-the-fact reporting; see
`_servo_pose` for the measurement and what still protects that path.

**The workspace box is enforced here on every command and move**, so no script can skip it:
pose targets are clamped, joint targets kept inside, joint moves checked along their path,
and the measured TCP watched, during blocking moves too.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from ..config import RobotCfg
from ..transforms import angle_between, make_pose, pose_to_rotation
from .dashboard import DashboardClient
from .kinematics import box_excess, pose_matrix, tcp_position
from .rtde_io import (
    ENABLE_EXIT,
    ENABLE_HOLD,
    ENABLE_SERVO,
    ENABLE_SERVO_JOINT,
    MAX_JOINT_STEP,
    RobotState,
    build_state,
)
from .safety import ForceLimitExceeded, SafetyAbort

_log = logging.getLogger(__name__)

#: Joint acceleration for blocking moves, rad/s^2. This is the value the
#: old hand-written movej script used. ur_rtde's moveJ defaults to 1.4,
#: nearly 3x faster -- passing this explicitly keeps homing as gentle as it
#: has always been.
HOME_ACCEL = 0.5

#: TOOL acceleration for blocking Cartesian moves, m/s^2. Note the units: moveL
#: is metres, moveJ is radians, and HOME_ACCEL does not apply here. ur_rtde
#: defaults to 1.2; this is gentler, for the same reason homing is.
MOVE_L_ACCEL = 0.5

#: Ticks over safety.max_tcp_force before stopping (24 ms), so one noisy sample cannot.
FORCE_TRIP_TICKS = 3
#: Settling time after homing before the force sensor is zeroed, s.
FORCE_ZERO_SETTLE_S = 0.2

#: Spacing of the inverse-kinematics samples checked along a straight move, m and rad.
IK_SAMPLE_M = 0.04
IK_SAMPLE_RAD = 0.15
#: Largest joint change between neighbouring samples; a configuration flip jumps far more.
IK_STEP_MAX = 0.5
#: Elbow and wrist 2 are singular at 0 and pi: |sin| of both must stay above this.
SINGULAR_SIN_MIN = 0.1
#: How long a just-issued move may take to report itself running, s.
ASYNC_START_S = 0.5
#: A move has ended once no joint exceeds this speed (rad/s) for STEADY_TICKS ticks.
STEADY_QD = 0.01
STEADY_TICKS = 3
#: Timeout of a Cartesian move beyond three times its nominal duration, s.
MOVE_TIMEOUT_S = 10.0
#: Decelerations for stopping a move: stopL in m/s^2, stopJ in rad/s^2.
STOP_L_DECEL = 10.0
STOP_J_DECEL = 4.0
#: Largest disagreement between the host's arm model and the controller's TCP at connect, m.
KIN_TOL_M = 0.005
#: Spacing of the samples checked along a joint move, rad.
JOINT_SAMPLE_RAD = 0.02
#: Slack for a joint target sliding along the box, whose step is linearised, m.
BOX_SLACK_M = 1e-4
#: Room for tracking error on the measured TCP during a blocking move, m.
WATCH_MARGIN_M = 0.005


class UR5eArm:
    """Connection, homing, and the servo command channel."""

    def __init__(self, cfg: RobotCfg) -> None:
        self.cfg = cfg
        self._rtde_c = None  # RTDEControlInterface: commands, IK, pacing
        self._rtde_r = None  # RTDEReceiveInterface: state
        self._servo_running = False
        #: Open period handle from initPeriod(), closed by the next waitPeriod().
        self._t0 = None
        #: Latched hold target. None means "re-latch at the next hold".
        self._q_hold: np.ndarray | None = None
        #: Previous joint target, the reference for branch-flip detection.
        self._q_prev: np.ndarray | None = None
        #: Branch flips seen this session. See _detect_branch_flip.
        self.branch_flips = 0
        #: (3, 2) workspace box, enforced by this class on every motion path.
        #: Read once: a caller mutating cfg mid-session must not widen it.
        self._box = cfg.safety.workspace.as_array()
        #: Servo targets pulled back into the box. Non-zero means something
        #: upstream skipped SafetyLimiter, which is worth knowing about.
        self.workspace_clamped = 0
        self._clamp_warned = False
        #: Read once, like the box: a caller mutating cfg must not raise it.
        self._max_force = float(cfg.safety.max_tcp_force)
        self._force_over = 0
        #: Last measured TCP force magnitude, and the peak since a caller reset it, N.
        self.tcp_force = 0.0
        self.peak_tcp_force = 0.0
        self._tol = cfg.safety.workspace_tolerance
        #: Flange-to-TCP transform from the pendant, set by connect(). Joint commands and
        #: moves are refused without it, since they could not be checked against the box.
        self._tool: np.ndarray | None = None
        #: Last joint target sent; None after a pose command, which the controller solves.
        self._q_cmd: np.ndarray | None = None
        self._box_warned = False

    # -- workspace enforcement -------------------------------------------- #
    #
    # The box lives here, not only in SafetyLimiter, because the limiter is
    # opt-in at the call site: `arm.move_to_pose(...)` reaches the controller
    # without passing through it, which is how a bad detection once drove the
    # arm to a folded configuration. These three checks cannot be routed
    # around, so they bound every script whether or not it remembered to
    # filter. SafetyLimiter stays the informative layer; this is the backstop.

    def _assert_in_box(self, pose: np.ndarray, what: str) -> None:
        """Raise SafetyAbort if a COMMANDED pose is outside the workspace.

        For one-shot blocking moves, where refusing is correct: a target this
        far off is a bad input, and clamping it would drive somewhere nobody
        asked for. No tolerance -- that exists for measured state, not targets.
        """
        pos = np.asarray(pose, dtype=np.float64).reshape(6)[:3]
        if np.any(pos < self._box[:, 0]) or np.any(pos > self._box[:, 1]):
            raise SafetyAbort(
                f"{what} target {np.round(pos, 4).tolist()} is outside the "
                f"workspace box (x{self._box[0].tolist()} y{self._box[1].tolist()} "
                f"z{self._box[2].tolist()}). Refusing to move."
            )

    def _clamp_into_box(self, pose: np.ndarray) -> np.ndarray:
        """Pull a servo target inside the box, counting each time it bites.

        Clamped rather than refused because this runs at frequency_hz: raising
        would stop the command stream and the controller's watchdog would
        protective-stop the arm, which is a worse outcome than a bounded move.
        """
        pose = np.asarray(pose, dtype=np.float64).reshape(6)
        pos = np.clip(pose[:3], self._box[:, 0], self._box[:, 1])
        if not np.array_equal(pos, pose[:3]):
            self.workspace_clamped += 1
            if not self._clamp_warned:
                self._clamp_warned = True
                _log.warning(
                    "Servo target %s clamped into the workspace box by UR5eArm. "
                    "SafetyLimiter should have caught this first -- check the "
                    "caller is filtering.", np.round(pose[:3], 4).tolist(),
                )
            pose = pose.copy()
            pose[:3] = pos
        return pose

    # -- preflight -------------------------------------------------------- #

    def preflight(self, require_remote: bool = True) -> dict[str, str]:
        """Check the robot is in a state that can accept external control.

        Raises SafetyAbort with an actionable message rather than letting the
        failure surface later as an unexplained lack of motion. Still worth
        doing before connecting: RTDEControlInterface's constructor uploads a
        script, and in Local mode that upload is silently discarded.
        """
        info: dict[str, str] = {}
        with DashboardClient(self.cfg) as dash:
            info["polyscope"] = dash.polyscope_version()
            info["robot_mode"] = dash.robot_mode()
            info["safety_status"] = dash.safety_status()
            info["program_state"] = dash.program_state()
            remote = dash.is_in_remote_control()
            info["remote_control"] = str(remote)

        if require_remote and not remote:
            raise SafetyAbort(
                "Robot is in Local mode. e-Series controllers silently ignore URScript "
                "sent to ports 30001-30003 unless Remote Control is active.\n"
                "  Fix: on the teach pendant, top-right menu -> Remote Control."
            )
        if info["safety_status"] not in ("NORMAL", "REDUCED"):
            raise SafetyAbort(
                f"Safety status is {info['safety_status']}, expected NORMAL or REDUCED. "
                "Clear the stop on the teach pendant first."
            )
        if info["robot_mode"] != "RUNNING":
            raise SafetyAbort(
                f"Robot mode is {info['robot_mode']}, expected RUNNING. "
                "Power on the arm and release the brakes."
            )
        return info

    # -- lifecycle -------------------------------------------------------- #

    def connect(self, require_remote: bool = True) -> dict[str, str]:
        import rtde_control  # noqa: PLC0415 -- keep import cost off module load
        import rtde_receive  # noqa: PLC0415

        info = self.preflight(require_remote=require_remote)
        host = self.cfg.connection.host
        freq = self.cfg.control.frequency_hz
        self._rtde_r = rtde_receive.RTDEReceiveInterface(host, freq)
        # Uploads ur_rtde's own control script and waits for it to be running.
        # This is the step the old start_servo() did by hand.
        self._rtde_c = rtde_control.RTDEControlInterface(host, freq)
        _log.info("ur_rtde connected to %s at %.0f Hz", host, freq)
        self._check_force_sensor()
        self._init_kinematics()
        return info

    def _init_kinematics(self) -> None:
        """Take the pendant's TCP offset; refuse if the host's arm model then disagrees with the
        controller about where the TCP is."""
        tool = pose_matrix(self._rtde_c.getTCPOffset())
        state = build_state(self._rtde_r)
        err = float(np.linalg.norm(tcp_position(state.actual_q, tool) - state.actual_tcp_pose[:3]))
        if err > KIN_TOL_M:
            raise SafetyAbort(
                f"The host's arm model puts the TCP {err * 1000:.1f} mm from where the "
                f"controller reports it (limit {KIN_TOL_M * 1000:.0f} mm), so joint targets "
                "cannot be checked against the workspace box. Check the TCP on the pendant.")
        self._tool = tool
        _log.info("Arm model agrees with the controller to %.1f mm", err * 1000)

    def _check_force_sensor(self) -> None:
        """Refuse to run without a force reading or with the wrong payload configured."""
        if len(self._rtde_r.getActualTCPForce()) < 3:
            raise SafetyAbort("The controller gives no TCP force reading; the force guard "
                              "cannot run.")
        mass = float(self._rtde_r.getPayload())
        s = self.cfg.safety
        if abs(mass - s.payload_kg) > s.payload_tolerance_kg:
            raise SafetyAbort(
                f"The pendant's payload is {mass:.2f} kg, expected {s.payload_kg:.2f} kg "
                f"+-{s.payload_tolerance_kg:.2f}. Set it under Installation -> General -> "
                "Payload, or update safety.payload_kg in config/robot.yaml if the hand's "
                "real mass differs."
            )
        _log.info("Payload %.2f kg; force guard at %.0f N", mass, self._max_force)

    def _check_force(self, baseline: np.ndarray | None = None) -> None:
        """Stop the arm when the TCP force stays over the limit for FORCE_TRIP_TICKS.

        `baseline` is subtracted first: a monitored move judges the force it adds,
        so a sensor offset from before the first zeroing cannot trip it.
        """
        f = np.asarray(self._rtde_r.getActualTCPForce(), dtype=np.float64)[:3]
        if baseline is not None:
            f = f - baseline
        self.tcp_force = float(np.linalg.norm(f))
        self.peak_tcp_force = max(self.peak_tcp_force, self.tcp_force)
        self._force_over = self._force_over + 1 if self.tcp_force > self._max_force else 0
        if self._force_over >= FORCE_TRIP_TICKS:
            self._force_over = 0
            self.stop_servo()
            raise ForceLimitExceeded(
                f"TCP force {self.tcp_force:.0f} N stayed above safety.max_tcp_force "
                f"{self._max_force:.0f} N; the arm was stopped where it is."
            )

    def zero_force(self) -> None:
        """Zero the TCP force reading. Only with the arm still and the hand touching nothing."""
        if self._servo_running:
            raise RuntimeError("Stop the servo program before zeroing the force sensor.")
        time.sleep(FORCE_ZERO_SETTLE_S)
        if not self._rtde_c.zeroFtSensor():
            _log.warning("zeroFtSensor was refused; the force guard keeps the previous zero.")
        self._force_over = 0

    def close(self) -> None:
        try:
            if self._servo_running:
                self.stop_servo()
        finally:
            for iface in (self._rtde_c, self._rtde_r):
                if iface is None:
                    continue
                try:
                    if iface is self._rtde_c:
                        iface.stopScript()
                    iface.disconnect()
                except Exception as exc:  # never let shutdown raise past the caller
                    _log.warning("Error closing RTDE interface: %s", exc)
            self._rtde_c = None
            self._rtde_r = None

    def __enter__(self) -> UR5eArm:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- data ------------------------------------------------------------- #

    def receive(self) -> RobotState:
        """Next sample, paced to `frequency_hz`. Blocks, so it paces the loop.

        The wait closes the period opened by the PREVIOUS call, so the time the
        caller spent computing and commanding is subtracted from the sleep
        rather than added to it. That is what keeps the loop at frequency_hz
        instead of frequency_hz minus the work.
        """
        if self._rtde_r is None or not self._rtde_r.isConnected():
            raise SafetyAbort("RTDE stream ended: the controller closed the connection.")
        if self._t0 is not None:
            self._rtde_c.waitPeriod(self._t0)
        self._t0 = self._rtde_c.initPeriod()
        state = build_state(self._rtde_r)
        self._check_measured(state)
        self._check_force()
        return state

    def _check_measured(self, state: RobotState) -> None:
        """Abort if the MEASURED TCP has left the box.

        The backstop for joint-space commands. `command_joints`, `move_to_joints`
        and replay all send joint targets, and checking those against a
        Cartesian box would need forward kinematics -- measured at 16 ms, two
        control cycles, so it cannot run on this path. `actual_tcp_pose` is
        already in every RTDE sample, so watching the consequence costs nothing.
        Reactive: it bounds how far a bad command gets, it does not prevent one.

        Tolerance applies here and not to targets: the arm lags its command and
        overshoots a boundary slightly, and the demonstrations themselves reach
        z=0.0994 against a 0.10 floor.
        """
        pos = state.actual_tcp_pose[:3]
        tol = self.cfg.safety.workspace_tolerance
        if np.any(pos < self._box[:, 0] - tol) or np.any(pos > self._box[:, 1] + tol):
            raise SafetyAbort(
                f"TCP left the workspace box at {np.round(pos, 4).tolist()} "
                f"(x{self._box[0].tolist()} y{self._box[1].tolist()} "
                f"z{self._box[2].tolist()}, tolerance {tol} m); stopping."
            )

    def command(self, pose: np.ndarray, enable: int = ENABLE_SERVO) -> None:
        """Act on a SafetyVerdict: a TCP pose, joint targets, or a hold."""
        if enable == ENABLE_HOLD:
            self.hold()
        elif enable == ENABLE_SERVO_JOINT:
            self.command_joints(pose)
        elif enable == ENABLE_EXIT:
            self.stop_servo()
        elif enable == ENABLE_SERVO:
            self._servo_pose(np.asarray(pose, dtype=np.float64).reshape(6))
        else:
            raise ValueError(f"Unknown enable value {enable!r}.")

    def _servo_pose(self, pose: np.ndarray) -> None:
        """Servo to a TCP pose. The CONTROLLER solves IK, inside the move.

        Deliberately `servoL` rather than `getInverseKinematics` + `servoJ`.
        Solving IK host-side was measured at a median of **16.0 ms per call, or
        exactly 2.00 control cycles**, against an 8 ms tick -- a 2x overrun on
        every tick, which would drop teleoperation to ~60 Hz and make the arm
        visibly lag the operator's hand.

        That cost is structural, not slow code. `getForwardKinematics` -- six
        matrix multiplies -- measured 15.996 ms against inverse kinematics'
        15.995, and 100% of samples landed on exactly two control cycles. The
        time is not spent computing; it is spent waiting for a request to be
        picked up on the controller's next 125 Hz cycle and answered on the one
        after. No faster host touches it. `scripts/troubleshooting/ik_latency_test.py`
        reproduces the measurement.

        `servoL` sends the pose and lets the controller solve IK as part of
        executing the move, so there is no request/reply at all. This is what
        the old URScript did (`servoj(get_inverse_kin(pose), ...)`, on the
        controller), which is why that version was free.

        **What this gives up.** The branch-flip guard cannot run here any more:
        the joint solution is chosen inside the controller and never comes back
        before it executes. `_detect_branch_flip` reports one afterwards so the
        loss is visible rather than silent, but it cannot prevent one. Three
        things still stand between a flip and harm: the controller seeds its IK
        with the current configuration, which is what makes flips rare;
        `SafetyLimiter` clamps the workspace and Cartesian speed on this path;
        and the controller's own joint limits protective-stop a violent one.
        """
        servo = self.cfg.control.servoj
        pose = self._clamp_into_box(pose)
        self._q_hold = None  # moving; the next hold re-latches
        self._q_cmd = None
        ok = self._rtde_c.servoL(
            pose.tolist(),
            0.0,  # speed, unused by servoL
            0.0,  # acceleration, unused by servoL
            self.cfg.control.dt,
            servo.lookahead_time,
            servo.gain,
        )
        if ok is False:
            raise SafetyAbort("servoL was rejected by the controller.")
        self._detect_branch_flip()

    def _detect_branch_flip(self) -> None:
        """Report an elbow/wrist flip after the fact. Costs nothing.

        `getTargetQ` is a cached receive-side read -- measured at 0.000 ms, as
        against 16 ms for anything that must ask the controller a question --
        so this runs every tick for free.

        A flip is a valid pose reached through a completely different arm
        configuration, and as a single servo step it is a violent whole-arm
        reconfiguration. It shows as a joint-space jump far larger than any real
        motion at this rate: MAX_JOINT_STEP is 0.05 rad per tick, while
        `SafetyLimiter` already caps normal motion near 1.0 rad/s -- about
        0.008 rad per tick, six times smaller. Normal teleoperation does not
        reach this threshold.

        Reported, not aborted: this fires one tick after the fact, so stopping
        the loop would not undo the motion, and a false positive mid-session is
        worse than a logged warning. `branch_flips` accumulates the count for
        callers that want to report it.
        """
        q = np.asarray(self._rtde_r.getTargetQ(), dtype=np.float64)
        if self._q_prev is not None:
            step = float(np.linalg.norm(q - self._q_prev))  # L2, as the URScript used
            if step > MAX_JOINT_STEP:
                self.branch_flips += 1
                _log.warning(
                    "Joint target jumped %.3f rad in one tick (limit %.3f) -- most "
                    "likely an IK branch flip. servoL solves IK on the controller, "
                    "so this can be reported but not prevented.",
                    step, MAX_JOINT_STEP,
                )
        self._q_prev = q

    def command_joints(self, q: np.ndarray) -> None:
        """Command joint targets directly, bypassing IK.

        Used by replay and policy rollout: the dataset stores joint targets, so
        routing them back through inverse kinematics would not reproduce the
        recorded arm configuration exactly.
        """
        target = self._box_joints(np.asarray(q, dtype=np.float64).reshape(6))
        self._q_prev = target
        self._servoj(target)

    def _box_joints(self, desired: np.ndarray) -> np.ndarray:
        """`desired`, kept so its TCP stays in the workspace box: a target outside slides along
        the box from the last one sent, as the pose clamp does."""
        if self._tool is None:
            raise SafetyAbort("Joint commands need the arm model connect() reads; refusing.")
        if not np.all(np.isfinite(desired)):
            raise SafetyAbort("Refusing a non-finite joint target.")
        last = self._q_cmd
        if last is None:
            last = np.asarray(self._rtde_r.getActualQ(), dtype=np.float64)
        cmd = last + self._keep_in_box(last, desired - last)
        self._q_cmd = cmd.copy()
        return cmd

    def _keep_in_box(self, last: np.ndarray, step: np.ndarray) -> np.ndarray:
        """`step`, less the part that would carry the TCP out of the box, or further out than it
        is at `last`."""
        box, tool = self._box, self._tool
        p = tcp_position(last + step, tool)
        if np.all((p >= box[:, 0]) & (p <= box[:, 1])):
            return step
        self.workspace_clamped += 1
        if not self._box_warned:
            self._box_warned = True
            _log.warning("Joint target outside the workspace box; kept inside by UR5eArm.")
        h = 1e-6
        P = tcp_position(np.vstack([last, last + h * np.eye(6)]), tool)
        p_last, J = P[0], (P[1:] - P[0]).T / h
        out = step.copy()
        for k in range(3):
            g = J[k]
            if g @ g < 1e-12:
                continue
            hi, lo = box[k, 1], box[k, 0]
            if p[k] > hi:
                plane = hi if p_last[k] <= hi + BOX_SLACK_M else p_last[k]
                over = float(g @ out) - (plane - p_last[k])
                if over > 0:
                    out = out - (over / float(g @ g)) * g
            elif p[k] < lo:
                plane = lo if p_last[k] >= lo - BOX_SLACK_M else p_last[k]
                over = float(g @ out) - (plane - p_last[k])
                if over < 0:
                    out = out - (over / float(g @ g)) * g
        low = np.minimum(box[:, 0], p_last) - BOX_SLACK_M
        high = np.maximum(box[:, 1], p_last) + BOX_SLACK_M

        def inside(s: np.ndarray) -> bool:
            q = tcp_position(last + s, tool)
            return bool(np.all((q >= low) & (q <= high)))

        if inside(out):
            return out
        a, b = 0.0, 1.0
        for _ in range(12):
            if inside(((a + b) / 2) * step):
                a = (a + b) / 2
            else:
                b = (a + b) / 2
        return a * step

    def hold(self) -> None:
        """Hold position at wherever the arm was when holding began.

        Latched, not "stop commanding": a UR that stops receiving servo commands
        drifts and eventually trips its watchdog, while one servoed to a fixed
        target actively resists being pushed. The latch is taken once on
        entering a hold and cleared by the next real motion, so repeated calls
        keep targeting the same point rather than ratcheting along with any
        drift they failed to prevent.
        """
        if self._q_hold is None:
            self._q_hold = np.asarray(self._rtde_r.getActualQ(), dtype=np.float64)
        self._q_cmd = self._q_hold.copy()
        self._servoj(self._q_hold, latched=True)

    def _servoj(self, q: np.ndarray, latched: bool = False) -> None:
        if not latched:
            self._q_hold = None  # moving; the next hold re-latches here
        servo = self.cfg.control.servoj
        ok = self._rtde_c.servoJ(
            q.tolist(),
            0.0,  # speed, unused by servoJ
            0.0,  # acceleration, unused by servoJ
            self.cfg.control.dt,
            servo.lookahead_time,
            servo.gain,
        )
        if ok is False:
            raise SafetyAbort("servoJ was rejected by the controller.")

    # -- monitored moves -------------------------------------------------- #

    def _watch(self, baseline: np.ndarray, box_allow: float) -> RobotState:
        """One paced sample while the controller runs a move: faults, the force guard, and the
        measured TCP no further outside the box than `box_allow`."""
        if self._rtde_r is None or not self._rtde_r.isConnected():
            raise SafetyAbort("RTDE stream ended: the controller closed the connection.")
        if self._t0 is not None:
            self._rtde_c.waitPeriod(self._t0)
        self._t0 = self._rtde_c.initPeriod()
        state = build_state(self._rtde_r)
        if state.is_faulted:
            raise SafetyAbort(f"Robot faulted during the move: {', '.join(state.safety_flags)}")
        self._check_force(baseline)
        out = float(box_excess(state.actual_tcp_pose[:3], self._box))
        if out > box_allow:
            raise SafetyAbort(f"The TCP left the workspace box by {out * 1000:.0f} mm during the "
                              f"move, at {np.round(state.actual_tcp_pose[:3], 4).tolist()}.")
        return state

    def _run_move(self, start: Callable[[], object], stop: Callable[[], object],
                  what: str, timeout: float, box_allow: float) -> None:
        """Start a controller move and watch every tick until it ends; stop it on any error.

        The controller plans and executes the move. Watching it is what puts the
        force guard on homing and the approach: blocking until it returned left
        both with only the pendant's 100 N limit. Ctrl-C stops it too.

        The move counts as active while the controller reports it running or any
        joint is still turning, so the watch cannot end while the arm moves.
        """
        baseline = np.asarray(self._rtde_r.getActualTCPForce(), dtype=np.float64)[:3]
        self._force_over = 0
        if start() is False:
            raise SafetyAbort(f"The {what} was rejected by the controller.")
        t0, seen, still = time.monotonic(), False, 0
        try:
            while True:
                state = self._watch(baseline, box_allow)
                status = self._rtde_c.getAsyncOperationProgressEx()
                active = (bool(status.isAsyncOperationRunning())
                          or float(np.max(np.abs(state.actual_qd))) > STEADY_QD)
                seen = seen or active
                still = 0 if active else still + 1
                elapsed = time.monotonic() - t0
                if still >= STEADY_TICKS and (seen or elapsed > ASYNC_START_S):
                    return
                if elapsed > timeout:
                    raise SafetyAbort(f"The {what} did not finish within {timeout:.0f} s.")
        except BaseException:
            try:
                stop()
            except Exception as exc:  # noqa: BLE001 -- the original error matters more
                _log.warning("Stopping the %s failed: %s", what, exc)
            raise

    def _check_joint_path(self, target: np.ndarray) -> float:
        """Refuse a joint move that ends outside the box, or passes further outside it than the
        workspace tolerance or than the arm already is. Returns that allowance, m."""
        if self._tool is None:
            raise SafetyAbort("Joint moves need the arm model connect() reads; refusing.")
        q0 = build_state(self._rtde_r).actual_q
        n = max(2, int(np.ceil(float(np.max(np.abs(target - q0))) / JOINT_SAMPLE_RAD)) + 1)
        path = q0 + np.linspace(0.0, 1.0, n)[:, None] * (target - q0)
        out = box_excess(tcp_position(path, self._tool), self._box)
        where = f"the joint move to {np.round(target, 3).tolist()}"
        if out[-1] > 0.0:
            raise SafetyAbort(f"Refusing {where}: it ends {out[-1] * 1000:.0f} mm outside the "
                              "workspace box.")
        allow = max(float(out[0]), self._tol)
        worst = int(np.argmax(out))
        if out[worst] > allow:
            raise SafetyAbort(f"Refusing {where}: {worst / (n - 1):.0%} of the way the TCP "
                              f"would be {out[worst] * 1000:.0f} mm outside the workspace box.")
        return allow

    def _check_line(self, target: np.ndarray) -> None:
        """Refuse a straight move whose joint solution jumps, turns singular or leaves the
        controller's safety limits anywhere along the way.

        moveL solves IK on the controller along the line, starting from the current
        configuration. Sampling the same line here, each sample seeded with the
        previous solution, sees the configurations the move will pass through.
        """
        state = build_state(self._rtde_r)
        start, q = state.actual_tcp_pose, state.actual_q
        r0 = pose_to_rotation(start)
        turn = (r0.inv() * pose_to_rotation(target)).as_rotvec()
        n = int(np.ceil(max(np.linalg.norm(target[:3] - start[:3]) / IK_SAMPLE_M,
                            np.linalg.norm(turn) / IK_SAMPLE_RAD, 1.0)))
        for s in np.arange(1, n + 1) / n:
            pose = make_pose(start[:3] + s * (target[:3] - start[:3]),
                             r0 * Rotation.from_rotvec(s * turn)).tolist()
            where = f"{s:.0%} of the way to {np.round(target[:3], 3).tolist()}"
            if not self._rtde_c.isPoseWithinSafetyLimits(pose):
                raise SafetyAbort(f"Refusing to move: the pose {where} is outside the "
                                  "controller's safety limits.")
            if not self._rtde_c.getInverseKinematicsHasSolution(pose, q.tolist()):
                raise SafetyAbort(f"Refusing to move: no joint solution {where}.")
            nxt = np.asarray(self._rtde_c.getInverseKinematics(pose, q.tolist()),
                             dtype=np.float64)
            jump = float(np.max(np.abs(nxt - q)))
            if jump > IK_STEP_MAX:
                raise SafetyAbort(f"Refusing to move: the arm would change configuration "
                                  f"{where} (a {jump:.2f} rad joint jump).")
            if min(abs(np.sin(nxt[2])), abs(np.sin(nxt[4]))) < SINGULAR_SIN_MIN:
                raise SafetyAbort(f"Refusing to move: the elbow or wrist would straighten "
                                  f"into a singularity {where}.")
            q = nxt

    # -- programs --------------------------------------------------------- #

    def move_to_joints(self, q: list[float] | np.ndarray, speed: float | None = None,
                       timeout: float = 30.0, tolerance: float = 0.02) -> RobotState:
        """movej to an arbitrary joint configuration, watched by the force guard.

        Used for homing, and to reach the start of a recorded episode before
        replaying it: servoJ rejects large single-tick jumps, so streaming a
        trajectory that begins somewhere else would simply stall.
        """
        if self._servo_running:
            raise RuntimeError("Stop the servo program before a blocking move.")
        target = np.asarray(q, dtype=np.float64).reshape(6)
        speed = speed or self.cfg.home_speed
        allow = self._check_joint_path(target)
        self._run_move(lambda: self._rtde_c.moveJ(target.tolist(), speed, HOME_ACCEL, True),
                       lambda: self._rtde_c.stopJ(STOP_J_DECEL), "joint move", timeout,
                       allow + WATCH_MARGIN_M)

        state = build_state(self._rtde_r)
        if state.is_faulted:
            raise SafetyAbort(
                f"Robot faulted during joint move: {', '.join(state.safety_flags)}"
            )
        error = float(np.max(np.abs(state.actual_q - target)))
        if error > tolerance:
            raise SafetyAbort(
                f"Joint move did not reach the target (still {error:.3f} rad away, "
                f"tolerance {tolerance}).\n"
                "  If the arm never moved at all, check Remote Control is enabled "
                "on the pendant."
            )
        # A completed move invalidates both latches: the arm is somewhere else.
        self._q_hold = None
        self._q_prev = state.actual_q.copy()
        return state

    def move_to_pose(self, pose: list[float] | np.ndarray, speed: float | None = None,
                     accel: float = MOVE_L_ACCEL, tol_pos: float = 0.005,
                     tol_rot: float = 0.05) -> RobotState:
        """Blocking moveL to a TCP pose. Straight line in tool space.

        The counterpart to `move_to_joints`, for reaching a pose rather than a
        configuration -- used to drive the arm to a pre-grasp pose derived from
        the camera, where the joint angles are not known in advance.

        `pose` is [x, y, z, rx, ry, rz]: metres and an axis-angle rotation
        vector in the base frame, the same convention as `actual_tcp_pose`.

        Deliberately not servoL. That is the streaming path and needs a fresh
        target every tick; this hands the whole motion to the controller's own
        trajectory generator, which is what a one-shot approach move wants. The
        line is checked for configuration flips and singularities first, and the
        force guard watches the move.

        `speed` defaults to the same cap teleoperation runs under
        (`safety.max_linear_speed`), so a scripted move can never travel faster
        than a hand-driven one.
        """
        if self._servo_running:
            raise RuntimeError("Stop the servo program before a blocking move.")
        target = np.asarray(pose, dtype=np.float64).reshape(6)
        # moveL is a straight line in tool space and the box is convex, so a
        # target inside it cannot leave it on the way. Checking the endpoint is
        # therefore sufficient here, unlike moveJ.
        self._assert_in_box(target, "moveL")
        self._check_line(target)
        speed = speed if speed is not None else self.cfg.safety.max_linear_speed
        here = build_state(self._rtde_r).actual_tcp_pose[:3]
        dist = float(np.linalg.norm(target[:3] - here))
        self._run_move(lambda: self._rtde_c.moveL(target.tolist(), speed, accel, True),
                       lambda: self._rtde_c.stopL(STOP_L_DECEL), "Cartesian move",
                       MOVE_TIMEOUT_S + 3.0 * dist / speed,
                       float(box_excess(here, self._box)) + WATCH_MARGIN_M)

        state = build_state(self._rtde_r)
        if state.is_faulted:
            raise SafetyAbort(
                f"Robot faulted during Cartesian move: {', '.join(state.safety_flags)}"
            )
        # Position and orientation are checked separately: they have different
        # units, and averaging them would let a large rotation error hide behind
        # a small translation one.
        reached = state.actual_tcp_pose
        err_pos = float(np.linalg.norm(reached[:3] - target[:3]))
        err_rot = angle_between(pose_to_rotation(reached), pose_to_rotation(target))
        if err_pos > tol_pos or err_rot > tol_rot:
            raise SafetyAbort(
                f"Cartesian move did not reach the target "
                f"({err_pos * 1000:.1f} mm and {np.degrees(err_rot):.1f} deg away; "
                f"tolerance {tol_pos * 1000:.0f} mm, {np.degrees(tol_rot):.0f} deg).\n"
                "  An unreachable pose or a singularity looks exactly like this."
            )
        # A completed move invalidates both latches: the arm is somewhere else.
        self._q_hold = None
        self._q_prev = state.actual_q.copy()
        return state

    def move_home(self, timeout: float = 30.0) -> RobotState:
        """Blocking joint move to the configured home pose, using movej.

        Deliberately not servoj: this is a large motion to a known
        configuration, so the controller's own trajectory generator with proper
        acceleration limits is the right tool.
        """
        target = np.asarray(self.cfg.home_q, dtype=np.float64)
        _log.info("Homing to %s", np.round(target, 3).tolist())
        state = self.move_to_joints(target, speed=self.cfg.home_speed, timeout=timeout)
        error = float(np.max(np.abs(state.actual_q - target)))
        _log.info("Home reached (max joint error %.4f rad)", error)
        # Home is the one pose known to have the hand in free air.
        self.zero_force()
        return state

    def rehome(self, timeout: float = 30.0) -> RobotState:
        """Return to the home configuration mid-session, then resume servoing.

        `movej` and servoing cannot both own the arm, so servoing is stopped for
        the duration and restarted afterwards. Callers must re-anchor any
        retargeting afterwards -- the TCP has moved a long way and the
        operator's hand has not.
        """
        was_running = self._servo_running
        if was_running:
            self.stop_servo()
        state = self.move_home(timeout=timeout)
        if was_running:
            self.start_servo()
        return state

    @property
    def servo_running(self) -> bool:
        """Whether the servo program is live.

        `command()` without it does not raise -- it kills the control script,
        and the failure surfaces on the NEXT blocking move as "RTDE control
        script is not running". Callers driving their own loop should assert
        this before the first command rather than discover it an episode later.
        """
        return self._servo_running

    def approach(self, pose: list[float] | np.ndarray, speed: float | None = None,
                 accel: float = MOVE_L_ACCEL, tol_pos: float = 0.005,
                 tol_rot: float = 0.05,
                 before_move: Callable[[], None] | None = None,
                 vertical: bool = False) -> RobotState:
        """Blocking moveL mid-session, then resume servoing.

        The Cartesian counterpart to `rehome`. `moveL` and servoing cannot both
        own the arm, so servoing is stopped for the duration and restarted
        afterwards. Callers must re-anchor any retargeting and reset the safety
        limiter afterwards -- the TCP has moved a long way and the operator's
        hand has not, so resuming from the old anchor would command a jump.

        `vertical` crosses at the current height and then descends straight onto
        the target, so nothing between here and there is swept at grasp height.
        """
        was_running = self._servo_running
        if was_running:
            self.stop_servo()
        # Servoing is stopped, so the RTDE watchdog is not counting: this is the
        # one window in a session where a caller may take a second or two
        # without risking a protective stop. Used to ramp the hand.
        if before_move is not None:
            before_move()
        target = np.asarray(pose, dtype=np.float64).reshape(6)
        if vertical:
            via = target.copy()
            via[2] = max(target[2], float(build_state(self._rtde_r).actual_tcp_pose[2]))
            if via[2] > target[2] + 1e-3:
                self.move_to_pose(via, speed=speed, accel=accel, tol_pos=tol_pos,
                                  tol_rot=tol_rot)
        state = self.move_to_pose(target, speed=speed, accel=accel,
                                  tol_pos=tol_pos, tol_rot=tol_rot)
        if was_running:
            self.start_servo()
        return state

    def start_servo(self, timeout: float = 10.0) -> None:
        """Enter servo mode.

        Cheap now, and deliberately so. There is no script to upload -- ur_rtde
        uploaded its control script back in connect() -- so this only latches
        the current configuration and checks the arm is healthy. `timeout` is
        kept for API compatibility and is unused.

        It still matters that callers keep slow work out of the gap between this
        and the first receive(): the controller expects a steady command stream
        once servoing starts.
        """
        if self._servo_running:
            return
        state = self.receive()
        if state.is_faulted:
            raise SafetyAbort(
                f"Robot faulted while starting: {', '.join(state.safety_flags)}"
            )
        # Latch here so a first tick that commands a hold holds where the arm
        # actually is, not wherever a previous session left the target.
        self._q_hold = state.actual_q.copy()
        self._q_prev = state.actual_q.copy()
        self._q_cmd = state.actual_q.copy()
        self._servo_running = True
        _log.info("Servo mode active.")

    def stop_servo(self, timeout: float = 5.0) -> None:
        """Leave servo mode, decelerating to a stop. `timeout` is unused."""
        if not self._servo_running:
            return
        _log.info("Stopping servo mode.")
        try:
            self._rtde_c.servoStop()
        except Exception as exc:  # never let shutdown raise past the caller
            _log.warning("Error while stopping servo mode: %s", exc)
        finally:
            self._servo_running = False
            self._q_hold = None
            self._q_prev = None
            self._q_cmd = None
            self._t0 = None

"""Host-side safety filtering.

Every commanded pose passes through `SafetyLimiter.filter` before it reaches an
RTDE register. This layer is deliberately separate from both the teleop code and
the policy rollout code, so autonomous execution gets exactly the same limits as
a human operator -- see run_policy.py, which reuses this unchanged.

This is defence in depth, not the only defence. The controller-side watchdog and
IK-jump check in ur5e_arm.py cover the case where this process dies, and the
teach pendant's own safety limits cover the case where both do.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..config import SafetyCfg
from ..transforms import angle_between, clamp_position, limit_step, pose_to_rotation
from .rtde_io import ENABLE_HOLD, ENABLE_SERVO, ENABLE_SERVO_JOINT, RobotState

_log = logging.getLogger(__name__)


@dataclass
class SafetyVerdict:
    """What the robot should actually be commanded this tick, and why.

    `command` holds whatever the six pose registers mean in the mode named by
    `enable`: a TCP pose under ENABLE_SERVO, joint targets under
    ENABLE_SERVO_JOINT. Always pass the pair together rather than assuming
    which one you have.
    """

    command: np.ndarray
    enable: int
    clamped: bool = False
    speed_limited: bool = False
    accel_limited: bool = False
    rejected_jump: bool = False
    reason: str = ""

    @property
    def moving(self) -> bool:
        return self.enable in (ENABLE_SERVO, ENABLE_SERVO_JOINT)


class SafetyAbort(RuntimeError):
    """The robot is in a state where continuing would be unsafe or pointless."""


class ForceLimitExceeded(SafetyAbort):
    """The TCP contact force went over safety.max_tcp_force; the arm was stopped."""


@dataclass
class SafetyStats:
    """Counters, reported at session end so problems are visible after the fact."""

    ticks: int = 0
    clamped: int = 0
    speed_limited: int = 0
    accel_limited: int = 0
    rejected_jumps: int = 0
    stale_holds: int = 0
    disabled_holds: int = 0

    def summary(self) -> str:
        if self.ticks == 0:
            return "no ticks"
        pct = lambda n: f"{n} ({100.0 * n / self.ticks:.1f}%)"  # noqa: E731
        return (
            f"{self.ticks} ticks | workspace-clamped {pct(self.clamped)} | "
            f"speed-limited {pct(self.speed_limited)} | "
            f"accel-limited {pct(self.accel_limited)} | "
            f"jumps rejected {pct(self.rejected_jumps)} | "
            f"stale {pct(self.stale_holds)} | clutch-out {pct(self.disabled_holds)}"
        )


class JointLimiter:
    """Rate-limits and gates commanded JOINT targets.

    Replay and policy rollout command joints directly, so they bypass the
    Cartesian limits in SafetyLimiter entirely. That matters most for a policy:
    an unconverged checkpoint will cheerfully emit a target several radians
    away, and the only thing between that and the arm is this.
    """

    def __init__(self, cfg: SafetyCfg, dt: float) -> None:
        self._cfg = cfg
        self._dt = dt
        self._max_step = cfg.max_joint_speed * dt
        #: Largest change in per-tick step, i.e. an acceleration cap. Without
        #: one, a target far from the current pose is clamped to FULL speed on
        #: the very first tick: 0 -> 1 rad/s in 8 ms is 125 rad/s^2, well past a
        #: UR5e's joint acceleration limit, and the robot replies with a
        #: PROTECTIVE_STOP before the arm has visibly moved. servoj's lookahead
        #: smooths a one-tick spike but not a sustained ramp, which is exactly
        #: what an unconverged policy produces.
        self._max_accel_step = cfg.max_joint_accel * dt * dt
        self._max_jump = cfg.max_jump_joint
        self._box = cfg.workspace.as_array()
        self._last_cmd: np.ndarray | None = None
        #: Per-joint step actually commanded last tick -- the current velocity,
        #: in rad/tick. Acceleration is measured against this.
        self._last_step = np.zeros(6, dtype=np.float64)
        #: Previous DESIRED target. Jumps are measured against this, not against
        #: _last_cmd -- see filter() for why that distinction matters.
        self._last_desired: np.ndarray | None = None
        self.stats = SafetyStats()

    def reset(self, q: np.ndarray) -> None:
        self._last_cmd = np.asarray(q, dtype=np.float64).reshape(6).copy()
        self._last_desired = None
        # Start from rest, so the first motion after a reset ramps up rather
        # than resuming whatever speed the previous rollout ended at.
        self._last_step = np.zeros(6, dtype=np.float64)

    @property
    def last_command(self) -> np.ndarray | None:
        return None if self._last_cmd is None else self._last_cmd.copy()

    def check_workspace(self, state: RobotState) -> None:
        """Abort if the measured TCP leaves the configured box.

        Reactive rather than preventive -- without forward kinematics on the
        host we cannot tell where a joint target puts the TCP before commanding
        it. It still bounds how far a bad policy can get.
        """
        pos = state.actual_tcp_pose[:3]
        # Tolerance, because this tests the MEASURED pose while the box bounds
        # COMMANDED ones. The arm lags its target and overshoots a boundary by a
        # fraction of a millimetre; the demonstrations themselves were recorded
        # at y=+0.4503 against a 0.45 limit. Aborting on that is a false alarm.
        # A policy genuinely heading out of the cell clears this easily.
        tol = self._cfg.workspace_tolerance
        if np.any(pos < self._box[:, 0] - tol) or np.any(pos > self._box[:, 1] + tol):
            raise SafetyAbort(
                f"TCP left the workspace box at {np.round(pos, 4).tolist()} "
                f"(box x{self._box[0].tolist()} y{self._box[1].tolist()} "
                f"z{self._box[2].tolist()}, tolerance {tol} m); stopping."
            )

    def filter(
        self, desired_q: np.ndarray | None, state: RobotState, motion_enabled: bool
    ) -> SafetyVerdict:
        self.stats.ticks += 1
        if self._last_cmd is None:
            self.reset(state.actual_q)

        hold = SafetyVerdict(command=self._last_cmd.copy(), enable=ENABLE_HOLD)

        if not motion_enabled:
            self.stats.disabled_holds += 1
            hold.reason = "deadman released"
            self._last_cmd = state.actual_q.copy()
            hold.command = self._last_cmd.copy()
            return hold

        if desired_q is None:
            hold.reason = "no target"
            return hold

        desired = np.asarray(desired_q, dtype=np.float64).reshape(6)
        if not np.all(np.isfinite(desired)):
            self.stats.rejected_jumps += 1
            hold.reason = "non-finite joint target"
            return hold

        # Detect glitches in the INPUT stream, not against what was last
        # commanded -- the same correction SafetyLimiter already carries.
        #
        # Comparing against _last_cmd deadlocks: a target that is merely far
        # away gets rejected, _last_cmd therefore never advances, and every
        # subsequent target is just as far away. Observed on the first ACT
        # rollout, where 1873 of 1875 ticks were rejected and the arm never
        # moved at all. A far target is not a glitch; it is approached at the
        # speed cap. A DISCONTINUITY between consecutive targets is a glitch.
        # On the first tick there is no previous target, so compare against
        # where the arm actually is: a policy's opening target should be near
        # the current pose, and a wild one must still be caught rather than
        # crept toward at the speed cap.
        if self._last_desired is None:
            self._last_desired = state.actual_q.copy()
        if self._last_desired is not None:
            jump = float(np.max(np.abs(desired - self._last_desired)))
            if jump > self._max_jump:
                self.stats.rejected_jumps += 1
                hold.rejected_jump = True
                hold.reason = f"joint jump rejected ({jump:.3f} rad)"
                self._last_desired = desired.copy()
                return hold
        self._last_desired = desired.copy()

        delta = desired - self._last_cmd
        largest = float(np.max(np.abs(delta)))
        verdict = SafetyVerdict(command=desired.copy(), enable=ENABLE_SERVO_JOINT)
        if largest > self._max_step:
            # Scale the whole vector rather than clipping per joint, so the arm
            # slows along its path instead of bending off it.
            verdict.command = self._last_cmd + delta * (self._max_step / largest)
            verdict.speed_limited = True
            self.stats.speed_limited += 1

        step = verdict.command - self._last_cmd
        accel = step - self._last_step
        worst = float(np.max(np.abs(accel)))
        if worst > self._max_accel_step:
            accel *= self._max_accel_step / worst
            step = self._last_step + accel
            verdict.command = self._last_cmd + step
            verdict.accel_limited = True
            self.stats.accel_limited += 1
            if not verdict.reason:
                verdict.reason = f"accel-limited ({worst / (self._dt ** 2):.1f} rad/s^2)"

        self._last_step = step
        self._last_cmd = verdict.command.copy()
        return verdict


class SafetyLimiter:
    """Clamps, rate-limits and gates commanded poses."""

    def __init__(self, cfg: SafetyCfg, dt: float) -> None:
        self._cfg = cfg
        self._dt = dt
        self._box = cfg.workspace.as_array()
        self._max_lin_step = cfg.max_linear_speed * dt
        self._max_ang_step = cfg.max_angular_speed * dt
        self._last_cmd: np.ndarray | None = None
        #: Per-joint step actually commanded last tick -- the current velocity,
        #: in rad/tick. Acceleration is measured against this.
        self._last_step = np.zeros(6, dtype=np.float64)
        # Tracked separately from _last_cmd: jumps are detected in the INPUT
        # stream, not against what we last commanded. A target that is simply
        # far away (hand outside the workspace box) is not a glitch, and must
        # be approached at the speed cap rather than rejected -- otherwise the
        # arm freezes for as long as the operator's hand is out of bounds.
        self._last_desired: np.ndarray | None = None
        self.stats = SafetyStats()

    # -- lifecycle -------------------------------------------------------- #

    def reset(self, pose: np.ndarray) -> None:
        """Anchor the limiter at a known pose (usually the current TCP pose)."""
        self._last_cmd = np.asarray(pose, dtype=np.float64).copy()
        self._last_desired = None

    @property
    def last_command(self) -> np.ndarray | None:
        return None if self._last_cmd is None else self._last_cmd.copy()

    # -- checks ----------------------------------------------------------- #

    def check_robot(self, state: RobotState) -> None:
        """Raise SafetyAbort if the robot itself has stopped cooperating."""
        if state.is_faulted:
            raise SafetyAbort(
                f"Robot safety fault: {', '.join(state.safety_flags)}. "
                "Clear it on the teach pendant before restarting."
            )
        if state.robot_mode != 7:  # RUNNING
            raise SafetyAbort(
                f"Robot mode is {state.mode_name}, expected RUNNING. "
                "Check that the arm is powered and brakes are released."
            )

    def filter(
        self,
        desired: np.ndarray | None,
        state: RobotState,
        motion_enabled: bool,
        data_age: float,
    ) -> SafetyVerdict:
        """Decide what to command this tick.

        `desired` may be None when no target has been produced yet (no clutch
        engagement, no tracking). `motion_enabled` is the clutch. `data_age` is
        how long ago the Quest last sent a frame.
        """
        self.stats.ticks += 1

        if self._last_cmd is None:
            self.reset(state.actual_tcp_pose)

        hold = SafetyVerdict(command=self._last_cmd.copy(), enable=ENABLE_HOLD)

        if not motion_enabled:
            self.stats.disabled_holds += 1
            hold.reason = "clutch released"
            # Re-anchor on the live pose so re-engaging does not command a jump
            # back to wherever the arm was when the clutch was last held.
            self._last_cmd = state.actual_tcp_pose.copy()
            self._last_desired = None
            hold.command = self._last_cmd.copy()
            return hold

        if data_age > self._cfg.quest_timeout_s:
            self.stats.stale_holds += 1
            hold.reason = f"quest data stale ({data_age * 1e3:.0f} ms)"
            # Whatever arrives after a dropout is discontinuous with what came
            # before it, so forget the history rather than flagging the
            # recovery frame as a glitch.
            self._last_desired = None
            return hold

        if desired is None:
            hold.reason = "no target yet"
            return hold

        desired = np.asarray(desired, dtype=np.float64).reshape(6)

        # 1. Glitch detection, on the raw input stream. Always update the
        #    history even when rejecting, so a one-frame glitch costs a couple
        #    of ticks rather than wedging the arm until the clutch is cycled.
        previous_desired = self._last_desired
        self._last_desired = desired.copy()
        if previous_desired is not None:
            jump_lin = float(np.linalg.norm(desired[:3] - previous_desired[:3]))
            jump_ang = angle_between(
                pose_to_rotation(desired), pose_to_rotation(previous_desired)
            )
            if jump_lin > self._cfg.max_jump_linear or jump_ang > self._cfg.max_jump_angular:
                self.stats.rejected_jumps += 1
                hold.rejected_jump = True
                hold.reason = (
                    f"tracking jump rejected "
                    f"({jump_lin * 1e3:.0f} mm, {np.rad2deg(jump_ang):.0f} deg in one frame)"
                )
                return hold

        verdict = SafetyVerdict(command=desired.copy(), enable=ENABLE_SERVO)

        # 2. Pull the target inside the workspace box.
        pos, clamped = clamp_position(desired[:3], self._box)
        verdict.command[:3] = pos
        verdict.clamped = clamped
        if clamped:
            self.stats.clamped += 1

        # 3. Cap velocity. This is the ceiling that makes fast hand motion
        #    produce slower robot motion rather than fast robot motion, and it
        #    is what bounds the approach to a far-away (or clamped) target.
        limited, hit = limit_step(
            verdict.command, self._last_cmd, self._max_lin_step, self._max_ang_step
        )
        verdict.command = limited
        verdict.speed_limited = hit
        if hit:
            self.stats.speed_limited += 1

        self._last_cmd = verdict.command.copy()
        return verdict

"""Deciding whether a grasp actually holds anything.

The hand has no tactile sensing wired up and the arm reports no force, but the
answer is already in the data we record: the observation carries the hand's
MEASURED joint positions while the action carries its COMMANDED ones. Fingers
closing on air arrive where they were told; fingers closing on an object stall
short. The gap between the two is the grasp signal.

Measured over 122 recorded demonstrations, mean over all 20 joints:

    commanded closed, holding the cube : +0.0323   (min +0.0149, p5 +0.0171)
    commanded open, nothing in hand    : -0.0011   (max +0.0006, p95 +0.0000)

A threshold anywhere in that gap separates 100% of holds from 100% of opens.
Every joint contributes, though the thumb's opposition axis carries the most
(+0.20) since it does the pressing.

The same number is three things at once: grasp success for the modular handoff,
the sparse reward for RL fine-tuning, and a hardware health monitor -- a
transmission slip shows up as a residual that grows over sessions, which is what
preceded the previous hand's thumb failure (2.25x over four weeks).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pathlib import Path
import numpy as np

from .rtde_io import ENABLE_SERVO

_log = logging.getLogger(__name__)

#: Residual above which the hand is judged to be holding something, in
#: normalised joint units. The measured distributions leave a gap of 0.017
#: between p5(holding) and p95(empty); this sits in it, nearer the empty side so
#: a marginal grasp reads as a failure rather than a success.
#:
#: Re-derive it if the hand or the retargeting calibration changes. The left
#: hand's thumb opposition is remapped (`hand_retarget.THUMB_J1_KNOTS`), which
#: changes the commanded value that this residual is measured against.
HOLD_THRESHOLD = 0.010

#: How far to lift when testing a grasp, metres. Enough that a cube resting on
#: the table is unambiguously off it, small enough to stay well inside the
#: workspace box.
LIFT_M = 0.05

#: Smallest growth of a joint's residual worth flagging, in joint range. Residuals of
#: 0.01-0.08 either way are normal and several baselines are negative, so a ratio alone
#: flags noise; the hand that failed had grown by 0.25.
HEALTH_FLOOR = 0.03


@dataclass
class GraspOutcome:
    """The result of a lift test."""

    success: bool
    residual: float
    #: (5, 4) per-joint residual. Kept for the health monitor: a single joint
    #: drifting while the others hold steady is the signature of a slipping
    #: transmission, not of a different object.
    per_joint: np.ndarray = field(default_factory=lambda: np.zeros((5, 4)))

    def describe(self) -> str:
        verdict = "HOLDING" if self.success else "empty"
        return f"{verdict} (residual {self.residual:+.4f}, threshold {HOLD_THRESHOLD:.3f})"


def grasp_residual(commanded: np.ndarray, measured: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean and per-joint commanded-minus-measured finger position.

    Both arguments are normalised 0..1, either (20,) or (5, 4). Positive means
    the fingers fell short of their command, i.e. something stopped them.
    """
    c = np.asarray(commanded, dtype=np.float64).reshape(5, 4)
    m = np.asarray(measured, dtype=np.float64).reshape(5, 4)
    per_joint = c - m
    return float(per_joint.mean()), per_joint


def place_report(tcp_poses, closure, dt: float) -> dict | None:
    """How a turn-and-place attempt ended: the wrist's turn, and when the hand opened.

    tcp_poses (T, 6) measured, closure (T,) the policy's own mean finger command, one row per
    policy tick. None when the attempt never grasped, lifted and turned 75 degrees.
    `release_s` is the time from the lowest point to a quarter of the way open; None, never.
    """
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    p, c = np.asarray(tcp_poses, dtype=np.float64), np.asarray(closure, dtype=np.float64)
    if len(c) < 20:
        return None
    c0, rise = c[:5].mean(), c.max() - c[:5].mean()
    if rise < 0.08:
        return None
    start = Rotation.from_rotvec(p[0, 3:])
    turn = np.degrees((Rotation.from_rotvec(p[:, 3:]) * start.inv()).magnitude())
    closed = int(np.argmax(c > c0 + 0.5 * rise))
    lifted = closed + int(np.argmax(p[closed:closed + 60, 2]))
    if turn[lifted:].max() < 75.0:
        return None
    turned = lifted + int(np.argmax(turn[lifted:] > 75.0))
    bottom = turned + int(np.argmax(p[turned:, 2] < p[turned:, 2].min() + 0.004))
    hold = float(np.median(c[lifted:turned + 1]))
    first = max(turned, bottom - 10)
    opened = np.flatnonzero(c[first:] < hold - 0.25 * (hold - c0))
    return {"turn_deg": float(turn[turned:].max()),
            "release_s": None if len(opened) == 0 else float((first + opened[0] - bottom) * dt),
            "bottom_s": float((len(c) - 1 - bottom) * dt)}


def lift_and_verify(
    arm,
    gripper,
    commanded: np.ndarray,
    dt: float,
    lift_m: float = LIFT_M,
    seconds: float = 1.0,
    settle_s: float = 0.3,
    limiter=None,
) -> GraspOutcome:
    """Raise the hand by `lift_m` and report whether it is still holding.

    Assumes servo mode is active and the hand has already been commanded shut;
    `commanded` is that closing command, which the residual is measured against.

    The lift is streamed as a servoL ramp rather than a blocking moveL, because
    a blocking move needs servo mode stopped and this runs inside a rollout. The
    loop calls `arm.receive()` every tick, which both paces it and keeps the
    controller's watchdog fed -- skipping that is what turns a pause into a
    protective stop.

    Pass `limiter` (a SafetyLimiter) to route the ramp through the workspace
    clamp and speed cap. Without it the ramp is unchecked, which is tolerable
    only because it is small, slow, vertical and starts from a known pose.
    """
    start = arm.receive().actual_tcp_pose.copy()
    steps = max(1, int(seconds / dt))
    for i in range(steps):
        state = arm.receive()
        target = start.copy()
        target[2] += lift_m * (i + 1) / steps
        if limiter is not None:
            verdict = limiter.filter(target, state, motion_enabled=True, data_age=0.0)
            arm.command(verdict.command, verdict.enable)
        else:
            arm.command(target, ENABLE_SERVO)

    # Let the fingers settle before reading them. The hand's own low-pass runs
    # at 5 Hz, so a reading taken immediately after the lift still reflects the
    # motion rather than the hold.
    settle_ticks = max(1, int(settle_s / dt))
    for _ in range(settle_ticks):
        state = arm.receive()
        target = start.copy()
        target[2] += lift_m
        arm.command(target, ENABLE_SERVO)

    residual, per_joint = grasp_residual(commanded, gripper.state())
    outcome = GraspOutcome(residual > HOLD_THRESHOLD, residual, per_joint)
    _log.info("Grasp %s", outcome.describe())
    return outcome


BASELINE_PATH = Path("config/hand_baseline.json")


def load_baseline(serial: str, path: Path | None = None) -> np.ndarray | None:
    """The first session's per-joint residual for this hand, or None.

    Keyed by serial because a replacement hand is a different mechanism: two
    units have now failed at the thumb, and comparing a new one against its
    predecessor's baseline would either hide a fault or invent one.
    """
    import json  # noqa: PLC0415

    p = path or BASELINE_PATH
    if not p.exists() or not serial:
        return None
    entry = json.loads(p.read_text()).get(serial)
    return None if entry is None else np.asarray(entry["per_joint"]).reshape(5, 4)


def save_baseline(serial: str, per_joint: np.ndarray, path: Path | None = None) -> None:
    """Record this hand's healthy residual, once, on its first good session."""
    import json  # noqa: PLC0415
    from datetime import UTC, datetime  # noqa: PLC0415

    if not serial:
        return
    p = path or BASELINE_PATH
    data = json.loads(p.read_text()) if p.exists() else {}
    if serial in data:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    data[serial] = {"per_joint": np.asarray(per_joint).reshape(5, 4).tolist(),
                    "recorded_at": datetime.now(UTC).isoformat()}
    p.write_text(json.dumps(data, indent=2))
    _log.info("Stored the wear baseline for hand %s", serial)


def health_warning(per_joint: np.ndarray, baseline: np.ndarray | None = None,
                   factor: float = 1.8) -> str | None:
    """Flag a joint whose residual has grown against a stored baseline.

    Returns a message, or None when nothing looks wrong. Intended to be called
    once per autonomous session and compared against the first session's --
    the previous hand's thumb reported a transmission slip four weeks before it
    detached, and its tracking residual had grown 2.25x over the same window.
    """
    if baseline is None:
        return None
    baseline = np.asarray(baseline, dtype=np.float64)
    growth = np.asarray(per_joint, dtype=np.float64) - baseline
    grown = growth > np.maximum((factor - 1.0) * np.abs(baseline), HEALTH_FLOOR)
    if not grown.any():
        return None
    names = ("thumb", "index", "middle", "ring", "pinky")
    worst = np.unravel_index(int(np.argmax(growth)), (5, 4))
    return (f"{int(grown.sum())} joint(s) now stall more than {factor:.1f}x their "
            f"baseline; worst is {names[worst[0]]} J{worst[1]}. A transmission slip "
            f"looks like this. Stop and inspect before continuing unattended.")

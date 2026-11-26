"""Fingertip admittance for the real Wuji Hand 2: the simulation's law, ported to numpy.

Two layers, as in simulation:

  DynamicAdmittance / StaticAdmittance
      The laws themselves, line by line the simulation's AdmittanceController and
      StaticAdmittanceController (source/tunable_comp_thesis/.../controllers/), for one hand
      (5 points) instead of N environments:

          dynamic:  M e'' + D e' + K e = f,   D = 2 sqrt(K M),   semi-implicit Euler
          static:   e = f / K

      e = x_cmd - x_ref [m], positive along the contact force (the fingertip yields away from the
      object); |e| <= max_offset with its direction kept (dynamic: velocity zeroed at the clamp);
      K = inf = rigid (e = 0). At rest both give e = f / K.

  FingertipAdmittance
      What the simulation's FingertipAdmittanceAction did with the law, without its I/O: the force
      on each fingertip as a vector from the glove's scalar and the pad normal, the law, the
      damped-least-squares (DLS) map of e to joint offsets through the fingertip Jacobian at the
      measured pose, and the commanded joint angles. Inputs and outputs in radians; reading the
      glove and the hand and converting to the driver's 0..1 values is tactile_admittance.py's job.

Differences from simulation (agreed with Balázs, 2026-11-25):
  - No smoothing inside the law by default (force_filter_alpha = 1.0): the glove is already
    smoothed per cell by the supervisor's ContactFilter with the same EMA (alpha = 0.3). That EMA
    is upstream of the law but in the same loop, so it also plays the stabilising role the
    filter had for the static law in simulation (see StaticAdmittance.min_stable_stiffness).
  - e lives in the hand frame (r_wrist), where the Jacobians and pad normals are.
  - Control period 8 ms (125 Hz servo loop) instead of 10 ms: the dynamic law's stability limit
    rises to about 3200 N/m at M = 0.3 kg; a finite K above it is refused.
  - Contact point = the centre of mass of each distal link (as PhysX used the tip link's COM in
    simulation), read from the URDF; the fingertip frame's origin is an option.
  - DLS damping 0.003 instead of 0.01 (see AdmittanceParams.ik_damping).
  - M = 0.3 kg as in simulation for now; to be retuned on the powered hand (delays of the glove
    filter and of the driver's 5 Hz command low-pass are not in the simulation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .hand_model import FINGERS, HandModel

N_POINTS = len(FINGERS)


@dataclass
class AdmittanceParams:
    """Parameters of the fingertip admittance. K is not one of them: it is set per episode."""

    law: str = "dynamic"
    """'dynamic' (mass-spring-damper) or 'static' (e = f / K)."""
    mass: float = 0.3
    """Virtual mass M [kg] (dynamic law). Sets the speed of the response, w = sqrt(K / M)."""
    max_offset: float = 0.03
    """Clamp on |e| per fingertip [m]."""
    force_filter_alpha: float = 1.0
    """EMA on the force inside the law, weight of the newest sample. 1.0 = none (the glove is
    already smoothed upstream); 0.3 reproduces the simulation's filter."""
    ik_damping: float = 0.003
    """DLS damping lambda [m]. 0.003 (Balázs, 2026-11-25) instead of the simulation's 0.01: on the
    Hand 2 the fingertip Jacobian's smallest singular value is ~0.02 m/rad at mid flexion, where
    0.01 delivered only ~88 % of e (fingertip rendered stiffer than K); 0.003 loses ~2 %."""
    contact_point: str = "com"
    """Point on the distal link the offset acts on: 'com' (its centre of mass) or 'tip'."""


class DynamicAdmittance:
    """M e'' + D e' + K e = f for P points (port of the simulation's AdmittanceController)."""

    def __init__(self, params: AdmittanceParams, dt: float, num_points: int = N_POINTS) -> None:
        self.p, self.dt, self.num_points = params, dt, num_points
        self._e = np.zeros((num_points, 3))        # offset [m]
        self._e_dot = np.zeros((num_points, 3))    # offset velocity [m/s]
        self._f_filt = np.zeros((num_points, 3))   # filtered force [N]

    @property
    def offset(self) -> np.ndarray:
        return self._e

    @property
    def filtered_force(self) -> np.ndarray:
        return self._f_filt

    @property
    def max_stable_stiffness(self) -> float:
        """Stiffness above which the Euler update goes unstable [N/m]: critically damped
        semi-implicit Euler is stable while w dt < 2 (sqrt(2) - 1) = 0.83."""
        w_max = 2.0 * (math.sqrt(2.0) - 1.0) / self.dt
        return self.p.mass * w_max**2

    def reset(self) -> None:
        self._e[:] = 0.0
        self._e_dot[:] = 0.0
        self._f_filt[:] = 0.0

    def compute(self, f_ext: np.ndarray, stiffness: np.ndarray | float) -> np.ndarray:
        """One step. f_ext: force on each point [N] (P, 3); stiffness [N/m]: scalar or (P,),
        inf = rigid. Returns the offset e [m] (P, 3)."""
        K = np.broadcast_to(np.asarray(stiffness, dtype=float).reshape(-1, 1), (self.num_points, 1))
        # inf = rigid: compute with a dummy K, then force e to zero (inf * 0 would be NaN)
        rigid = ~np.isfinite(K)
        K = np.where(rigid, 1.0, K)

        a = self.p.force_filter_alpha
        self._f_filt = a * np.asarray(f_ext, dtype=float) + (1.0 - a) * self._f_filt

        # semi-implicit Euler: velocity first, then position with the NEW velocity
        M = self.p.mass
        D = 2.0 * np.sqrt(K * M)  # critical damping with zeta = 1, follows K automatically
        e_ddot = (self._f_filt - D * self._e_dot - K * self._e) / M
        self._e_dot = self._e_dot + self.dt * e_ddot
        self._e = self._e + self.dt * self._e_dot

        # clamp |e| per point, keeping its direction; stop the velocity at the clamp
        norm = np.maximum(np.linalg.norm(self._e, axis=-1, keepdims=True), 1e-9)
        scale = np.minimum(self.p.max_offset / norm, 1.0)
        self._e = self._e * scale
        self._e_dot = np.where(scale < 1.0, 0.0, self._e_dot)

        # rigid entries never yield
        self._e = np.where(rigid, 0.0, self._e)
        self._e_dot = np.where(rigid, 0.0, self._e_dot)
        return self._e


class StaticAdmittance:
    """e = f / K for P points (port of the simulation's StaticAdmittanceController)."""

    def __init__(self, params: AdmittanceParams, dt: float, num_points: int = N_POINTS) -> None:
        self.p, self.dt, self.num_points = params, dt, num_points   # dt unused (no dynamics)
        self._e = np.zeros((num_points, 3))
        self._f_filt = np.zeros((num_points, 3))

    @property
    def offset(self) -> np.ndarray:
        return self._e

    @property
    def filtered_force(self) -> np.ndarray:
        return self._f_filt

    def min_stable_stiffness(self, drive_stiffness: float, alpha: float | None = None) -> float:
        """Lowest K for which the contact loop is stable, given the drives' stiffness at the tip
        [N/m]: K_drive / (2 / alpha - 1). `alpha` = the EMA in the loop; on the real hand that is
        the glove filter's 0.3 (upstream), so pass it when the law's own filter is off."""
        a = self.p.force_filter_alpha if alpha is None else alpha
        return drive_stiffness / (2.0 / a - 1.0)

    def reset(self) -> None:
        self._e[:] = 0.0
        self._f_filt[:] = 0.0

    def compute(self, f_ext: np.ndarray, stiffness: np.ndarray | float) -> np.ndarray:
        K = np.broadcast_to(np.asarray(stiffness, dtype=float).reshape(-1, 1), (self.num_points, 1))
        rigid = ~np.isfinite(K)
        K = np.where(rigid, 1.0, K)

        a = self.p.force_filter_alpha
        self._f_filt = a * np.asarray(f_ext, dtype=float) + (1.0 - a) * self._f_filt

        self._e = self._f_filt / K

        norm = np.maximum(np.linalg.norm(self._e, axis=-1, keepdims=True), 1e-9)
        self._e = self._e * np.minimum(self.p.max_offset / norm, 1.0)

        self._e = np.where(rigid, 0.0, self._e)
        return self._e


def dls(J: np.ndarray, e: np.ndarray, damping: float) -> np.ndarray:
    """Damped least squares: joint offset dq (n,) that moves the point by about e (3,),
    dq = J^T (J J^T + lambda^2 I)^-1 e."""
    return J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(J.shape[0]), e)


class FingertipAdmittance:
    """Glove force -> fingertip offset -> joint command, for the five fingers of the Hand 2.

    Call set_stiffness() at the start of an episode and step() once per control tick.
    All joint quantities are (20,) radians in device order (hand_model.FINGERS, 4 joints each).
    """

    def __init__(self, hand: HandModel, params: AdmittanceParams | None = None, dt: float = 1.0 / 125.0) -> None:
        self.hand = hand
        self.p = params or AdmittanceParams()
        if self.p.law == "dynamic":
            self.law = DynamicAdmittance(self.p, dt)
        elif self.p.law == "static":
            self.law = StaticAdmittance(self.p, dt)
        else:
            raise ValueError(f"law must be 'dynamic' or 'static', not {self.p.law!r}")
        if self.p.contact_point == "com":
            self.points = {f: hand.distal_com[f] for f in FINGERS}
        elif self.p.contact_point == "tip":
            self.points = {f: None for f in FINGERS}
        else:
            raise ValueError(f"contact_point must be 'com' or 'tip', not {self.p.contact_point!r}")
        self.stiffness = np.full(N_POINTS, np.inf)
        # Last step, for recording and plots.
        self.force = np.zeros((N_POINTS, 3))       # f on each fingertip [N or glove units], r_wrist
        self.normal = np.zeros((N_POINTS, 3))      # pad normal at the measured pose
        self.joint_offset = np.zeros(20)           # dq [rad]
        self.q_ref = np.zeros(20)
        self.q_cmd = np.zeros(20)
        self.tip_reference_offset = np.zeros((N_POINTS, 3))  # J (q_ref - q): how far the reference reaches in
        self.tip_command_offset = np.zeros((N_POINTS, 3))    # J (q_cmd - q)

    @property
    def offset(self) -> np.ndarray:
        """e per fingertip [m] (5, 3), r_wrist frame."""
        return self.law.offset

    @property
    def max_stable_stiffness(self) -> float:
        return self.law.max_stable_stiffness if isinstance(self.law, DynamicAdmittance) else math.inf

    def set_stiffness(self, stiffness: float | np.ndarray) -> None:
        """K [N/m] for all fingers (scalar) or per finger (5,); inf = admittance off. Resets the law."""
        K = np.broadcast_to(np.asarray(stiffness, dtype=float), (N_POINTS,)).copy()
        finite = K[np.isfinite(K)]
        if np.any(finite <= 0):
            raise ValueError(f"stiffness must be > 0 (or inf), got {K}")
        if finite.size and finite.max() > self.max_stable_stiffness:
            raise ValueError(f"stiffness {finite.max():g} N/m is above the dynamic law's stability limit "
                             f"{self.max_stable_stiffness:.0f} N/m at dt = {self.law.dt * 1000:.1f} ms")
        self.stiffness = K
        self.reset()

    def reset(self) -> None:
        self.law.reset()

    def step(self, q_ref: np.ndarray, q: np.ndarray, force: np.ndarray) -> np.ndarray:
        """One control tick.

        q_ref: the operator's reference (20,) [rad]; q: measured joint angles (20,) [rad];
        force: contact force on each fingertip pad (5,), >= 0 (0 when the glove is not ready).
        Returns q_cmd (20,) [rad], clipped to the hand's command range.
        """
        q_ref = np.asarray(q_ref, dtype=float).reshape(20)
        q = np.asarray(q, dtype=float).reshape(20)
        F = np.clip(np.asarray(force, dtype=float).reshape(N_POINTS), 0.0, None)

        J = {}
        for i, f in enumerate(FINGERS):
            pose = self.hand.finger(q, f)
            self.normal[i] = pose.pad_normal
            J[f] = self.hand.jacobian(q, f, self.points[f])
        # The object pushes the pad inwards: the force on the fingertip is against the pad normal.
        self.force = -F[:, None] * self.normal

        e = self.law.compute(self.force, self.stiffness)

        dq = np.zeros(20)
        for i, f in enumerate(FINGERS):
            sl = self.hand.finger_slice(f)
            dq[sl] = dls(J[f], e[i], self.p.ik_damping)
            self.tip_reference_offset[i] = J[f] @ (q_ref[sl] - q[sl])
        q_cmd = np.clip(q_ref + dq, self.hand.command_lower, self.hand.upper)
        for i, f in enumerate(FINGERS):
            sl = self.hand.finger_slice(f)
            self.tip_command_offset[i] = J[f] @ (q_cmd[sl] - q[sl])
        self.joint_offset, self.q_ref, self.q_cmd = dq, q_ref, q_cmd
        return q_cmd
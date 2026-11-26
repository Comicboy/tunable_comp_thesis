"""The hand's admittance to contact forces measured by the tactile glove.

Per finger f, on its four joints, with e the command minus the operator's reference in
normalised joint units (0..1 spans each joint's range) and F_f the finger's contact force:

    M e'' + D e' + K e = -F_f w_f,      q_cmd = clip(q_ref + e, 0, 1)

w_f spreads the force over the finger's joints, flexion by default, so a contact opens the
finger against the operator's reference by F/K at rest; K is the stiffness. Integrated with
semi-implicit Euler at the servo rate; |e| is capped at `max_deviation` per joint.

Another law plugs in with --admittance-law module:Class: a class built from AdmittanceParams,
with reset() and step(q_ref (20,), force (5,), dt) -> q_cmd (20,), normalised 0..1, thumb..little
in the device's joint order. force is the glove pressure above rest and above the noise floor,
summed over each finger's cells in contact.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from ..tactile.contact import FINGERS, ContactFilter, FingerRegions, load_floor
from ..tactile.glove import GloveClient

N_JOINTS = 20


@dataclass
class AdmittanceParams:
    stiffness: float
    mass: float = 0.2
    damping: float | None = None
    damping_ratio: float = 1.0
    max_deviation: float = 0.4
    weights: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path, **override) -> AdmittanceParams:
        d = yaml.safe_load(Path(path).read_text()) or {}
        d.update({k: v for k, v in override.items() if v is not None})
        return cls(**d)

    @property
    def d(self) -> float:
        """Damping: as given, else damping_ratio of critical, 2 sqrt(K M)."""
        if self.damping is not None:
            return float(self.damping)
        return 2.0 * self.damping_ratio * float(np.sqrt(self.stiffness * self.mass))

    def weight_matrix(self) -> np.ndarray:
        """(5, 4): how each finger's force spreads over its joints."""
        w = np.zeros((5, 4))
        for i, name in enumerate(FINGERS):
            w[i] = self.weights.get(name, [0.577, 0.0, 0.577, 0.577])
        return w

    def check(self) -> None:
        if not self.stiffness > 0 or not self.mass > 0 or self.d < 0 or not self.max_deviation > 0:
            raise ValueError("admittance needs stiffness > 0, mass > 0, damping >= 0, max_deviation > 0")


class HandAdmittance:
    """The law above, joint by joint."""

    def __init__(self, params: AdmittanceParams) -> None:
        params.check()
        self.p = params
        self.w = params.weight_matrix().reshape(-1)
        self.finger = np.repeat(np.arange(5), 4)
        self.reset()

    def reset(self) -> None:
        self.e = np.zeros(N_JOINTS)
        self.v = np.zeros(N_JOINTS)

    def step(self, q_ref: np.ndarray, force: np.ndarray, dt: float) -> np.ndarray:
        """q_cmd (20,) from the reference (20,) and the per-finger force (5,)."""
        f = -np.asarray(force, dtype=np.float64)[self.finger] * self.w
        a = (f - self.p.d * self.v - self.p.stiffness * self.e) / self.p.mass
        self.v += a * dt
        self.e += self.v * dt
        cap = self.p.max_deviation
        hit = np.abs(self.e) > cap
        self.e = np.clip(self.e, -cap, cap)
        self.v[hit] = 0.0
        return np.clip(np.asarray(q_ref, dtype=np.float64) + self.e, 0.0, 1.0)


def law_class(spec: str | None):
    """HandAdmittance, or the class `module:Class` names."""
    if not spec:
        return HandAdmittance
    module, _, name = spec.partition(":")
    if not name:
        raise ValueError(f"--admittance-law takes module:Class, not {spec!r}")
    return getattr(importlib.import_module(module), name)


class TactileAdmittance:
    """Glove -> contacts -> per-finger forces -> admittance, once per servo tick.

    With no stiffness it only reads and records the glove, and commands the reference.
    Until the glove's rest is learned, and whenever its data is stale, the command is the
    reference and the forces are zero.
    """

    def __init__(self, glove: GloveClient, regions: FingerRegions, params: AdmittanceParams | None,
                 dt: float, rest_s: float = 2.0, stale_s: float = 0.1, law: str | None = None,
                 floor_path: str | Path | None = None) -> None:
        self.glove, self.regions, self.params, self.dt = glove, regions, params, dt
        self.law_name = law or "quest_teleop2.control.admittance:HandAdmittance"
        self.law = law_class(law)(params) if params is not None else None
        self.floor_path = floor_path
        floor = load_floor(floor_path)
        self.filter = ContactFilter(rate_hz=float(glove.info.get("rate_hz", 117.0)), floor=floor)
        self.has_floor = floor is not None
        self.stale_s = stale_s
        self.rest_frames = max(10, int(rest_s * float(glove.info.get("rate_hz", 117.0))))
        self._rest_frames: list[np.ndarray] | None = None
        self._seq = None
        self.pressure = np.full((24, 32), np.nan, np.float32)
        self.force = np.zeros(6)
        self.reference = np.zeros(N_JOINTS)
        self.command = np.zeros(N_JOINTS)
        self.stale = True

    def learn_rest(self) -> None:
        """Learn the glove's rest from the next rest_s of frames, with nothing touching it."""
        self._rest_frames = []
        if self.law is not None:
            self.law.reset()

    @property
    def learning(self) -> bool:
        return self._rest_frames is not None

    @property
    def ready(self) -> bool:
        return self.filter.ready and not self.learning and not self.stale

    def step(self, q_ref: np.ndarray) -> np.ndarray:
        frame = self.glove.latest()
        now = time.monotonic()
        self.stale = frame is None or self.glove.lost or now - frame.received > self.stale_s
        if frame is not None and frame.sequence != self._seq:
            self._seq = frame.sequence
            self.pressure = frame.pressure
            if self._rest_frames is not None:
                self._rest_frames.append(frame.pressure)
            elif self.filter.ready:
                lit, d = self.filter.update(frame.pressure)
                self.force = self.regions.forces(d, lit)
        if self._rest_frames is not None and len(self._rest_frames) >= self.rest_frames:
            self.filter.calibrate(self._rest_frames)
            self._rest_frames = None
        if not self.ready:
            self.force = np.zeros(6)
        self.reference = np.asarray(q_ref, dtype=np.float64).reshape(-1)
        if self.law is None or not self.ready:
            if self.law is not None:
                self.law.reset()
            self.command = self.reference.copy()
        else:
            self.command = self.law.step(self.reference, self.force[:5], self.dt)
        return self.command

    def observation(self) -> dict[str, np.ndarray]:
        k = self.params.stiffness if self.params is not None else np.inf
        return {"tactile": self.pressure.astype(np.float32),
                "finger_force": self.force[:5].astype(np.float32),
                "stiffness": np.array([k], dtype=np.float32)}

    def extra(self) -> dict[str, np.ndarray]:
        return {"hand_reference": self.reference.astype(np.float32),
                "hand_admittance": self.command.astype(np.float32),
                "palm_force": np.array([self.force[5]], dtype=np.float32)}

    def attrs(self) -> dict[str, object]:
        a = {"glove_serial": str(self.glove.info.get("serial", "")), "admittance": self.law is not None,
             "glove_noise_floor": str(self.floor_path) if self.has_floor else ""}
        if self.params is not None:
            a["admittance_law"] = self.law_name
            a.update(stiffness=float(self.params.stiffness), admittance_mass=float(self.params.mass),
                     admittance_damping=float(self.params.d),
                     admittance_max_deviation=float(self.params.max_deviation),
                     admittance_weights=self.params.weight_matrix())
        return a

    def status(self) -> str:
        if self.glove.lost:
            return " GLOVE LOST"
        if self.learning:
            return " glove: learning rest, touch nothing"
        if self.stale:
            return " GLOVE STALE"
        k = f"K{self.params.stiffness:g} " if self.params is not None else ""
        return f" {k}F " + " ".join(f"{n[0].upper()}{v:.1f}" for n, v in zip(FINGERS, self.force[:5]))

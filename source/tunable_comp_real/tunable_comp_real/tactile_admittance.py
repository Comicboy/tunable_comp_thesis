"""Our hand_filter for the supervisor's TeleopSession: glove -> fingertip forces -> our admittance.

The real-robot counterpart of the simulation's FingertipAdmittanceAction: once per servo tick
(125 Hz) it reads the glove and the measured hand, runs tunable_comp_real.admittance.
FingertipAdmittance and returns the command. It plugs into
quest_teleop2.teleop.session.TeleopSession as `hand_filter` (the slot his TactileAdmittance
uses) and provides what the session calls: step(), ready, status(), learn_rest(), attrs(),
observation(), extra().

Per tick (step):
  1. glove: newest frame from GloveClient; the supervisor's ContactFilter (rest, common mode,
     contact hysteresis, noise floor if given) and our fingertip map (FingerRegions) -> 5 sums
     of contact pressure, one per fingertip (glove units);
  2. newtons: times each finger's calibration factor (config/real/glove_calibration.yaml; 1 and
     unit "glove" until the calibration exists);
  3. hand: the operator's reference (0..1 from the session) and the measured joint angles
     (MeasuredWujiHand2) in radians;
  4. law: FingertipAdmittance.step -> q_cmd [rad] -> 0..1 for the driver.
While the glove is not ready (rest not learned, learning, lost or stale) the force is 0: the law
keeps running and relaxes smoothly instead of being reset (agreed 2026-11-25; the supervisor's
filter resets it, which snaps the hand back to the reference).

Per-tick history (agreed 2026-11-26): the session records every 5th tick (25 Hz), but the law runs
every tick (125 Hz). extra() therefore also carries the last `ticks_per_record` ticks, oldest first
and ending with the recorded tick, of everything the law needs to be re-run exactly offline: the
fingertip forces, the reference, the measured angles, the command, the offset e and each tick's
real interval. Consecutive samples' windows join up without gaps or overlap, so an episode's
extra/*_ticks reshaped to (T * ticks_per_record, ...) is the full 125 Hz record. The offset
velocity is not stored: the semi-implicit Euler step gives it from two consecutive offsets,
de/dt_k = (e_k - e_{k-1}) / dt, except on the tick the 30 mm clamp acts.

Units: everything recorded by this class is in the simulation's units (radians, metres, newtons
once calibrated); the session itself records the driver's 0..1 values next to them. K is never
shown in status() (hidden from the operator).
"""

from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from quest_teleop2.tactile.contact import FINGERS, ContactFilter, FingerRegions, load_floor
from quest_teleop2.tactile.glove import GloveClient

from .admittance import AdmittanceParams, FingertipAdmittance
from .hand_model import HandModel

N_JOINTS = 20
LATE_TICK_S = 0.012  # a tick this much longer than nominal counts as late in the statistics


@dataclass
class GloveCalibration:
    """Glove units -> force per finger: F = factor * (sum of contact pressure on the fingertip)."""

    unit: str = "glove"
    factor: dict[str, float] = field(default_factory=lambda: {f: 1.0 for f in FINGERS})
    path: str = ""

    @classmethod
    def load(cls, path: str | Path | None) -> GloveCalibration:
        """From a yaml file `unit: N` / `factor: {thumb: .., ..}`; the identity if path is None or missing."""
        if path is None or not Path(path).exists():
            return cls()
        d = yaml.safe_load(Path(path).read_text()) or {}
        factor = {f: float(d.get("factor", {}).get(f, np.nan)) for f in FINGERS}
        if any(not np.isfinite(v) or v <= 0 for v in factor.values()):
            raise ValueError(f"{path}: needs a positive factor for every finger, got {factor}")
        return cls(unit=str(d.get("unit", "N")), factor=factor, path=str(path))

    @property
    def factors(self) -> np.ndarray:
        return np.array([self.factor[f] for f in FINGERS])


class TactileFingertipAdmittance:
    """hand_filter: glove + measured hand -> our fingertip admittance, once per servo tick."""

    #: Keys this filter adds to every recorded tick (declare them as more_obs_keys /
    #: more_extra_keys of the supervisor's EpisodeRecorder).
    OBS_KEYS = ("tactile", "fingertip_force", "stiffness", "hand_q")
    #: Per-tick history in extra(), each recorded as (ticks_per_record, *shape): the fingertip
    #: forces, reference, measured angles and command [rad], offset e [m], tick interval [s].
    TICK_SHAPES = {"fingertip_force_ticks": (5,), "hand_q_ref_ticks": (N_JOINTS,),
                   "hand_q_ticks": (N_JOINTS,), "hand_q_cmd_ticks": (N_JOINTS,),
                   "tip_offset_ticks": (5, 3), "tick_interval_ticks": ()}
    TICK_KEYS = tuple(TICK_SHAPES)
    EXTRA_KEYS = ("hand_q_ref", "hand_q_cmd", "hand_dq", "tip_offset", "tip_force", "pad_normal",
                  "tip_reference_offset", "tip_command_offset", "contact_cells", "not_tip_force",
                  "glove_age", "hand_age", "tick_interval", "tick_interval_max") + TICK_KEYS

    def __init__(self, glove: GloveClient, hand, model: HandModel, tips: FingerRegions,
                 params: AdmittanceParams | None = None, dt: float = 1.0 / 125.0,
                 calibration: GloveCalibration | None = None, regions_path: str | Path = "",
                 floor_path: str | Path | None = None, rest_s: float = 2.0, stale_s: float = 0.1,
                 ticks_per_record: int = 5) -> None:
        """glove: a started GloveClient; hand: the connected driver (MeasuredWujiHand2, or any
        object with state(), to_radians(), to_normalised() and optionally measured_radians()).
        ticks_per_record: servo ticks per recorded sample (robot_cfg.control.record_decimation)."""
        self.glove, self.hand, self.model, self.tips = glove, hand, model, tips
        self.law = FingertipAdmittance(model, params, dt)
        self.dt = dt
        self.calibration = calibration or GloveCalibration()
        self.regions_path = str(regions_path)
        self.floor_path = str(floor_path) if floor_path else ""
        floor = load_floor(floor_path)
        rate = float(glove.info.get("rate_hz", 117.0))
        self.filter = ContactFilter(rate_hz=rate, floor=floor)
        self.has_floor = floor is not None
        self.stale_s = stale_s
        self.rest_frames = max(10, int(rest_s * rate))
        self._rest: list[np.ndarray] | None = None
        self._seq = None
        self.stale = True
        # Last tick, for observation() / extra() / status().
        self.pressure = np.full((24, 32), np.nan, np.float32)
        self.region_sums = np.zeros(len(FINGERS) + 1)   # glove units, 5 fingertips + not a fingertip
        self.force = np.zeros(len(FINGERS))             # calibrated fingertip forces
        self.contact_cells = np.zeros(len(FINGERS))
        self.glove_age = np.inf
        self.q = np.zeros(N_JOINTS)
        # Tick timing (dt is fixed; the real intervals are only logged).
        self._last_tick: float | None = None
        self.tick_interval = np.nan
        self._interval_max = 0.0
        self.ticks = 0
        self.late_ticks = 0
        self._interval_sum = 0.0
        # The last ticks_per_record ticks, for extra()'s *_ticks keys.
        self.ticks_per_record = int(ticks_per_record)
        self._history: deque[dict[str, np.ndarray]] = deque(maxlen=self.ticks_per_record)

    # -- episode control ----------------------------------------------------------------------- #

    def set_stiffness(self, stiffness: float | np.ndarray) -> None:
        """K [N/m] for this episode (scalar or per finger; inf = off). Resets the law. Call at
        episode start; never shown to the operator."""
        self.law.set_stiffness(stiffness)

    @property
    def stiffness(self) -> np.ndarray:
        return self.law.stiffness

    def learn_rest(self) -> None:
        """Learn the glove's rest from the next rest_s of frames; nothing may touch the glove."""
        self._rest = []

    @property
    def learning(self) -> bool:
        return self._rest is not None

    @property
    def ready(self) -> bool:
        return self.filter.ready and not self.learning and not self.stale

    # -- the tick -------------------------------------------------------------------------------- #

    def _read_glove(self) -> None:
        frame = self.glove.latest()
        now = time.monotonic()
        self.glove_age = np.inf if frame is None else now - frame.received
        self.stale = frame is None or self.glove.lost or self.glove_age > self.stale_s
        if frame is not None and frame.sequence != self._seq:
            self._seq = frame.sequence
            self.pressure = frame.pressure
            if self._rest is not None:
                self._rest.append(frame.pressure)
            elif self.filter.ready:
                lit, d = self.filter.update(frame.pressure)
                self.region_sums = self.tips.forces(d, lit)
                self.contact_cells = np.array([(lit & m).sum() for m in self.tips.masks[:len(FINGERS)]], float)
        if self._rest is not None and len(self._rest) >= self.rest_frames:
            self.filter.calibrate(self._rest)
            self._rest = None
        if not self.ready:
            self.region_sums = np.zeros(len(FINGERS) + 1)
            self.contact_cells = np.zeros(len(FINGERS))

    def _time_tick(self) -> None:
        now = time.monotonic()
        if self._last_tick is not None:
            self.tick_interval = now - self._last_tick
            self._interval_max = max(self._interval_max, self.tick_interval)
            self.ticks += 1
            self._interval_sum += self.tick_interval
            self.late_ticks += self.tick_interval > LATE_TICK_S
        self._last_tick = now

    def _measured_q(self) -> np.ndarray:
        u = self.hand.state()                            # reads the stream (once per tick)
        q = getattr(self.hand, "measured_radians", lambda: None)()
        if q is None:
            q = np.asarray(self.hand.to_radians(u), dtype=float)
        return np.asarray(q, dtype=float).reshape(N_JOINTS)

    def step(self, q_ref: np.ndarray) -> np.ndarray:
        """One servo tick: the operator's reference (20,) in 0..1 -> the command (20,) in 0..1."""
        self._time_tick()
        self._read_glove()
        self.force = self.calibration.factors * self.region_sums[:len(FINGERS)]  # 0 when not ready
        q_ref_rad = np.asarray(self.hand.to_radians(q_ref), dtype=float).reshape(N_JOINTS)
        self.q = self._measured_q()
        q_cmd = self.law.step(q_ref_rad, self.q, self.force)
        self._history.append({
            "fingertip_force_ticks": self.force.copy(),          # (5,)
            "hand_q_ref_ticks": q_ref_rad.copy(),                 # (20,) rad
            "hand_q_ticks": self.q.copy(),                        # (20,) rad, measured
            "hand_q_cmd_ticks": np.asarray(q_cmd, float).copy(),  # (20,) rad
            "tip_offset_ticks": self.law.offset.copy(),           # (5, 3) m
            "tick_interval_ticks": np.array(self.tick_interval),  # () s; nan on the very first tick
        })
        return np.asarray(self.hand.to_normalised(q_cmd), dtype=float).reshape(N_JOINTS)

    def _tick_history(self) -> dict[str, np.ndarray]:
        """The last ticks_per_record ticks per key, oldest first; nan-padded at the front until
        that many ticks have run."""
        out = {}
        for key, shape in self.TICK_SHAPES.items():
            rows = [h[key] for h in self._history]
            pad = [np.full(shape, np.nan)] * (self.ticks_per_record - len(rows))
            out[key] = np.stack(pad + rows).astype(np.float32)
        return out

    # -- what the session records ---------------------------------------------------------------- #

    def observation(self) -> dict[str, np.ndarray]:
        return {
            "tactile": np.asarray(self.pressure, np.float32),                     # (24, 32), raw
            "fingertip_force": self.force.astype(np.float32),                     # (5,) N (or glove units)
            "stiffness": self.law.stiffness.astype(np.float32),                   # (5,) N/m, inf = off
            "hand_q": self.q.astype(np.float32),                                  # (20,) rad, measured
        }

    def extra(self) -> dict[str, np.ndarray]:
        hand_age = getattr(self.hand, "measurement_age", np.nan)
        out = {
            "hand_q_ref": self.law.q_ref.astype(np.float32),                     # (20,) rad
            "hand_q_cmd": self.law.q_cmd.astype(np.float32),                     # (20,) rad
            "hand_dq": self.law.joint_offset.astype(np.float32),                 # (20,) rad
            "tip_offset": self.law.offset.astype(np.float32),                    # (5, 3) m, e
            "tip_force": self.law.force.astype(np.float32),                      # (5, 3) force vectors
            "pad_normal": self.law.normal.astype(np.float32),                    # (5, 3)
            "tip_reference_offset": self.law.tip_reference_offset.astype(np.float32),  # (5, 3) m
            "tip_command_offset": self.law.tip_command_offset.astype(np.float32),      # (5, 3) m
            "contact_cells": self.contact_cells.astype(np.float32),              # (5,) lit fingertip cells
            "not_tip_force": np.array([self.region_sums[-1]], np.float32),      # glove units
            "glove_age": np.array([self.glove_age], np.float32),                 # s
            "hand_age": np.array([hand_age], np.float32),                        # s
            "tick_interval": np.array([self.tick_interval], np.float32),         # s, last tick
            "tick_interval_max": np.array([self._interval_max], np.float32),     # s, since the last record
        }
        out.update(self._tick_history())                                         # (ticks_per_record, ...)
        self._interval_max = 0.0
        return out

    def attrs(self) -> dict[str, object]:
        p = self.law.p
        regions = Path(self.regions_path)
        digest = hashlib.sha1(regions.read_bytes()).hexdigest()[:12] if regions.is_file() else ""
        return {
            "admittance": "tunable_comp_real.FingertipAdmittance",
            "admittance_law": p.law,
            "stiffness": self.law.stiffness.astype(float),
            "admittance_mass": float(p.mass),
            "admittance_max_offset": float(p.max_offset),
            "admittance_force_filter_alpha": float(p.force_filter_alpha),
            "admittance_ik_damping": float(p.ik_damping),
            "admittance_contact_point": p.contact_point,
            "admittance_dt": float(self.dt),
            "ticks_per_record": int(self.ticks_per_record),
            "force_unit": self.calibration.unit,
            "glove_calibration": self.calibration.path,
            "glove_calibration_factors": self.calibration.factors,
            "glove_regions": self.regions_path,
            "glove_regions_sha1": digest,
            "glove_noise_floor": self.floor_path if self.has_floor else "",
            "glove_serial": str(self.glove.info.get("serial", "")),
            "hand_urdf": str(self.model.urdf),
            "hand_q_lower": self.model.command_lower.astype(float),
            "hand_q_upper": self.model.upper.astype(float),
            "hand_joint_names": np.array(self.model.joint_names, dtype="S"),
        }

    def tick_stats(self) -> dict[str, float]:
        """Real tick intervals since start: count, mean [s], late ticks (> LATE_TICK_S)."""
        mean = self._interval_sum / self.ticks if self.ticks else np.nan
        return {"ticks": self.ticks, "mean_interval": mean, "late_ticks": self.late_ticks}

    def status(self) -> str:
        """For the operator's status line: glove state and fingertip forces, never K."""
        if self.glove.lost:
            return " GLOVE LOST"
        if self.learning:
            return " glove: learning rest, touch nothing"
        if self.stale:
            return " GLOVE STALE"
        unit = "N" if self.calibration.unit == "N" else "g.u."
        return f" F[{unit}] " + " ".join(f"{n[0].upper()}{v:.1f}" for n, v in zip(FINGERS, self.force))
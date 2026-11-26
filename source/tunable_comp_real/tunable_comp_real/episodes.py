"""Episode logic of the real-robot recording: success check, one recorder per K, the hidden K draw.

The real-robot counterpart of the episode handling in the simulation's
scripts/teleop/teleop_lift.py (agreed with Balázs 2026-11-26):

  * every demo starts at the supervisor's home pose, the cube on one fixed spot; recording starts
    at the left pinch + release and ends by itself on success;
  * success = the cube held 15 cm above the table for 3 s. The real robot has no object state,
    so SuccessDetector uses a proxy from what we measure: the cube is grasped when the thumb
    and at least one other fingertip press on something (glove forces), and it is lifted by
    how far the TCP has risen since the grasp formed (the cube was on the table then);
  * K is drawn per episode, weighted by the demos each K still lacks, and hidden from the
    operator; one HDF5 file per K (as in sim), train / val in separate folders;
  * `d` deletes the last demo saved in this session (the supervisor's recorder cannot delete).

Nothing here touches hardware: session.RealTeleopSession calls these classes from the control
loop, and scripts/real/test_episodes.py tests them offline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from quest_teleop2.data.recorder import EpisodeRecorder

#: Stored in every file we write; a file with another value is never appended to.
FORMAT = ("tunable_comp_real v1: actions = [arm target_q (6) | hand command 0..1 (20)]; "
          "obs/extra of TactileFingertipAdmittance in rad, m, N (or glove units)")


# ------------------------------------------------------------------------------------------------ #
# success
# ------------------------------------------------------------------------------------------------ #

@dataclass
class SuccessParams:
    """When a lift counts as a success. Defaults as the sim's success term (15 cm for 3 s)."""

    #: Fingertip force that counts as contact, in the glove's force unit (N once calibrated; the
    #: sim's perturbations used 0.5 N). Before the calibration: glove units, set from test grasps.
    contact_threshold: float = 0.5
    #: Lift of the TCP since the grasp formed, m.
    lift_m: float = 0.15
    #: How long grasped and lifted must hold without a break, s.
    hold_s: float = 3.0
    #: A grasp counts as lost only after this long without contact, s: the glove's contact
    #: detection can drop a frame or two while the grip is unchanged.
    release_s: float = 0.2


class SuccessDetector:
    """Grasped + lifted >= lift_m, continuously for hold_s. Call update() once per servo tick."""

    def __init__(self, params: SuccessParams | None = None) -> None:
        self.p = params or SuccessParams()
        self.reset()

    def reset(self) -> None:
        """New episode."""
        self.z0: float | None = None      # TCP height when the grasp formed, m
        self.grasped = False
        self.lift = 0.0                   # m
        self.hold = 0.0                   # s grasped and lifted without a break
        self.max_lift = 0.0               # m, largest lift of this episode
        self.succeeded = False
        self._lost_for = 0.0
        self._t: float | None = None

    def contact(self, force: np.ndarray) -> bool:
        """Thumb and at least one of the four fingers above the threshold."""
        F = np.asarray(force, dtype=float).reshape(5)
        thr = self.p.contact_threshold
        return bool(F[0] > thr and np.any(F[1:] > thr))

    def update(self, force: np.ndarray, tcp_z: float, now: float) -> bool:
        """force: the 5 fingertip forces (thumb..little); tcp_z: measured TCP height, m; now: a
        monotonic time, s. Returns True from the tick the success condition is met."""
        dt = 0.0 if self._t is None else max(now - self._t, 0.0)
        self._t = now
        if self.contact(force):
            self._lost_for = 0.0
            if self.z0 is None:
                self.z0 = float(tcp_z)    # the grasp forms with the cube still on the table
        elif self.z0 is not None:
            self._lost_for += dt
            if self._lost_for > self.p.release_s:
                self.z0 = None            # let go (or dropped): a new grasp sets a new reference
        self.grasped = self.z0 is not None
        self.lift = float(tcp_z) - self.z0 if self.grasped else 0.0
        self.max_lift = max(self.max_lift, self.lift)
        self.hold = self.hold + dt if (self.grasped and self.lift >= self.p.lift_m) else 0.0
        self.succeeded = self.succeeded or self.hold >= self.p.hold_s
        return self.succeeded

    def status(self) -> str:
        """For the operator's status line (says nothing about K)."""
        if not self.grasped:
            return " grasp: no"
        return f" lift {100 * self.lift:4.1f}/{100 * self.p.lift_m:.0f} cm, hold {self.hold:3.1f}/{self.p.hold_s:g} s"

    def attrs(self) -> dict[str, float]:
        """Per-episode attributes: the criterion and what this episode reached."""
        return {"success_contact_threshold": self.p.contact_threshold, "success_lift_m": self.p.lift_m,
                "success_hold_s": self.p.hold_s, "success_release_s": self.p.release_s,
                "max_lift_m": self.max_lift}


# ------------------------------------------------------------------------------------------------ #
# the hidden K
# ------------------------------------------------------------------------------------------------ #

def draw_stiffness(k_values: list[float], missing: dict[float, int] | None,
                   rng: np.random.Generator) -> float:
    """The next episode's K: weighted by the demos each K still lacks (as the sim's draw_k), so the
    sets end up balanced while the order stays unpredictable; uniform without targets."""
    if missing:
        m = np.array([max(missing.get(k, 0), 0) for k in k_values], dtype=float)
        if m.sum() > 0:
            return float(rng.choice(k_values, p=m / m.sum()))
    return float(rng.choice(k_values))


def k_name(k: float) -> str:
    return "inf" if math.isinf(k) else f"{k:g}"


def demo_path(folder: str | Path, k: float) -> Path:
    """One file per K, named as in sim: <folder>/lift_K<K>.hdf5."""
    return Path(folder) / f"lift_K{k_name(k)}.hdf5"


# ------------------------------------------------------------------------------------------------ #
# recorders
# ------------------------------------------------------------------------------------------------ #

class DeletableRecorder(EpisodeRecorder):
    """The supervisor's EpisodeRecorder, plus deleting its last demo and file attributes that keep
    other recordings out of the file (subclassed, not edited: option A', notes section 4).

    file_attrs (e.g. format, split, stiffness) are written to the file's data group on every save;
    opening a file whose stored values differ, or a non-empty file without them, is refused, so
    training and validation demos, or demos of two K values, never end up in one file.
    """

    MUST_MATCH = ("format", "split", "stiffness", "task")

    def __init__(self, path: str | Path, task, file_attrs: dict[str, object], **kwargs) -> None:
        super().__init__(path, task, **kwargs)
        self.file_attrs = dict(file_attrs)
        self._check_file()

    def _check_file(self) -> None:
        if not self.path.exists():
            return
        with h5py.File(self.path, "r") as f:
            if "data" not in f or len(f["data"]) == 0:
                return
            stored = f["data"].attrs
            if "format" not in stored:
                raise ValueError(f"{self.path} holds demos not written by tunable_comp_real; "
                                 "record to another file")
            for key in self.MUST_MATCH:
                if key not in self.file_attrs:
                    continue
                old, new = stored.get(key), self.file_attrs[key]
                same = (old is not None and np.isclose(float(old), float(new), equal_nan=True)
                        if isinstance(new, float) else _as_str(old) == str(new))
                if not same:
                    raise ValueError(f"{self.path} was recorded with {key} = {_as_str(old)}, "
                                     f"this run has {new}; record to another file")

    def save_episode(self, success: bool = True) -> int:
        index = super().save_episode(success)
        with h5py.File(self.path, "a") as f:
            for key, value in self.file_attrs.items():
                f["data"].attrs[key] = value
        return index

    def delete_demo(self, index: int) -> None:
        """Delete demo_<index>, which must be the last one in the file (the supervisor's recorder
        names the next demo demo_<n_saved>, so only the last can go without leaving a hole)."""
        if self.is_recording:
            raise RuntimeError("stop the episode before deleting a demo")
        if index != self.n_saved - 1:
            raise ValueError(f"demo_{index} is not the last demo of {self.path} ({self.n_saved} demos)")
        with h5py.File(self.path, "a") as f:
            data = f["data"]
            n = int(data[f"demo_{index}"].attrs["num_samples"])
            del data[f"demo_{index}"]
            data.attrs["total"] = int(data.attrs.get("total", 0)) - n
        self._n_saved -= 1   # the parent's counter, which names the next demo


def _as_str(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


class StiffnessRecorders:
    """One DeletableRecorder per K behind the session's Recorder interface.

    select(k) picks the file of the coming episode; everything the session calls goes to it.
    n_saved is the total over all K, and save_episode returns a running number, so nothing the
    session shows the operator tells which file (which K) a demo went to. Per-episode attributes
    put in `pending` before an episode are applied when it starts (the parent's start_episode
    clears them, as in the supervisor's record_grasps.PendingAttrs).
    """

    def __init__(self, recorders: dict[float, DeletableRecorder], target_per_k: int | None = None) -> None:
        self.by_k = dict(recorders)
        self.target_per_k = target_per_k
        self.k: float | None = None
        self.pending: dict[str, object] = {}
        self._saved: list[tuple[float, int]] = []   # (K, index) of every demo saved this session

    # -- choosing the file ------------------------------------------------------------------- #

    def select(self, k: float) -> None:
        if self.is_recording:
            raise RuntimeError("cannot change K while recording")
        if k not in self.by_k:
            raise KeyError(f"no recorder for K = {k_name(k)}")
        self.k = k

    @property
    def current(self) -> DeletableRecorder:
        if self.k is None:
            raise RuntimeError("no K selected")
        return self.by_k[self.k]

    @property
    def counts(self) -> dict[float, int]:
        return {k: r.n_saved for k, r in self.by_k.items()}

    @property
    def missing(self) -> dict[float, int] | None:
        """Demos each K still lacks; None without a target."""
        if self.target_per_k is None:
            return None
        return {k: max(self.target_per_k - n, 0) for k, n in self.counts.items()}

    @property
    def targets_reached(self) -> bool:
        m = self.missing
        return m is not None and all(v == 0 for v in m.values())

    # -- the session's Recorder interface ------------------------------------------------------ #

    @property
    def is_recording(self) -> bool:
        return any(r.is_recording for r in self.by_k.values())

    @property
    def n_saved(self) -> int:
        return sum(self.counts.values())

    @property
    def n_frames(self) -> int:
        return self.current.n_frames if self.k is not None else 0

    def start_episode(self, start_idx: int, goal_idx: int) -> None:
        self.current.start_episode(start_idx, goal_idx)
        if self.pending:
            self.current.set_episode_attrs(**self.pending)

    def set_episode_attrs(self, **attrs: object) -> None:
        self.current.set_episode_attrs(**attrs)

    def add(self, obs, action, extra, images=None) -> None:
        self.current.add(obs, action, extra, images=images)

    def save_episode(self, success: bool = True) -> int:
        index = self.current.save_episode(success)
        self._saved.append((self.k, index))
        return self.n_saved - 1

    def discard_episode(self) -> None:
        if self.k is not None:
            self.current.discard_episode()

    # -- ours ------------------------------------------------------------------------------------ #

    def delete_last(self) -> tuple[float, int] | None:
        """Delete the last demo saved in this session; returns its (K, index), None if there is none.
        Saves are undone in reverse order, so each one is the last demo of its file."""
        if not self._saved:
            return None
        k, index = self._saved.pop()
        self.by_k[k].delete_demo(index)
        return k, index

    def summary(self) -> str:
        return ", ".join(f"K = {k_name(k)}: {n}" for k, n in self.counts.items())
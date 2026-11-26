"""RealTeleopSession: the supervisor's TeleopSession with our episode flow (record_real.py).

Subclassed, not edited (option A', notes section 4). His loop, safety layer, retargeting,
recording alignment and hand_filter slot run unchanged; this class adds the episode flow agreed
with Balázs on 2026-11-26, modelled on the simulation's scripts/teleop/teleop_lift.py:

  home (h)   fingers ramped to the neutral pose with the servo stopped (so a held cube drops from
             low height), then his go_home (arm to home_q, force sensor zeroed, anchor dropped,
             limiter reset); then the glove rest is relearned (touch nothing for 2 s), a new K is
             drawn and set, hidden, and the arm stays disengaged.
  start      left pinch, then release (as the sim's start gate): engaged, recording starts.
             Needs the glove ready. SPACE is ignored while waiting, so it cannot skip the gate.
  success    SuccessDetector (episodes.py): grasped and lifted >= 15 cm for 3 s -> the demo is
             saved by itself. Teleop stays live and unrecorded: put the cube down, let go, h.
  r          discard the episode (if one runs) and go home. Also on the time limit and when the
             right hand's tracking is lost (his session then opens the hand: the demo is spoilt).
  d          delete the last demo saved in this session.
  t, s, f    off: episodes start with the gate and end on success only.
  q          quit (his).

Why these methods:
  * _handle_events(snap, state) is his only method called every tick with both the Quest snapshot
    (clutch, gate pinch, hand) and the robot state (TCP pose), at the point where his own episode
    starts happen, before the arm and hand are commanded. Ours runs his, then the episode logic.
  * go_home is his `h`; ours wraps it.
  * key_hook (his extension point) handles the keys; `d` is added to KEYS.
  * _print_status is replaced by a line with our episode state; nothing on it reveals K.
While waiting at home the hand holds the neutral pose: the frozen branch of his loop repeats the
last reference, which is set to the neutral pose here, and his tracking-lost fallback (the open
pose) is replaced by the neutral pose until the episode starts.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable

import numpy as np

from quest_teleop2.teleop.session import TeleopSession

from .episodes import StiffnessRecorders, SuccessDetector, draw_stiffness, k_name

_log = logging.getLogger(__name__)

WAITING, RECORDING, DONE, FINISHED = "waiting", "recording", "done", "finished"


class RealTeleopSession(TeleopSession):
    """TeleopSession + home / pinch-release start / automatic success / hidden K per episode."""

    KEYS = {**TeleopSession.KEYS, "d": "delete_last"}

    def __init__(self, *args, k_values: list[float], success: SuccessDetector,
                 rng: np.random.Generator | None = None, time_limit_s: float = 60.0,
                 park_s: float = 1.0, show_k: bool = False,
                 episode_attrs: dict[str, object] | None = None,
                 clock: Callable[[], float] = time.monotonic, **kwargs) -> None:
        """Arguments of TeleopSession, plus:
        k_values: the stiffnesses drawn from [N/m]; success: the success check;
        time_limit_s: an episode running longer is discarded; park_s: finger ramp to neutral;
        show_k: print K (practice only); episode_attrs: stored with every demo (cube spot,
        hand cutoff, ...); clock: monotonic seconds (replaceable for offline tests).
        The recorder, if any, must be a StiffnessRecorders; the hand_filter is required."""
        super().__init__(*args, **kwargs)
        if self.hand_filter is None:
            raise ValueError("RealTeleopSession needs the hand_filter (TactileFingertipAdmittance)")
        if self.recorder is not None and not isinstance(self.recorder, StiffnessRecorders):
            raise TypeError("the recorder must be a StiffnessRecorders (one file per K)")
        self.k_values = [float(k) for k in k_values]
        self.success = success
        self.rng = rng or np.random.default_rng()
        self.time_limit_s = time_limit_s
        self.park_s = park_s
        self.show_k = show_k
        self.episode_attrs = dict(episode_attrs or {})
        self.clock = clock
        self.key_hook = self._on_key
        self.phase = WAITING
        self.k: float | None = None
        self.episodes_started = 0
        self._open_pose = self._hand_open_pose.copy()   # his tracking-lost fallback, restored at start
        self._seen_pinch = False
        self._t_home = self._t_start = 0.0
        self._late0 = 0

    @property
    def recording(self) -> bool:
        return self.recorder is not None and self.recorder.is_recording

    # -- home ------------------------------------------------------------------------------------ #

    def prepare(self) -> None:
        """At home, before the next episode: relearn the glove rest, draw K, wait for the gate.
        Call once after the script's own first homing (hand parked, servo started), before run()."""
        neutral = np.asarray(self.gripper.neutral_pose, dtype=np.float64).reshape(-1)
        # His loop repeats the last reference while frozen, and opens the hand when tracking is
        # lost; at home both should hold the neutral pose the hand was parked in.
        self._last_reference = neutral.copy()
        self._last_gripper_cmd = neutral.copy()
        self._hand_open_pose = neutral.copy()
        self.engaged = False
        self._seen_pinch = False
        self.success.reset()
        self.hand_filter.learn_rest()
        self._t_home = self.clock()
        if self.recorder is not None and self.recorder.targets_reached:
            self.phase = FINISHED
            print(f"\n  all demos recorded ({self.recorder.summary()}); q quits")
            self.server.banner("all demos recorded", "ok")
            return
        missing = self.recorder.missing if self.recorder is not None else None
        self.k = draw_stiffness(self.k_values, missing, self.rng)
        self.hand_filter.set_stiffness(self.k)
        if self.recorder is not None:
            self.recorder.select(self.k)
        self.phase = WAITING
        shown = f" K = {k_name(self.k)} N/m." if self.show_k else ""
        print(f"\n  at home.{shown} Glove: learning its rest, touch nothing. "
              "Then place the cube; left pinch + release starts the episode.")
        self.server.banner("touch nothing, then pinch + release", "idle")

    def go_home(self) -> None:
        """Fingers to neutral (servo stopped), his go_home, then prepare() the next episode."""
        if self.recording:
            self._discard("homing")
        print("\n  parking the fingers...", end="", flush=True)
        self.arm.stop_servo()     # ramp_to blocks; the RTDE watchdog only counts while servoing
        self.gripper.ramp_to(self.gripper.neutral_pose, seconds=self.park_s)
        super().go_home()         # his: rehome (servo stopped: moves, zeros the force sensor), clear
        self.arm.start_servo()    # anchor, reset limiter + pinch, mark the tick stale
        self.prepare()

    # -- the episode, every tick ----------------------------------------------------------------- #

    def _handle_events(self, snap, state) -> None:
        super()._handle_events(snap, state)
        if self.phase == WAITING:
            self._wait_for_gate(snap)
        elif self.phase == RECORDING:
            self._check_episode(snap, state)

    def _wait_for_gate(self, snap) -> None:
        """Left pinch, then release, with the glove ready -> start."""
        if not self.hand_filter.ready:
            self._seen_pinch = False
            return
        # DualHandInput reports the gate pinch whether engaged or not; its age is controller_age.
        gate_fresh = snap.gate_hand is not None and snap.controller_age < 0.3
        if snap.gate_pinching:
            self._seen_pinch = True
        elif self._seen_pinch and gate_fresh:
            self._start()

    def _start(self) -> None:
        now = self.clock()
        self._hand_open_pose = self._open_pose.copy()
        self.success.reset()
        if self.recorder is not None:
            self.recorder.pending = {**self.episode_attrs, "idle_s": now - self._t_home,
                                     "k_values": np.asarray(self.k_values, float),
                                     "k_hidden": not self.show_k,
                                     "time_limit_s": float(self.time_limit_s)}
            self._start_episode()            # his: checks the glove and cameras, resets the tick
            if not self.recorder.is_recording:
                self._seen_pinch = False      # refused (banner says why); wait for the next gate
                return
        self.engaged = True                  # the clutch rises on the next tick: anchor + finger ease
        self.phase = RECORDING
        self.episodes_started += 1
        self._t_start = now
        self._late0 = self.hand_filter.late_ticks
        print("\n  episode started" + ("" if self.recorder is not None else " (practice, not recorded)"))

    def _check_episode(self, snap, state) -> None:
        now = self.clock()
        if snap.hand is None:
            self._discard("tracking lost")
            self.go_home()
            return
        if now - self._t_start > self.time_limit_s:
            self._discard(f"time limit {self.time_limit_s:g} s")
            self.go_home()
            return
        if self.success.update(self.hand_filter.force, float(state.actual_tcp_pose[2]), now):
            self._succeed(now)

    def _succeed(self, now: float) -> None:
        self.phase = DONE
        if self.recorder is None:
            print("\n  SUCCESS (practice). Put the cube down, let go, h for the next episode.")
            self.server.banner("SUCCESS", "ok")
            return
        self.recorder.set_episode_attrs(**self.success.attrs(), duration_s=now - self._t_start,
                                        late_ticks=int(self.hand_filter.late_ticks - self._late0))
        n_before = self.recorder.n_saved
        self._save_episode(success=True)     # his: refuses < 10 frames, banners, haptic
        if self.recorder.n_saved > n_before:
            print(f"\n  SUCCESS: demo saved ({self.recorder.n_saved} in the files). "
                  "Put the cube down, let go, h for the next episode.")
        if self.recorder.targets_reached:
            self.phase = FINISHED
            print(f"  every K has its demos ({self.recorder.summary()}): recording finished; "
                  "put the cube down, then q")

    def _discard(self, why: str) -> None:
        if self.recording:
            self._discard_episode()
            print(f"\n  episode discarded ({why})")
        elif self.phase == RECORDING:
            print(f"\n  episode ended ({why})")
        self.phase = WAITING

    # -- keys -------------------------------------------------------------------------------------- #

    def _on_key(self, action: str, state) -> bool:
        """key_hook: True consumes the key."""
        if action == "home":
            self.go_home()
        elif action == "discard":
            self._discard("r")
            self.go_home()
        elif action in ("start", "save", "save_failure"):
            self.server.banner("start: pinch + release; saved on success", "bad")
        elif action == "engage" and self.phase == WAITING:
            self.server.banner("pinch + release to start", "bad")
        elif action == "delete_last":
            self._delete_last()
        else:
            return False
        return True

    def _delete_last(self) -> None:
        if self.recorder is None:
            return
        if self.recording:
            self.server.banner("stop the episode first", "bad")
            return
        deleted = self.recorder.delete_last()
        if deleted is None:
            print("\n  nothing to delete (no demo saved in this session)")
            return
        where = "" if not self.show_k else f" (K = {k_name(deleted[0])}, demo_{deleted[1]})"
        print(f"\n  deleted the last saved demo{where}; {self.recorder.n_saved} in the files")
        self.server.banner("last demo deleted", "warn")
        if self.phase == FINISHED and not self.recorder.targets_reached:
            self.phase = DONE                # h starts episodes again

    # -- status line ------------------------------------------------------------------------------ #

    def _print_status(self, snap, verdict_reason: str) -> None:
        """His status line, with our episode state instead of the task label; never K."""
        now = time.monotonic()
        if now - self._last_print < 0.5:
            return
        self._last_print = now
        hand = "ok" if snap.hand is not None else "LOST"
        clutch = "MOVE" if snap.clutch else ("FRZN" if snap.gate_pinching else "off ")
        if self.phase == RECORDING:
            episode = (f"REC {self.recorder.n_frames:5d}" if self.recording else "PRACTICE ") + \
                      f" {self.clock() - self._t_start:4.1f} s" + self.success.status()
        elif self.phase == WAITING:
            episode = "home: " + ("pinch+release" if self.hand_filter.ready else "glove rest...")
        elif self.phase == DONE:
            episode = "saved: cube down, h"
        else:
            episode = "finished: q"
        line = (f"[{episode}] hand:{hand:4} clutch:{clutch} {verdict_reason:<28.28}"
                f"{self._amps()}{self.hand_filter.status()}")
        print("\r" + line[:shutil.get_terminal_size((120, 24)).columns - 1], end="", flush=True)
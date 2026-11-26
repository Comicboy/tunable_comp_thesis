#!/usr/bin/env python3
"""Offline checks of hand_driver.MeasuredWujiHand2 and tactile_admittance (no hand, no glove).

    python scripts/real/test_tactile_admittance.py

Fakes replace the glove (frames made up here) and the hand's network stream; the supervisor's
contact filter, our fingertip map, the hand model and the admittance law are the real ones.
  1. MeasuredWujiHand2.state() returns the last *measurement* when no new frame has arrived
     (the supervisor's returns the last command), and the command only before any measurement;
  2. TactileFingertipAdmittance as the session drives it: rest learning, a press on the index
     fingertip opens only the index, the recorded keys and shapes, K absent from the status line,
     a glove dropout sets the force to 0 and the offset relaxes, tick intervals are logged;
  3. the per-tick history in extra(): (5, ...) windows ending at the recorded tick, nan-padded
     at the start, and consecutive windows joining up into the full 125 Hz record.
Prints PASS / FAIL per check; exit code 1 if any fails.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from quest_teleop2.tactile.contact import FINGERS, FingerRegions
from quest_teleop2.tactile.glove import TactileFrame
from tunable_comp_real.admittance import AdmittanceParams
from tunable_comp_real.hand_driver import MeasuredWujiHand2
from tunable_comp_real.hand_model import HandModel
from tunable_comp_real.tactile_admittance import TactileFingertipAdmittance

failures = 0


def check(ok: bool, text: str) -> None:
    global failures
    failures += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}")


class FakeGlove:
    """Serves frames we set; `freeze()` stops new frames (a dropout)."""

    def __init__(self) -> None:
        self.info = {"rate_hz": 117.0, "serial": "FAKE"}
        self.lost = False
        rng = np.random.default_rng(0)
        self.base = rng.uniform(0.05, 0.2, (24, 32)).astype(np.float32)
        self.press = np.zeros((24, 32), np.float32)
        self._seq, self._frozen, self._last = 0, False, None
        self._rng = rng

    def latest(self) -> TactileFrame:
        if self._frozen and self._last is not None:
            return self._last
        self._seq += 1
        p = self.base + self.press + self._rng.normal(0, 0.0005, (24, 32)).astype(np.float32)
        self._last = TactileFrame(p, self._seq, 0, time.monotonic())
        return self._last

    def freeze(self) -> None:
        self._frozen = True


class FakeStreamHand(MeasuredWujiHand2):
    """MeasuredWujiHand2 without a network: _read_positions() serves a scripted sequence."""

    def __init__(self, limits) -> None:
        super().__init__(limits, dry_run=True)
        self.frames: list[np.ndarray | None] = []

    def _read_positions(self):
        return self.frames.pop(0) if self.frames else None


def driver(model: HandModel) -> None:
    print("1. MeasuredWujiHand2.state()")
    hand = FakeStreamHand((model.lower, model.upper))
    hand.command(np.full(20, 0.3))                       # a command, no measurement yet
    check(np.allclose(hand.state(), 0.3), "before any measurement: the command (as the parent)")
    m1 = np.full((5, 4), 0.5)
    m2 = np.full((5, 4), 0.8)
    hand.frames = [m1, None, None, m2]
    hand.repeated_reads = 0
    reads = [hand.state() for _ in range(4)]
    u1, u2 = hand.to_normalised(m1), hand.to_normalised(m2)
    check(all(np.allclose(r, u1) for r in reads[:3]) and np.allclose(reads[3], u2),
          "no new frame -> the last measurement, never the command")
    check(hand.repeated_reads == 2, f"repeated reads counted ({hand.repeated_reads})")
    check(np.allclose(hand.measured_radians(), m2.reshape(-1)), "measured_radians() = last frame [rad]")


def tactile(model: HandModel, regions: Path) -> None:
    print("2. TactileFingertipAdmittance, driven like the session's hand_filter")
    tips = FingerRegions.load(regions)
    glove = FakeGlove()
    hand = FakeStreamHand((model.lower, model.upper))
    filt = TactileFingertipAdmittance(glove, hand, model, tips, AdmittanceParams(), regions_path=regions)
    filt.set_stiffness(500.0)
    # Reference: flexion joints at 0.6 rad; the hand "measures" exactly the reference.
    q_ref = np.zeros(20)
    for f in FINGERS:
        q_ref[model.finger_slice(f)] = [0.6, 0.0, 0.6, 0.6] if f != "thumb" else [0.3, -0.3, 0.6, 0.6]
    u_ref = model.to_normalised(q_ref)

    def tick(n: int) -> np.ndarray:
        u = u_ref
        for _ in range(n):
            hand.frames.append(q_ref.reshape(5, 4))
            u = filt.step(u_ref)
            time.sleep(0.0005)
        return u

    tick(3)
    check(not filt.ready, "not ready before the rest is learned")
    filt.learn_rest()
    tick(filt.rest_frames + 5)
    check(filt.ready, "ready after learning the rest")
    u = tick(50)
    check(np.allclose(u, u_ref, atol=1e-9), "no contact: command = reference")
    index = tips.masks[tips.names.index("index")]
    glove.press[index] = 0.15                               # press the index fingertip
    u = tick(250)                                           # 2 s
    F = filt.force
    check(F[1] > 0 and np.all(F[[0, 2, 3, 4]] == 0), f"force only on the index: {np.round(F, 2)} (glove units)")
    sl = model.finger_slice("index")
    q_cmd = model.to_radians(u)
    e = filt.law.offset[1]
    check(np.linalg.norm(e) > 0 and float(e @ filt.law.normal[1]) < 0, "index offset against its pad normal")
    others = [i for i in range(20) if not (sl.start <= i < sl.stop)]
    check(np.allclose(q_cmd[others], q_ref[others], atol=1e-9), "the other fingers keep their reference")
    obs, extra = filt.observation(), filt.extra()
    check(set(obs) == set(filt.OBS_KEYS) and set(extra) == set(filt.EXTRA_KEYS), "recorded keys as declared")
    check(obs["tactile"].shape == (24, 32) and obs["hand_q"].shape == (20,) and extra["tip_offset"].shape == (5, 3),
          "shapes: tactile (24, 32), hand_q (20,), tip_offset (5, 3)")
    status = filt.status()
    check("500" not in status and "K" not in status, f"status line without K: {status!r}")
    a = filt.attrs()
    check(a["force_unit"] == "glove" and np.allclose(a["stiffness"], 500.0), "attrs: unit and K recorded")
    glove.freeze()                                          # dropout: no new frames
    time.sleep(0.15)
    tick(250)
    check(not filt.ready and np.all(filt.force == 0), "dropout: not ready, force 0")
    check(np.linalg.norm(filt.law.offset[1]) < 1e-5, "dropout: the index offset relaxed to 0 (no reset)")
    stats = filt.tick_stats()
    check(stats["ticks"] > 0 and np.isfinite(stats["mean_interval"]),
          f"tick intervals logged: {stats['ticks']} ticks, mean {stats['mean_interval'] * 1000:.2f} ms")


def tick_history(model: HandModel, regions: Path) -> None:
    print("3. per-tick history in extra() (recorded every 5th tick, like the session)")
    tips = FingerRegions.load(regions)
    glove = FakeGlove()
    hand = FakeStreamHand((model.lower, model.upper))
    filt = TactileFingertipAdmittance(glove, hand, model, tips, AdmittanceParams(), regions_path=regions,
                                      ticks_per_record=5)
    filt.set_stiffness(300.0)
    rng = np.random.default_rng(1)
    q_ref = model.to_radians(np.full(20, 0.4))
    u_ref = model.to_normalised(q_ref)

    def one_tick() -> None:
        hand.frames.append((q_ref + rng.normal(0, 0.01, 20)).reshape(5, 4))   # measured q varies per tick
        filt.step(u_ref)
        time.sleep(0.0005)

    one_tick(), one_tick()
    x = filt.extra()
    check(all(x[k].shape == (5, *filt.TICK_SHAPES[k]) for k in filt.TICK_KEYS),
          "shapes (5, ...) per key: " + ", ".join(f"{k} {x[k].shape}" for k in filt.TICK_KEYS))
    check(np.all(np.isnan(x["hand_q_ticks"][:3])) and np.all(np.isfinite(x["hand_q_ticks"][3:])),
          "after 2 ticks: 3 nan rows at the front, then the 2 ticks")
    # Learn the rest, press the index, then record like the session: extra() on every 5th tick.
    filt.learn_rest()
    while not filt.ready:
        one_tick()
    glove.press[tips.masks[tips.names.index("index")]] = 0.15
    every_tick_cmd, every_tick_e, windows = [], [], []
    for t in range(5 * 40):
        one_tick()
        every_tick_cmd.append(filt.law.q_cmd.copy())
        every_tick_e.append(filt.law.offset.copy())
        if t % 5 == 4:
            windows.append(filt.extra())
    last = windows[-1]
    check(np.allclose(last["hand_q_cmd_ticks"][-1], last["hand_q_cmd"]) and
          np.allclose(last["tip_offset_ticks"][-1], last["tip_offset"]),
          "the window's last row = the recorded tick's values")
    joined_cmd = np.concatenate([w["hand_q_cmd_ticks"] for w in windows])
    joined_e = np.concatenate([w["tip_offset_ticks"] for w in windows])
    check(np.allclose(joined_cmd, np.array(every_tick_cmd), atol=1e-6) and
          np.allclose(joined_e, np.array(every_tick_e), atol=1e-7),
          f"{len(windows)} windows joined = all {len(every_tick_cmd)} ticks, no gap or overlap")
    check(np.any(joined_e[:, 1] != 0), "the index offset moves within the windows (force on)")
    dt = np.concatenate([w["tick_interval_ticks"] for w in windows])
    check(np.all(np.isfinite(dt)) and np.all(dt > 0), f"tick intervals per tick, mean {dt.mean() * 1000:.2f} ms")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regions", type=Path, default=Path("config/real/glove_tip_regions.yaml"))
    ap.add_argument("--urdf", type=Path, default=None)
    args = ap.parse_args()
    model = HandModel(args.urdf)
    driver(model)
    tactile(model, args.regions)
    tick_history(model, args.regions)
    print(f"\n{'all checks passed' if failures == 0 else f'{failures} check(s) FAILED'}")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
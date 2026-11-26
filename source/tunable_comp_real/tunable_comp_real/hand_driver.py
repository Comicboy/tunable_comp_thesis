"""Our subclass of the supervisor's Wuji Hand 2 driver.

Origin: quest_teleop2.robot.gripper.WujiHand2Gripper (third_party/quest_teleop_2, zip of
2026-11-25), used unchanged except for `state()`, overridden here.

Why: his `state()` drains the joint-state stream to the newest frame, and when no new frame has
arrived since the previous call it returns the last *command* instead of the last measurement.
His session reads the hand once per recording tick (every 40 ms), so for him this is rare; our
fingertip admittance reads the measured joint angles every 8 ms tick, the rate the hand streams
at, so the session's own read would often find no new frame and record the command as the
"measured" hand (against his own rule in session._observe). This class keeps the last real
measurement and returns it whenever no new frame has arrived; only before the first measurement
does it fall back to the command, as his does. Reported to the supervisor (2026-11-26); drop this
class once his driver does the same.

Everything else (connect / enable, the 0..1 contract and hyperextension floor, the 5 Hz command
low-pass, MIT frames, currents, teardown) is inherited unchanged.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from quest_teleop2.robot.gripper import WujiHand2Gripper

_log = logging.getLogger(__name__)


class MeasuredWujiHand2(WujiHand2Gripper):
    """WujiHand2Gripper whose state() never returns a command as a measurement."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._measured: np.ndarray | None = None   # (5, 4) rad, last measured frame
        self._measured_t: float | None = None      # time.monotonic() when it arrived
        #: Reads that found no new frame (answered with the previous measurement).
        self.repeated_reads = 0

    def connect(self) -> None:
        super().connect()
        # His connect() waited for a first measured pose and stored it, normalised, in _value.
        self._measured = self.to_radians(self._value)
        self._measured_t = time.monotonic()

    def _poll(self) -> None:
        """Take the newest frame from the stream, if a new one has arrived."""
        try:
            measured = self._read_positions()
        except Exception as exc:  # noqa: BLE001 -- a state read must never kill the control loop
            _log.warning("hand state read failed, keeping the last measurement: %s", exc)
            measured = None
        if measured is not None:
            self._measured = measured
            self._measured_t = time.monotonic()
        else:
            self.repeated_reads += 1

    def state(self) -> np.ndarray:
        """Measured joint positions, normalised (20,): the newest frame, or the last one if no new
        frame has arrived. Before any measurement, the command (as the parent class)."""
        self._poll()
        if self._measured is None:
            return self._value.copy()
        return self.to_normalised(self._measured)

    def measured_radians(self) -> np.ndarray | None:
        """The last measured joint angles (20,) [rad], unclipped; None before the first one.
        Does not read the stream: call state() once per tick first."""
        return None if self._measured is None else self._measured.reshape(-1).copy()

    @property
    def measurement_age(self) -> float:
        """Seconds since the last new measured frame arrived (inf before the first)."""
        return np.inf if self._measured_t is None else time.monotonic() - self._measured_t
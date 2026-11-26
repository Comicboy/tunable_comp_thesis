"""Client of scripts/glove/glove_server.py: the latest tactile frame, read on a background thread.

The glove needs wujihandpy 1.7 or later, which lives in its own environment (wuji-glove), so
it is read in a separate process and streamed here over a local socket.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "scripts" / "glove" / "glove_server.py"
SOCKET = "/tmp/quest_teleop2_glove.sock"
GLOVE_PYTHON = Path(os.environ.get("QUEST_TELEOP2_GLOVE_PYTHON",
                                   Path.home() / "miniforge3" / "envs" / "wuji-glove" / "bin" / "python"))
HEADER = struct.Struct("<HI")
ROWS, COLS = 24, 32
FRAME_BYTES = HEADER.size + ROWS * COLS * 4


@dataclass
class TactileFrame:
    pressure: np.ndarray    #: (24, 32) float32, 0..1, NaN outside the sensors
    sequence: int
    device_ms: int
    received: float        #: time.monotonic() when it arrived


class GloveClient:
    """Connects to the glove server, starting one if none is running."""

    def __init__(self, path: str = SOCKET, python: Path = GLOVE_PYTHON) -> None:
        self.path, self.python = path, Path(python)
        self.info: dict = {}
        self.lost = False
        self.frames = 0
        self._sock: socket.socket | None = None
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: TactileFrame | None = None
        self._times: deque[float] = deque(maxlen=256)
        self._stop = threading.Event()

    def start(self, timeout_s: float = 15.0) -> GloveClient:
        if not self._connect():
            if not self.python.exists():
                raise FileNotFoundError(f"No python at {self.python} to read the glove; set "
                                        "QUEST_TELEOP2_GLOVE_PYTHON or pass --glove-python.")
            self._proc = subprocess.Popen([str(self.python), "-I", str(SERVER), "--socket", self.path],
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            deadline = time.monotonic() + timeout_s
            while not self._connect():
                if self._proc.poll() is not None:
                    out = self._proc.stdout.read() if self._proc.stdout else ""
                    raise RuntimeError(f"The glove server exited ({self._proc.returncode}):\n{out[-1500:]}")
                if time.monotonic() > deadline:
                    self.close()
                    raise TimeoutError(f"The glove server did not start in {timeout_s:.0f} s.")
                time.sleep(0.2)
        self.info = json.loads(self._readline())
        self._thread = threading.Thread(target=self._read_loop, name="glove", daemon=True)
        self._thread.start()
        return self

    def _connect(self) -> bool:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.connect(self.path)
        except OSError:
            s.close()
            return False
        s.settimeout(5.0)
        self._sock = s
        return True

    def _readline(self) -> str:
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = self._sock.recv(1)
            if not chunk:
                raise ConnectionError("the glove server closed the connection")
            buf += chunk
        return buf.decode()

    def _read_loop(self) -> None:
        buf = bytearray(FRAME_BYTES)
        view = memoryview(buf)
        while not self._stop.is_set():
            got = 0
            try:
                while got < FRAME_BYTES:
                    n = self._sock.recv_into(view[got:])
                    if n == 0:
                        raise ConnectionError
                    got += n
            except TimeoutError:
                continue
            except (OSError, ConnectionError):
                self.lost = True
                return
            seq, ms = HEADER.unpack_from(buf, 0)
            p = np.frombuffer(bytes(buf[HEADER.size:]), dtype=np.float32).reshape(ROWS, COLS).copy()
            now = time.monotonic()
            with self._lock:
                self._latest = TactileFrame(p, seq, ms, now)
                self._times.append(now)
                self.frames += 1

    def latest(self) -> TactileFrame | None:
        with self._lock:
            return self._latest

    @property
    def rate_hz(self) -> float:
        with self._lock:
            now = time.monotonic()
            return float(sum(1 for t in self._times if now - t < 1.0))

    def close(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None


def collect(client: GloveClient, seconds: float, each=None) -> list[np.ndarray]:
    """Every new frame for `seconds`, passing each to `each(pressure)` if given."""
    frames, seq = [], None
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        f = client.latest()
        if f is not None and f.sequence != seq:
            seq = f.sequence
            frames.append(f.pressure)
            if each is not None:
                each(f.pressure)
        if client.lost:
            raise ConnectionError("the glove server stopped")
        time.sleep(0.002)
    return frames

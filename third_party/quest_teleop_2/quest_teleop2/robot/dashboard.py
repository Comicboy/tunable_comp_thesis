"""Dashboard server client (TCP 29999).

Used only for preflight checks and error reporting. The dashboard is the only
interface that can tell us whether the robot is in Remote Control mode, which is
the single most common reason a UR e-Series refuses URScript from a socket --
and it fails silently, so checking for it up front saves a lot of confusion.
"""

from __future__ import annotations

import socket

from ..config import RobotCfg


class DashboardClient:
    """Line-oriented request/response client for the UR dashboard server."""

    def __init__(self, cfg: RobotCfg, timeout: float = 3.0) -> None:
        self._host = cfg.connection.host
        self._port = cfg.connection.dashboard_port
        self._timeout = timeout
        self._sock: socket.socket | None = None
        self._file: object | None = None

    def connect(self) -> str:
        self._sock = socket.create_connection((self._host, self._port), timeout=self._timeout)
        self._file = self._sock.makefile("rwb")
        return self._readline()  # greeting: "Connected: Universal Robots Dashboard Server"

    def disconnect(self) -> None:
        for closeable in (self._file, self._sock):
            try:
                if closeable is not None:
                    closeable.close()
            except Exception:
                pass
        self._file = None
        self._sock = None

    def __enter__(self) -> DashboardClient:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.disconnect()

    def _readline(self) -> str:
        assert self._file is not None
        return self._file.readline().decode("utf-8", errors="replace").strip()

    def send(self, command: str) -> str:
        if self._file is None:
            raise RuntimeError("Dashboard not connected.")
        self._file.write((command + "\n").encode("utf-8"))
        self._file.flush()
        return self._readline()

    # -- queries ---------------------------------------------------------- #

    def robot_mode(self) -> str:
        return self.send("robotmode").removeprefix("Robotmode:").strip()

    def safety_status(self) -> str:
        return self.send("safetystatus").removeprefix("Safetystatus:").strip()

    def program_state(self) -> str:
        return self.send("programState").strip()

    def is_in_remote_control(self) -> bool:
        """False means URScript sent to ports 30001-30003 will be silently ignored."""
        return self.send("is in remote control").strip().lower() == "true"

    def polyscope_version(self) -> str:
        return self.send("PolyscopeVersion").strip()

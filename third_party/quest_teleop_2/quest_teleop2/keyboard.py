"""Non-blocking keyboard input, with no dependencies.

Uses raw-mode termios rather than a library like pynput: pynput needs macOS
Accessibility permission, which is an install-time prompt and a support burden
for something this small. Raw stdin needs nothing.

Degrades to a no-op when stdin is not a TTY (piped input, CI), so scripts that
poll it still run under automation.
"""

from __future__ import annotations

import contextlib
import logging
import os
import select
import sys
import termios
import tty

_log = logging.getLogger(__name__)


class KeyboardReader:
    """Reads single keypresses without blocking or waiting for Enter."""

    def __init__(self) -> None:
        self._fd: int | None = None
        self._saved: list | None = None
        self.enabled = False

    def __enter__(self) -> KeyboardReader:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def start(self) -> None:
        if not sys.stdin.isatty():
            _log.warning("stdin is not a TTY; keyboard input disabled.")
            return
        try:
            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            # cbreak, not raw: keeps Ctrl-C working as SIGINT, which is the
            # operator's way out of the control loop.
            tty.setcbreak(self._fd)
            self.enabled = True
        except (termios.error, ValueError) as exc:
            _log.warning("Could not set up keyboard input: %s", exc)
            self._fd = None

    def stop(self) -> None:
        if self._fd is not None and self._saved is not None:
            with contextlib.suppress(termios.error, ValueError):
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
        self._fd = None
        self._saved = None
        self.enabled = False

    def poll(self) -> list[str]:
        """Every key pressed since the last call, lower-cased. Never blocks."""
        if not self.enabled or self._fd is None:
            return []
        keys: list[str] = []
        while True:
            ready, _, _ = select.select([self._fd], [], [], 0)
            if not ready:
                break
            data = os.read(self._fd, 1)
            if not data:
                break
            try:
                keys.append(data.decode("utf-8").lower())
            except UnicodeDecodeError:
                continue
        return keys

    def drain(self) -> None:
        """Discard anything typed while the operator was not being asked."""
        self.poll()

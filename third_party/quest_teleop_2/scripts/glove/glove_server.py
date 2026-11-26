#!/usr/bin/env python3
"""Stream the tactile glove's frames to local programs, from the env that can read the glove.

    ~/miniforge3/envs/wuji-glove/bin/python -I scripts/glove/glove_server.py

Programs in this repository start it themselves when they need the glove, and several can
read it at once. Each client gets one JSON line describing the glove, then one fixed-size
frame per sample: little-endian sequence (uint16) and device time (uint32, ms), then the
24 x 32 pressure grid as float32, NaN outside the sensors.
"""

import argparse
import json
import os
import socket
import struct
import threading

import numpy as np

SOCKET = "/tmp/quest_teleop2_glove.sock"
HEADER = struct.Struct("<HI")
ROWS, COLS = 24, 32
VERSION = 1


class Broadcaster:
    """Sends every frame to every connected client; a client that cannot keep up is dropped."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.clients: list[socket.socket] = []

    def add(self, conn: socket.socket, hello: dict) -> None:
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        conn.sendall((json.dumps(hello) + "\n").encode())
        conn.setblocking(False)
        with self.lock:
            self.clients.append(conn)

    def send(self, sequence: int, device_ms: int, pressure: np.ndarray) -> None:
        msg = HEADER.pack(sequence & 0xFFFF, device_ms & 0xFFFFFFFF) + \
            np.ascontiguousarray(pressure, dtype=np.float32).tobytes()
        with self.lock:
            for c in list(self.clients):
                try:
                    if c.send(msg) == len(msg):
                        continue
                except OSError:
                    pass
                self.clients.remove(c)
                c.close()

    def close(self) -> None:
        with self.lock:
            for c in self.clients:
                c.close()
            self.clients.clear()


def serve(glove, path: str = SOCKET, ready=None, stop: threading.Event | None = None) -> None:
    """Stream `glove` (a wujihandpy.TactileGlove, or anything with its methods) on `path`."""
    info = glove.get_device_info()
    hello = {"version": VERSION, "serial": str(info.serial), "hand": str(glove.get_handedness()),
             "rate_hz": float(glove.get_sample_rate_hz()), "rows": ROWS, "cols": COLS}
    if not glove.get_streaming_enabled():
        glove.set_streaming(True)
    out = Broadcaster()
    lost = threading.Event()
    glove.set_disconnect_callback(lost.set)
    glove.start_streaming(lambda f: out.send(int(f.sequence), int(f.timestamp_ms), f.pressure))
    if os.path.exists(path):
        os.unlink(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(8)
    srv.settimeout(0.2)
    print(f"glove {hello['serial']} ({hello['hand']}, {hello['rate_hz']:.0f} Hz) on {path}", flush=True)
    if ready is not None:
        ready.set()
    try:
        while not lost.is_set() and not (stop is not None and stop.is_set()):
            try:
                conn, _ = srv.accept()
            except TimeoutError:
                continue
            out.add(conn, hello)
        if lost.is_set():
            print("the glove disconnected", flush=True)
    finally:
        glove.stop_streaming()
        out.close()
        srv.close()
        if os.path.exists(path):
            os.unlink(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--socket", default=SOCKET)
    ap.add_argument("--serial", default=None, help="glove USB serial, if more than one")
    args = ap.parse_args()
    import wujihandpy

    with wujihandpy.TactileGlove(args.serial) as glove:
        try:
            serve(glove, args.socket)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

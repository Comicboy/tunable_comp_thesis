#!/usr/bin/env python
"""SAM 3 for the robot scripts: segments the frames they send. Runs in the sam3 env.

    ~/miniforge3/envs/sam3/bin/python scripts/camera/sam_server.py

Robot scripts run with --sam start it when it is not running and stop it when
they exit. Started by hand, it stays up and keeps the model loaded between runs.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quest_teleop2.vision.sam import (  # noqa: E402
    SERVER_FLOOR,
    SOCKET_PATH,
    pack_masks,
    recv_msg,
    send_msg,
)
from scripts.camera.sam_live import load_model, segment  # noqa: E402


def serve(conn: socket.socket, processor) -> None:
    while True:
        try:
            header, payload = recv_msg(conn)
        except ConnectionError:
            return
        try:
            frame = np.frombuffer(payload, dtype=np.uint8).reshape(header["shape"])
            t = time.perf_counter()
            found = segment(processor, frame, [str(p) for p in header["prompts"]], bf16=True)
            h, w = frame.shape[:2]
            masks = np.stack([d["mask"] for d in found]) if found else np.zeros((0, h, w), bool)
            send_msg(conn, {"ok": True, "shape": [h, w], "ms": (time.perf_counter() - t) * 1e3,
                            "detections": [{"prompt": d["prompt"], "score": d["score"],
                                            "box": [float(v) for v in d["box"]]}
                                           for d in found]},
                     pack_masks(masks))
        except Exception as exc:  # noqa: BLE001 -- answer, so the robot refuses rather than waits
            send_msg(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--socket", type=Path, default=SOCKET_PATH)
    args = p.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    t = time.perf_counter()
    processor = load_model(SERVER_FLOOR)
    args.socket.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(args.socket))
        os.chmod(args.socket, 0o600)
        server.listen(1)
        print(f"SAM 3 ready on {args.socket} after {time.perf_counter() - t:.0f} s", flush=True)
        while True:
            conn, _ = server.accept()
            with conn:
                serve(conn, processor)
    except KeyboardInterrupt:
        return 0
    finally:
        server.close()
        args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

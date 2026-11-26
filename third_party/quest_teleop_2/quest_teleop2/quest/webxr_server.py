"""HTTP + WebSocket server for the headset client.

Runs its own asyncio loop on a daemon thread, exactly like the Isaac
quest3_device.py did. The control loop never touches asyncio: it reads
`QuestInput` (lock-protected) and pushes HUD updates through thread-safe
helpers that hop back onto the server loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

from ..config import TeleopCfg
from .quest_input import INDEX_TIP, JOINT_NAMES, N_JOINTS, THUMB_TIP, QuestInput

_log = logging.getLogger(__name__)

_CLIENT_HTML = Path(__file__).parent / "client" / "index.html"


class WebXRServer:
    """Serves the WebXR page and bridges its WebSocket to a `QuestInput`."""

    def __init__(self, cfg: TeleopCfg, quest: QuestInput) -> None:
        self._cfg = cfg
        self._quest = quest
        self._html = self._render_html()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._clients: set[web.WebSocketResponse] = set()
        self._ready = threading.Event()
        self.session_active = False
        #: Latest input-source report from the headset. Quest switches between
        #: hand-tracking and controller modes, so this is what tells you whether
        #: the controller is present in the session at all.
        self.input_sources: list[dict] = []

    # -- page ------------------------------------------------------------- #

    def _render_html(self) -> str:
        tracked = self._cfg.controller.tracked_hand
        client_cfg = {
            "send_interval_ms": max(1, 1000 // self._cfg.server.send_hz),
            "tracked_hand": tracked,
            "other_hand": "left" if tracked == "right" else "right",
            "dual_hand": self._cfg.input.mode == "hands",
            "controller_hand": self._cfg.controller.handedness,
            "joint_names": list(JOINT_NAMES),
            "thumb_tip": THUMB_TIP,
            "index_tip": INDEX_TIP,
            "pinch_close_m": self._cfg.gripper.pinch_close_m,
            "squeeze_index": self._cfg.controller.buttons.squeeze,
            "press_threshold": self._cfg.controller.press_threshold,
        }
        html = _CLIENT_HTML.read_text()
        if "__CONFIG__" not in html:
            raise RuntimeError(f"{_CLIENT_HTML} is missing the __CONFIG__ placeholder.")
        return html.replace("__CONFIG__", json.dumps(client_cfg))

    # -- lifecycle -------------------------------------------------------- #

    def start(self, timeout: float = 5.0) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="quest-webxr")
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("WebXR server failed to start.")

    def stop(self, timeout: float = 2.0) -> None:
        """Close client connections before stopping the loop.

        Killing the loop with sockets still open leaves aiohttp printing
        'Task was destroyed but it is pending' over the exit summary, which
        buries whatever actually went wrong.
        """
        loop = self._loop
        if loop is None:
            return

        async def _shutdown() -> None:
            for ws in list(self._clients):
                with contextlib.suppress(Exception):
                    await ws.close()
            self._clients.clear()

        try:
            future = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
            future.result(timeout=timeout)
        except Exception:
            pass
        finally:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=timeout)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._serve())
            loop.run_forever()
        finally:
            loop.close()

    async def _serve(self) -> None:
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/ws", self._ws_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self._cfg.server.port)
        await site.start()
        _log.info("WebXR server listening on 0.0.0.0:%d", self._cfg.server.port)
        self._ready.set()

    async def _index(self, request: web.Request) -> web.Response:
        return web.Response(text=self._html, content_type="text/html")

    # -- websocket -------------------------------------------------------- #

    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=5.0)
        await ws.prepare(request)
        self._clients.add(ws)
        _log.info("Headset connected (%d client(s))", len(self._clients))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        self._handle(json.loads(msg.data))
                    except Exception as exc:
                        _log.warning("Bad message from headset: %s", exc)
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            self._clients.discard(ws)
            self.session_active = False
            self._quest.clear_hand()
            _log.info("Headset disconnected")
        return ws

    def _handle(self, data: dict) -> None:
        if "debug" in data:
            _log.info("[quest] %s", data["debug"])
            return
        if "fps" in data:
            _log.debug("[quest] %s fps", data["fps"])
            return
        if "session" in data:
            self.session_active = data["session"] == "started"
            _log.info("[quest] XR session %s", data["session"])
            return
        if "sources" in data:
            self.input_sources = data["sources"]
            return

        dual = self._cfg.input.mode == "hands"
        for key, handedness in (("hand", self._cfg.controller.tracked_hand), ("left", "left")):
            if key == "left" and not dual:
                continue
            payload = data.get(key)
            if payload is None:
                if dual:
                    self._quest.clear_hand(handedness)
                elif key == "hand":
                    self._quest.clear_hand()
                continue
            joints = payload.get("joints")
            quat = payload.get("wrist_quat")
            if joints is None or quat is None or len(joints) != N_JOINTS:
                continue
            joints_arr = np.asarray(joints, dtype=np.float64)
            quat_arr = np.asarray(quat, dtype=np.float64)
            # Optional: older clients send positions only. Dropped rather than
            # rejected if the length is wrong, so a client mismatch degrades to
            # the position-only path instead of killing the frame.
            quats = payload.get("quats")
            quats_arr = (
                np.asarray(quats, dtype=np.float64)
                if quats is not None and len(quats) == N_JOINTS
                else None
            )
            if dual:
                self._quest.update_hand(handedness, joints_arr, quat_arr, quats_arr)
            else:
                self._quest.update_hand(joints_arr, quat_arr, quats_arr)

        ctrl = data.get("ctrl")
        if ctrl is not None:
            buttons = ctrl.get("buttons") or []
            pressed = np.array([bool(b[0]) for b in buttons], dtype=bool)
            values = np.array([float(b[1]) for b in buttons], dtype=np.float64)
            self._quest.update_controller(
                position=np.asarray(ctrl["pos"], dtype=np.float64),
                quat=np.asarray(ctrl["quat"], dtype=np.float64),
                pressed=pressed,
                values=values,
                axes=np.asarray(ctrl.get("axes") or [], dtype=np.float64),
            )

    # -- outbound (thread-safe) ------------------------------------------- #

    def _broadcast(self, payload: dict) -> None:
        """Schedule a send from any thread. Silently no-ops if nothing is connected."""
        if self._loop is None or not self._clients:
            return
        text = json.dumps(payload)

        async def _send() -> None:
            for ws in list(self._clients):
                try:
                    await ws.send_str(text)
                except Exception:
                    self._clients.discard(ws)

        try:
            asyncio.run_coroutine_threadsafe(_send(), self._loop)
        except RuntimeError:
            pass  # loop shutting down

    def update_hud(self, **fields: object) -> None:
        """Push HUD field updates to the headset, e.g. update_hud(task='s2 -> g0')."""
        self._broadcast({"hud": fields})

    def banner(self, text: str, kind: str = "idle") -> None:
        """Flash a large message in the headset. kind: 'ok' | 'bad' | 'idle'."""
        self._broadcast({"banner": {"text": text, "kind": kind}})

    def haptic(self, intensity: float = 0.6) -> None:
        self._broadcast({"haptic": float(np.clip(intensity, 0.0, 1.0))})

    @property
    def n_clients(self) -> int:
        return len(self._clients)

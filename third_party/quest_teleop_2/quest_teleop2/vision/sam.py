"""SAM 3 as the object locator for the scripted reach.

SAM 3 needs its own environment, so it runs as a local server
(scripts/camera/sam_server.py). This is the robot side: the socket protocol, the
client, the object table, and the rules that turn one frame's masks into a reach
target or a refusal.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import struct
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
OBJECTS_PATH = ROOT / "config" / "objects.yaml"
SERVER_SCRIPT = ROOT / "scripts" / "camera" / "sam_server.py"
LOG_DIR = ROOT / "recordings" / "sam"
SOCKET_PATH = Path(f"/tmp/quest_teleop_sam3_{os.getuid()}.sock")
SAM_PYTHON = Path(os.environ.get("QUEST_TELEOP_SAM_PYTHON",
                                 Path.home() / "miniforge3" / "envs" / "sam3" / "bin" / "python"))

#: The server returns every mask scoring at least this; the rules judge the rest.
SERVER_FLOOR = 0.2
#: Default lowest score a reach target may have.
MIN_SCORE = 0.6
#: A second match makes the target ambiguous when it scores at least RIVAL_SCORE
#: and within RIVAL_MARGIN of the best: a near-tie, not a clearly weaker object.
RIVAL_SCORE = 0.35
RIVAL_MARGIN = 0.25
#: Masks overlapping more than this are one object, not two.
SAME_OBJECT_IOU = 0.5
#: Share of a mask its largest connected piece must hold.
MIN_MAIN_FRAC = 0.85
DEFAULT_AREA_CM2 = (4.0, 900.0)
#: Tallest object an approach is planned for, metres.
MAX_HEIGHT_M = 0.3
ANSWER_TIMEOUT_S = 10.0
START_TIMEOUT_S = 180.0

#: Colours that read the same as RGB or BGR.
GREEN, WHITE, MAGENTA = (0, 255, 0), (255, 255, 255), (255, 0, 255)


# -- protocol ---------------------------------------------------------------- #

def send_msg(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    head = json.dumps(header).encode()
    sock.sendall(struct.pack("!II", len(head), len(payload)) + head + payload)


def recv_msg(sock: socket.socket) -> tuple[dict, bytes]:
    n_head, n_payload = struct.unpack("!II", _recv_exact(sock, 8))
    return json.loads(_recv_exact(sock, n_head)), _recv_exact(sock, n_payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("the SAM 3 server closed the connection")
        buf += chunk
    return bytes(buf)


def pack_masks(masks: np.ndarray) -> bytes:
    return np.packbits(np.asarray(masks, dtype=bool).reshape(-1)).tobytes()


def unpack_masks(data: bytes, n: int, h: int, w: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), count=n * h * w)
    return bits.astype(bool).reshape(n, h, w)


# -- objects ----------------------------------------------------------------- #

@dataclass(frozen=True)
class ObjectSpec:
    name: str
    prompt: str
    height_m: float
    area_cm2: tuple[float, float] = DEFAULT_AREA_CM2
    min_score: float = MIN_SCORE


def load_objects(path: Path | str = OBJECTS_PATH) -> tuple[dict[str, ObjectSpec], ObjectSpec]:
    """Every object, and the one the table calibration was made with."""
    import yaml  # noqa: PLC0415

    d = yaml.safe_load(Path(path).read_text()) or {}
    objects = {}
    for name, v in (d.get("objects") or {}).items():
        spec = ObjectSpec(name=str(name), prompt=str(v.get("prompt", "")).strip(),
                          height_m=float(v.get("height_m", 0.0)),
                          area_cm2=tuple(float(a) for a in v.get("area_cm2", DEFAULT_AREA_CM2)),
                          min_score=float(v.get("min_score", MIN_SCORE)))
        lo, hi = spec.area_cm2
        if not spec.prompt or spec.height_m <= 0 or not 0 < lo < hi or not 0 < spec.min_score <= 1:
            raise ValueError(f"{path}: {name!r} needs a prompt, height_m > 0, area_cm2 "
                             "[lo, hi] with 0 < lo < hi, and 0 < min_score <= 1")
        objects[spec.name] = spec
    ref = d.get("calibrated_with")
    if ref not in objects:
        raise ValueError(f"{path}: calibrated_with must name one of the objects, not {ref!r}")
    return objects, objects[ref]


def object_spec(name: str, height_m: float | None = None, need_height: bool = True,
                path: Path | str = OBJECTS_PATH) -> tuple[ObjectSpec, ObjectSpec]:
    """`name` from config/objects.yaml (by name or prompt), or any short phrase as its own
    prompt; and the object the calibration was made with.

    A phrase has no recorded height, and the approach clearance depends on it, so it
    needs `height_m` unless nothing will move (need_height=False).
    """
    objects, ref = load_objects(path)
    name = " ".join(str(name).split())
    spec = objects.get(name) or next((o for o in objects.values() if o.prompt == name), None)
    if spec is None:
        if not name:
            raise ValueError("The object prompt is empty.")
        if height_m is None and need_height:
            raise ValueError(f"'{name}' is not in config/objects.yaml, so its height is "
                             "unknown: pass --object-height in metres, or add it there.")
        spec = ObjectSpec(name=name, prompt=name,
                          height_m=ref.height_m if height_m is None else float(height_m))
    elif height_m is not None:
        spec = replace(spec, height_m=float(height_m))
    if not 0 < spec.height_m <= MAX_HEIGHT_M:
        raise ValueError(f"Object height {spec.height_m} m is outside (0, {MAX_HEIGHT_M}].")
    return spec, ref


# -- rules ------------------------------------------------------------------- #

@dataclass
class Candidate:
    prompt: str
    score: float
    mask: np.ndarray
    px: np.ndarray       #: mask centroid, pixels
    xy: np.ndarray       #: table position under the centroid, metres
    area_cm2: float
    main_frac: float     #: share of the mask in its largest piece
    on_border: bool
    in_region: bool

    def problem(self, spec: ObjectSpec) -> str | None:
        """Why this cannot be the reach target, or None."""
        if self.score < spec.min_score:
            return f"scores {self.score:.2f}, below {spec.min_score:.2f}"
        if not self.in_region:
            return (f"is at ({self.xy[0]:+.3f}, {self.xy[1]:+.3f}) m, outside the "
                    "calibrated region")
        if self.on_border:
            return "is cut off by the image border"
        if self.main_frac < MIN_MAIN_FRAC:
            return f"is in pieces ({self.main_frac:.0%} in the largest)"
        lo, hi = spec.area_cm2
        if not lo <= self.area_cm2 <= hi:
            return f"covers {self.area_cm2:.0f} cm2, outside {lo:.0f}-{hi:.0f}"
        return None


def candidates(detections: list[dict], table) -> list[Candidate]:
    """Measure every mask: centroid, table position, metric area, pieces, border.

    With no table (None, while one is being calibrated) the position and area are unknown.
    """
    import cv2  # noqa: PLC0415

    out = []
    for d in detections:
        mask = np.asarray(d["mask"], dtype=bool)
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            continue
        h, w = mask.shape
        px = np.array([xs.mean(), ys.mean()])
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        main = float(stats[1:, cv2.CC_STAT_AREA].max()) / xs.size if n > 1 else 1.0
        xy, area = np.full(2, np.nan), float("nan")
        if table is not None:
            try:
                xy = table.to_table(px)
                area = xs.size * _pixel_area_m2(table, px) * 1e4
            except ValueError:
                pass
        b = None if table is None else table.bounds
        inside = bool(np.all(np.isfinite(xy))) and (
            b is None or bool(np.all((xy >= b[:, 0]) & (xy <= b[:, 1]))))
        out.append(Candidate(
            prompt=str(d["prompt"]), score=float(d["score"]), mask=mask, px=px, xy=xy,
            area_cm2=float(area), main_frac=main,
            on_border=bool(xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1
                           or ys.max() == h - 1),
            in_region=inside))
    return out


def _pixel_area_m2(table, px: np.ndarray) -> float:
    """Table area under one pixel at `px`: the homography's local Jacobian determinant."""
    u, v = float(px[0]), float(px[1])
    a = table.to_table((u + 0.5, v)) - table.to_table((u - 0.5, v))
    b = table.to_table((u, v + 0.5)) - table.to_table((u, v - 0.5))
    return abs(float(a[0] * b[1] - a[1] * b[0]))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


@dataclass
class Decision:
    spec: ObjectSpec
    found: list[Candidate]
    target: Candidate | None
    reason: str
    ms: float = 0.0


def decide(found: list[Candidate], spec: ObjectSpec, pick: str = "one") -> Decision:
    """One frame's candidates -> the reach target, or why there is none.

    pick="one" refuses when another object in the calibrated region scores
    nearly as well; pick="best" takes the highest-scoring acceptable one.
    """
    named = sorted((c for c in found if c.prompt == spec.prompt), key=lambda c: -c.score)
    good = [c for c in named if c.problem(spec) is None]
    if not good:
        why = f"the best match {named[0].problem(spec)}" if named else "nothing matches"
        return Decision(spec, found, None, f"'{spec.prompt}': {why}")
    best = good[0]
    floor = max(RIVAL_SCORE, best.score - RIVAL_MARGIN)
    rivals = [c for c in named if c is not best and c.score >= floor and c.in_region
              and _iou(c.mask, best.mask) < SAME_OBJECT_IOU]
    if rivals and pick != "best":
        scores = ", ".join(f"{c.score:.2f}" for c in (best, *rivals))
        return Decision(spec, found, None,
                        f"'{spec.prompt}': {len(rivals) + 1} near-equal matches ({scores}); "
                        "remove the others or pass --sam-pick best")
    return Decision(spec, found, best,
                    f"'{spec.prompt}' {best.score:.2f} at ({best.xy[0]:+.3f}, "
                    f"{best.xy[1]:+.3f}) m, {best.area_cm2:.0f} cm2")


def decide_pixel(detections: list[dict], spec: ObjectSpec) -> Decision:
    """The object's mask before any table maps it, for calibrating one: its centroid is the
    pixel the reach will later place. Only the rules that need no map apply, and a near-equal
    second match refuses."""
    found = candidates(detections, None)
    named = sorted((c for c in found if c.prompt == spec.prompt), key=lambda c: -c.score)

    def problem(c: Candidate) -> str | None:
        if c.score < spec.min_score:
            return f"scores {c.score:.2f}, below {spec.min_score:.2f}"
        if c.on_border:
            return "is cut off by the image border"
        if c.main_frac < MIN_MAIN_FRAC:
            return f"is in pieces ({c.main_frac:.0%} in the largest)"
        return None

    good = [c for c in named if problem(c) is None]
    if not good:
        why = f"the best match {problem(named[0])}" if named else "nothing matches"
        return Decision(spec, found, None, f"'{spec.prompt}': {why}")
    best = good[0]
    floor = max(RIVAL_SCORE, best.score - RIVAL_MARGIN)
    rivals = [c for c in named if c is not best and c.score >= floor
              and _iou(c.mask, best.mask) < SAME_OBJECT_IOU]
    if rivals:
        scores = ", ".join(f"{c.score:.2f}" for c in (best, *rivals))
        return Decision(spec, found, None, f"'{spec.prompt}': {len(rivals) + 1} near-equal "
                        f"matches ({scores}); leave only the object in view")
    return Decision(spec, found, best, f"'{spec.prompt}' {best.score:.2f} at pixel "
                    f"({best.px[0]:.1f}, {best.px[1]:.1f})")


def draw(img: np.ndarray, decision: Decision | None) -> np.ndarray:
    """`decision` drawn over a copy of `img`: every mask outlined, the target crossed."""
    import cv2  # noqa: PLC0415

    out = np.ascontiguousarray(img).copy()
    if decision is None:
        return out
    for c in decision.found:
        if c.mask.shape != out.shape[:2]:
            continue
        contours, _ = cv2.findContours(c.mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, GREEN if c is decision.target else WHITE, 2)
    if decision.target is not None:
        u, v = (int(round(x)) for x in decision.target.px)
        cv2.drawMarker(out, (u, v), GREEN, cv2.MARKER_CROSS, 18, 2)
    cv2.putText(out, decision.reason[:95], (8, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, GREEN if decision.target is not None else MAGENTA, 2)
    return out


# -- client and locator ------------------------------------------------------ #

class SamClient:
    """Talks to the SAM 3 server, starting one when none is listening."""

    def __init__(self, path: Path = SOCKET_PATH, python: Path = SAM_PYTHON) -> None:
        self.path, self.python = Path(path), Path(python)
        self._sock: socket.socket | None = None
        self._proc: subprocess.Popen | None = None

    def start(self, timeout_s: float = START_TIMEOUT_S) -> SamClient:
        if self._connect():
            return self
        if not self.python.exists():
            raise FileNotFoundError(f"No SAM 3 python at {self.python}; set "
                                    "QUEST_TELEOP_SAM_PYTHON to the sam3 env's python.")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / "server.log"
        with open(log_path, "ab") as log:
            self._proc = subprocess.Popen(
                [str(self.python), str(SERVER_SCRIPT), "--socket", str(self.path)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                code, self._proc = self._proc.returncode, None
                raise RuntimeError(f"The SAM 3 server exited with code {code}; see {log_path}")
            if self._connect():
                return self
            time.sleep(0.5)
        self.close()
        raise TimeoutError(f"The SAM 3 server did not start in {timeout_s:.0f} s; "
                           f"see {log_path}")

    def _connect(self) -> bool:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.connect(str(self.path))
        except OSError:
            s.close()
            return False
        s.settimeout(ANSWER_TIMEOUT_S)
        self._sock = s
        return True

    def segment(self, rgb: np.ndarray, prompts: list[str]) -> list[dict]:
        """Every mask for every prompt in one RGB frame: prompt, score, box, mask."""
        if self._sock is None and not self._connect():
            raise ConnectionError("no SAM 3 server is listening")
        frame = np.ascontiguousarray(rgb, dtype=np.uint8)
        try:
            send_msg(self._sock, {"prompts": list(prompts), "shape": list(frame.shape)},
                     frame.tobytes())
            header, payload = recv_msg(self._sock)
        except OSError:
            self._drop()
            raise
        if not header.get("ok"):
            raise RuntimeError(header.get("error", "the SAM 3 server failed"))
        h, w = header["shape"]
        dets = header["detections"]
        masks = unpack_masks(payload, len(dets), h, w)
        return [{"prompt": d["prompt"], "score": float(d["score"]),
                 "box": np.asarray(d["box"], dtype=np.float64), "mask": m}
                for d, m in zip(dets, masks)]

    def _drop(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def close(self) -> None:
        """Disconnect, and stop the server if this client started it."""
        self._drop()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None


class SamLocator:
    """What `table.locate_cube` asks instead of the background detector.

    Servoing is paused while SAM 3 looks: a segmentation takes about 0.2 s and
    the RTDE watchdog trips after 0.1 s without a command. Every decision is
    saved as an image, and a refusal moves nothing.
    """

    def __init__(self, client, spec: ObjectSpec, arm=None, pick: str = "one",
                 log_dir: Path | None = LOG_DIR) -> None:
        self.client, self.spec, self.arm, self.pick, self.log_dir = (
            client, spec, arm, pick, log_dir)
        self.name = f"sam3:{spec.name}"
        self.last: Decision | None = None

    def locate(self, frame: np.ndarray, table) -> tuple[np.ndarray, float] | None:
        paused = self.arm is not None and bool(getattr(self.arm, "servo_running", False))
        if paused:
            self.arm.stop_servo()
        t = time.perf_counter()
        try:
            found = candidates(self.client.segment(frame, [self.spec.prompt]), table)
            decision = decide(found, self.spec, self.pick)
        except (OSError, RuntimeError, ValueError) as exc:
            decision = Decision(self.spec, [], None, f"SAM 3 failed: {exc}")
        finally:
            if paused:
                self.arm.start_servo()
        decision.ms = (time.perf_counter() - t) * 1000.0
        self.last = decision
        self._save(frame, decision)
        if decision.target is None:
            _log.warning("Not moving. %s", decision.reason)
            return None
        _log.info("SAM 3: %s (%.0f ms)", decision.reason, decision.ms)
        return decision.target.xy.copy(), decision.target.score

    def draw(self, img: np.ndarray) -> np.ndarray:
        return draw(img, self.last)

    def _save(self, frame: np.ndarray, decision: Decision) -> None:
        if self.log_dir is None:
            return
        try:
            import cv2  # noqa: PLC0415

            self.log_dir.mkdir(parents=True, exist_ok=True)
            tag = "target" if decision.target is not None else "refused"
            name = re.sub(r"[^A-Za-z0-9]+", "_", self.spec.name).strip("_")
            path = self.log_dir / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{name}_{tag}.jpg"
            cv2.imwrite(str(path), cv2.cvtColor(draw(frame, decision), cv2.COLOR_RGB2BGR))
        except Exception as exc:  # noqa: BLE001 -- a log image must never block the reach
            _log.warning("Could not save the SAM 3 decision image: %s", exc)

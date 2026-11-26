"""Camera capture for recording and for policy rollout.

One capture path, shared by both, so the frames a policy acts on are the same
kind of frames it was trained on.

Three things here are not optional, and each one silently corrupts a dataset
rather than failing:

**Stable device paths.** `/dev/video4` is assignment order and changes on
replug. Two cameras that swap identity swap the model's inputs, and nothing
downstream can detect it. Devices are resolved through `/dev/v4l/by-id/`, which
is keyed on the USB serial number.

**Locked controls.** The Brio defaults to auto everything. Auto-exposure and
auto-white-balance make appearance drift between demos and again at deployment.
Worse, `exposure_dynamic_framerate` lets the camera *drop frame rate* in dim
light -- 25 Hz capture quietly becomes 15 -- and `focus_automatic_continuous`
hunts mid-episode, changing apparent scale and blurring frames. All four are
pinned before capture starts. v4l2-ctl does this, not OpenCV, whose property
setting for these is unreliable on the Brio.

**Latest-frame semantics.** Each camera runs its own thread holding exactly one
frame. The control loop never blocks on a camera; it takes whatever is newest
and records how old it was. A camera that stalls shows up as rising frame age
instead of as a stalled robot.

MJPG rather than YUYV: the Brio offers 120 fps at 640x480 compressed against 30
uncompressed, and the frames are downscaled for the network regardless.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)

BY_ID = Path("/dev/v4l/by-id")

#: Applied before capture, in this order. `auto_exposure=1` is UVC's Manual
#: Mode (3 is Aperture Priority, the default). The temperature and time values
#: only become settable once their automatic counterpart is off, which is why
#: the pairs are ordered.
LOCK_SEQUENCE: tuple[tuple[str, str], ...] = (
    ("auto_exposure", "1"),
    ("exposure_dynamic_framerate", "0"),
    ("white_balance_automatic", "0"),
    ("focus_automatic_continuous", "0"),
)


@dataclass
class CameraSpec:
    """Which physical camera, and how it is configured."""

    name: str  # 'wrist' or 'static'; becomes the HDF5 dataset name
    serial: str  # USB serial, the stable half of the by-id path
    #: What the network receives. Fixed at training time -- changing it later
    #: invalidates a checkpoint.
    width: int = 640
    height: int = 480
    #: What the SENSOR is read at, before any crop. The Brio is a 4K sensor, so
    #: asking it for 640x480 throws most of it away: a 3 cm cube lands on ~30
    #: pixels and, after ResNet18's stride-32, about ONE token out of 300.
    #: Reading 1920x1080 and cropping the workspace gives the same cube real
    #: sensor detail. Measured free -- 1080p+crop+resize still runs at 40 fps
    #: with the same per-frame cost, because the work hides inside the wait for
    #: the next frame. None means capture natively at width x height.
    capture_width: int | None = None
    capture_height: int | None = None
    #: (x0, y0, x1, y1) in CAPTURE pixels, applied before the resize to
    #: width x height. None uses the whole frame.
    roi: tuple[int, int, int, int] | None = None
    #: Ask for more than the control rate. The tick acts on the newest frame, so
    #: a faster camera means a fresher one: measured here, 30 fps gives a p95
    #: frame age of 31.4 ms and 60 gives 23.9 ms.
    #:
    #: You will not get 60. `exposure` is in units of 100 us, so 250 = 25 ms and
    #: caps the sensor at 40 fps -- measured 38.7. Requesting 90 changes nothing
    #: for the same reason. Lower the exposure (and add light) to go faster.
    fps: int = 60
    exposure: int = 250  # exposure_time_absolute, 3..2047, units of 100 us
    white_balance: int = 4000  # Kelvin, 2000..7500
    focus: int = 0  # focus_absolute, 0..255 step 5

    def device(self) -> Path:
        """Resolve to /dev/videoN through the serial-keyed symlink."""
        link = BY_ID / f"usb-046d_Logitech_BRIO_{self.serial}-video-index0"
        if not link.exists():
            available = sorted(p.name for p in BY_ID.glob("*-video-index0")) if BY_ID.exists() else []
            raise FileNotFoundError(
                f"camera {self.name!r} (serial {self.serial}) not found at {link}.\n"
                f"    connected: {available or 'none'}\n"
                "    Run `python -m quest_teleop2.vision.camera --list`."
            )
        return link.resolve()


def list_cameras() -> list[tuple[str, Path]]:
    """(serial, /dev/videoN) for every Brio present, by stable path."""
    if not BY_ID.exists():
        return []
    out = []
    for link in sorted(BY_ID.glob("usb-046d_Logitech_BRIO_*-video-index0")):
        serial = link.name.split("BRIO_")[1].split("-video")[0]
        out.append((serial, link.resolve()))
    return out


def lock_controls(spec: CameraSpec) -> dict[str, str]:
    """Pin exposure, white balance, focus and frame rate. Returns what stuck.

    Verified by reading the values back: a control silently refusing to change
    is exactly the kind of thing that shows up as unexplained appearance drift
    three hundred demos later.
    """
    dev = str(spec.device())
    settings = list(LOCK_SEQUENCE) + [
        ("exposure_time_absolute", str(spec.exposure)),
        ("white_balance_temperature", str(spec.white_balance)),
        ("focus_absolute", str(spec.focus)),
    ]
    for ctrl, value in settings:
        subprocess.run(["v4l2-ctl", "-d", dev, f"--set-ctrl={ctrl}={value}"],
                       capture_output=True, text=True, timeout=5)

    applied: dict[str, str] = {}
    names = [c for c, _ in settings]
    result = subprocess.run(["v4l2-ctl", "-d", dev, f"--get-ctrl={','.join(names)}"],
                            capture_output=True, text=True, timeout=5)
    for line in result.stdout.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            applied[k.strip()] = v.strip()

    for ctrl, want in settings:
        got = applied.get(ctrl)
        if got is None:
            continue
        # Menu controls read back as "1 (Manual Mode)", so compare the numeric
        # part only -- a plain string compare cries wolf on every lock.
        if got.split()[0] != want:
            _log.warning("camera %s: %s is %s, wanted %s", spec.name, ctrl, got, want)
    return applied


class Camera:
    """One camera, captured on its own thread, holding only the newest frame."""

    def __init__(self, spec: CameraSpec) -> None:
        self.spec = spec
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._captured = 0
        self._started = 0.0

    def start(self, lock: bool = True) -> None:
        import cv2  # noqa: PLC0415 -- deferred so --help works without opencv

        if lock:
            lock_controls(self.spec)

        dev = str(self.spec.device())
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            # Name the culprit. A V4L2 device is exclusive, and "could not open"
            # with no owner is a needlessly hard thing to chase.
            holder = ""
            try:
                out = subprocess.run(["fuser", "-v", dev], capture_output=True,
                                     text=True, timeout=5)
                blob = (out.stdout + out.stderr).strip().splitlines()
                if len(blob) > 1:
                    holder = "\n    held by: " + "; ".join(b.strip() for b in blob[1:])
            except Exception:  # noqa: BLE001 -- diagnostics only
                pass
            raise RuntimeError(
                f"cannot open {self.spec.name} at {dev}. A camera can only be "
                f"opened once at a time.{holder}"
            )
        # FOURCC before the size: setting it after can leave the driver in the
        # previous format at the new resolution.
        cap_w = self.spec.capture_width or self.spec.width
        cap_h = self.spec.capture_height or self.spec.height
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_h)
        cap.set(cv2.CAP_PROP_FPS, self.spec.fps)
        # NOT 1. Measured on this Brio: BUFFERSIZE=1 gives 15.0 fps, 2 or more
        # gives the full 30.0 -- with a single slot a frame arriving during
        # decode has nowhere to go and is dropped, halving the rate.
        #
        # Freshness does not come from starving the driver. The capture thread
        # below drains continuously and keeps only the newest frame, so a small
        # driver queue costs nothing and a deep one is never allowed to fill.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (got_w, got_h) != (cap_w, cap_h):
            cap.release()
            raise RuntimeError(
                f"{self.spec.name}: asked to capture {cap_w}x{cap_h}, got "
                f"{got_w}x{got_h}. The model's input size is fixed at training "
                "time, so a silent substitution here would corrupt every frame."
            )
        if self.spec.roi is not None:
            x0, y0, x1, y1 = self.spec.roi
            if not (0 <= x0 < x1 <= got_w and 0 <= y0 < y1 <= got_h):
                cap.release()
                raise ValueError(f"{self.spec.name}: roi {self.spec.roi} outside {got_w}x{got_h}")

        self._cap = cap
        self._running = True
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._loop, name=f"cam-{self.spec.name}", daemon=True)
        self._thread.start()
        # Wait for a first frame, so callers never see None.
        deadline = time.monotonic() + 5.0
        while self.latest()[0] is None:
            if time.monotonic() > deadline:
                self.stop()
                raise RuntimeError(f"{self.spec.name}: no frame within 5 s of opening {dev}")
            time.sleep(0.01)
        _log.info("camera %s: %dx%d MJPG on %s", self.spec.name, got_w, got_h, dev)

    def _loop(self) -> None:
        import cv2  # noqa: PLC0415

        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            # BGR from OpenCV; the model wants RGB. Convert once, here, so no
            # consumer has to remember -- a channel swap is invisible in a
            # thumbnail and fatal to a fine-tuned ImageNet backbone.
            if self.spec.roi is not None:
                x0, y0, x1, y1 = self.spec.roi
                frame = frame[y0:y1, x0:x1]
            if (frame.shape[1], frame.shape[0]) != (self.spec.width, self.spec.height):
                # INTER_AREA is the right filter for downscaling; the default
                # bilinear aliases and would put shimmer into every frame.
                frame = cv2.resize(frame, (self.spec.width, self.spec.height),
                                   interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now = time.monotonic()
            with self._lock:
                self._frame = rgb
                self._stamp = now
                if self._captured == 0:
                    self._started = now  # exclude open() latency from the rate
                self._captured += 1

    def latest(self) -> tuple[np.ndarray | None, float]:
        """(frame RGB HWC uint8, age in seconds). Never blocks."""
        with self._lock:
            if self._frame is None:
                return None, float("inf")
            return self._frame, time.monotonic() - self._stamp

    def snapshot(self) -> tuple[np.ndarray | None, float, int]:
        """(frame, age, sequence) -- `latest()` plus frame identity.

        `sequence` counts frames captured since start, so a poller can tell a
        NEW frame from the same one read twice. Reading `latest()` and a counter
        separately would race: the capture thread can swap the frame between the
        two calls, pairing frame N with count N+1. All three come from one lock
        acquisition for that reason.
        """
        with self._lock:
            if self._frame is None:
                return None, float("inf"), self._captured
            return self._frame, time.monotonic() - self._stamp, self._captured

    @property
    def measured_fps(self) -> float:
        """Steady-state rate, measured from the FIRST frame."""
        with self._lock:
            n, started = self._captured, self._started
        elapsed = time.monotonic() - started
        return (n - 1) / elapsed if n > 1 and elapsed > 0 else 0.0

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        cap = getattr(self, "_cap", None)
        if cap is not None:
            cap.release()


class CameraRig:
    """Several cameras as one unit. Order is fixed and must match training."""

    def __init__(self, specs: list[CameraSpec]) -> None:
        self.specs = specs
        self.cameras = [Camera(s) for s in specs]

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.specs]

    def start(self, lock: bool = True) -> None:
        for cam in self.cameras:
            cam.start(lock=lock)

    def grab(self) -> tuple[list[np.ndarray], float]:
        """Newest frame from each camera, in rig order, plus the worst age."""
        frames, worst = [], 0.0
        for cam in self.cameras:
            frame, age = cam.latest()
            if frame is None:
                raise RuntimeError(f"camera {cam.spec.name} has produced no frames")
            frames.append(frame)
            worst = max(worst, age)
        return frames, worst

    def snapshot(self) -> list[tuple[np.ndarray | None, float, int]]:
        """Per camera, in rig order: (frame, age, sequence). Never raises.

        Unlike `grab()`, a camera that has produced nothing yields `None` rather
        than raising -- a recorder should keep running through a hiccup, where a
        control loop must not act on a missing frame.
        """
        return [cam.snapshot() for cam in self.cameras]

    def stop(self) -> None:
        for cam in self.cameras:
            cam.stop()

    def __enter__(self) -> CameraRig:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


# --------------------------------------------------------------------------- #
# CLI: identify cameras, check framing, and confirm the controls actually stuck.
# --------------------------------------------------------------------------- #


def _sweep_focus(specs: list[CameraSpec], step: int = 15) -> int:
    """Step focus across its range and report the sharpest, per camera.

    Scores the CENTRE crop and the whole frame separately, because a lens has
    one focal plane and a scene usually has two things in it. A wrist camera
    aimed past a nearby object at a room across the lab is mostly room by area,
    so a whole-frame score picks the focus that suits the ROOM -- which is
    infinity, monotonically, and completely wrong for the object being grasped.
    The centre column is the one that matters; the full column is kept so a
    disagreement between them is visible rather than silent.

    Point the camera at the scene it will actually work in before running this.
    For a wrist camera that means parking the arm at a grasp pose, not at home.

    Sweeping by hand does not work: `start()` re-applies the control locks, so a
    focus set externally between runs is reset to the spec value before the first
    frame is taken. Measured that way, all six settings of a manual sweep
    returned the same sharpness -- they were all focus 0.
    """
    import cv2  # noqa: PLC0415

    def score(frame: np.ndarray) -> tuple[float, float]:
        grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        h, w = grey.shape
        centre = grey[h // 4:3 * h // 4, w // 4:3 * w // 4]
        return (float(cv2.Laplacian(centre, cv2.CV_64F).var()),
                float(cv2.Laplacian(grey, cv2.CV_64F).var()))

    for spec in specs:
        cam = Camera(spec)
        cam.start(lock=True)
        dev = str(spec.device())
        print(f"\n\033[1m{spec.serial}\033[0m  sweeping focus 0..255 step {step}")
        print("  Point the camera at the scene it will WORK in -- for a wrist camera,")
        print("  park the arm at a grasp pose first.\n")
        results: list[tuple[int, float, float]] = []
        try:
            for focus in range(0, 256, step):
                subprocess.run(["v4l2-ctl", "-d", dev, f"--set-ctrl=focus_absolute={focus}"],
                               capture_output=True, timeout=5)
                time.sleep(0.45)  # the lens is mechanical; let it settle
                frame, _ = cam.latest()
                centre, full = score(frame)
                results.append((focus, centre, full))
        finally:
            cam.stop()

        best_c = max(results, key=lambda r: r[1])
        best_f = max(results, key=lambda r: r[2])
        print(f"  {'focus':>7s}  {'centre':>9s}  {'full':>9s}")
        for focus, centre, full in results:
            bar = "#" * int(centre / max(best_c[1], 1e-9) * 32)
            mark = "  <- sharpest centre" if focus == best_c[0] else ""
            print(f"  {focus:7d}  {centre:9.1f}  {full:9.1f}  {bar}{mark}")

        spread = best_c[1] / max(min(r[1] for r in results), 1e-9)
        print(f"\n  \033[1mbest focus_absolute={best_c[0]}\033[0m  "
              f"(centre sharpest/blurriest = {spread:.2f}x)")
        if best_f[0] != best_c[0]:
            print(f"  \033[33mThe whole frame prefers {best_f[0]} -- the periphery sits at a "
                  f"different distance\n  than the centre. Trust the centre: that is where "
                  "the object is.\033[0m")
        if spread < 1.15:
            print("  \033[33mBarely any variation.\033[0m Either the scene has little fine "
                  "detail to focus on, or\n  the depth of field already covers it. Point at "
                  "something textured to confirm the\n  control works before trusting a flat result.")
    return 0


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Identify and check the cameras.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "  --list     serials and device paths of every Brio present\n"
            "  --probe    open each, lock controls, measure real FPS and frame age\n"
            "  --preview  write a JPEG per camera so framing can be checked\n"
        ),
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--preview", type=Path, default=None, metavar="DIR")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--serials", default="", help="comma-separated; default is all present")
    parser.add_argument("--focus", type=int, default=None)
    parser.add_argument("--exposure", type=int, default=None)
    parser.add_argument("--wb", type=int, default=None)
    parser.add_argument("--sweep-focus", action="store_true",
                        help="step focus 0..255 and report the sharpest. Do this rather than "
                             "setting focus_absolute by hand between runs: every start() "
                             "re-applies the locks, so an external setting is overwritten.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    found = list_cameras()
    if not found:
        print("\033[31mNo Logitech Brio found.\033[0m Check the USB connection.")
        return 1

    acting = args.probe or args.preview or args.sweep_focus
    if args.list or not acting:
        print(f"\n\033[1m{len(found)} Brio(s)\033[0m")
        for serial, dev in found:
            print(f"  serial {serial}   ->  {dev}")
        print("\nUse the SERIAL in config, never /dev/videoN -- the number changes on replug.")
        if not acting:
            return 0

    serials = [s.strip() for s in args.serials.split(",") if s.strip()] or [s for s, _ in found]
    overrides = {k: v for k, v in
                 (("focus", args.focus), ("exposure", args.exposure),
                  ("white_balance", args.wb)) if v is not None}
    specs = [CameraSpec(name=f"cam{i}", serial=s, **overrides) for i, s in enumerate(serials)]

    if args.sweep_focus:
        return _sweep_focus(specs)

    rig = CameraRig(specs)
    try:
        rig.start(lock=True)
    except Exception as exc:
        print(f"\033[31m{exc}\033[0m")
        return 1

    try:
        print(f"\n\033[1mMeasuring for {args.seconds:.0f}s\033[0m")
        time.sleep(args.seconds)
        frames, worst = rig.grab()
        for cam, frame in zip(rig.cameras, frames):
            fps = cam.measured_fps
            _, age = cam.latest()
            flag = "" if fps >= 24 else "  \033[33m<- below 25 Hz\033[0m"
            print(f"  {cam.spec.serial}  {fps:5.1f} fps  frame age {age * 1000:5.1f} ms  "
                  f"{frame.shape[1]}x{frame.shape[0]}{flag}")
        print(f"  worst frame age {worst * 1000:.1f} ms  (a 40 ms tick tolerates well under that)")

        if args.preview:
            import cv2  # noqa: PLC0415

            args.preview.mkdir(parents=True, exist_ok=True)
            for cam, frame in zip(rig.cameras, frames):
                out = args.preview / f"{cam.spec.serial}.jpg"
                cv2.imwrite(str(out), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                print(f"  wrote {out}")
            print("\nOpen these and check framing before recording anything.")
    finally:
        rig.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

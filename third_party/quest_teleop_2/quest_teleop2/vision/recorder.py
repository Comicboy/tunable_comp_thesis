"""Write a CameraRig's feeds to video files, off the control loop.

Used to film a policy rollout: the robot from the outside on a phone, and what
the robot itself sees, edited together afterwards.

Two things decide the design.

**The rig is shared, not reopened.** A V4L2 device can only be opened once, and
`run_policy.py` already holds both cameras for inference. So this pulls from the
same `CameraRig` rather than opening its own -- which is also the honest thing to
record, since the frames written are the frames the policy acted on, from one
capture path and one clock.

**Encoding never touches the control loop.** One thread per camera, and the
control loop never waits on them. This is not stylistic: a 600 ms stall inside
the servo loop stops the RTDE heartbeat and the robot answers with a
PROTECTIVE_STOP before the arm has visibly moved.

Output is MJPEG in a QuickTime container. That combination was chosen by probing
what this OpenCV build can actually write -- it refuses ProRes and H.264, having
neither encoder compiled in. MJPEG is intra-frame, so every frame stands alone
and scrubbing in an editor is instant, unlike long-GOP H.264. `.mov` is native
on macOS and imports into DaVinci Resolve directly.

Nothing is drawn on the picture: no overlay, no timestamp, no border. A burned-in
mark cannot be removed later, so the metadata that would sync these clips to
other footage goes in a sidecar `run.json` instead.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


# NOTE on the console noise. Opening each writer prints:
#
#   OpenCV: FFMPEG: tag 0x47504a4d/'MJPG' is not supported with codec id 7 and
#   format 'mov / QuickTime / MOV' ... fallback to use tag 0x6765706a/'jpeg'
#
# That is correct and expected -- 'jpeg' is the right QuickTime tag for Motion
# JPEG, and the fallback is the muxer doing its job. It is NOT suppressible from
# Python: it comes from OpenCV's own C++ layer, so neither
# OPENCV_FFMPEG_LOGLEVEL nor cv2.utils.logging.setLogLevel(SILENT) silences it
# (both tried). Ignore it in the logs.

_log = logging.getLogger(__name__)

#: JPEG quality for each frame. 95 is visually lossless on 640x480 camera
#: footage; the files are small enough at this size that trading quality for
#: bytes would be a poor deal on material destined for a presentation.
QUALITY = 95


@dataclass
class CameraTrack:
    """One camera's output file and the timing needed to sync it."""

    name: str
    path: Path
    #: Carried here rather than looked up by position in the rig: a rig index
    #: stops matching a track index the moment one camera writes two files.
    width: int = 0
    height: int = 0
    frames: int = 0
    dropped: int = 0
    #: Rate stamped into the container header. Kept so the sidecar can report
    #: it next to the rate actually achieved -- if they disagree the clip needs
    #: a speed change in the editor, and that must be visible, not inferred.
    header_fps: float = 0.0
    #: Seconds from the recorder's start, one per written frame. This is what
    #: lets an editor correct a drifting clip instead of discovering a silent
    #: desync halfway through a cut.
    times: list[float] = field(default_factory=list)

    @property
    def measured_fps(self) -> float:
        if len(self.times) < 2:
            return 0.0
        span = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / span if span > 0 else 0.0


class VideoRecorder:
    """Records every camera in a rig to its own file, on its own thread."""

    def __init__(self, rig, out_dir: Path, quality: int = QUALITY,
                 container: str = "mov",
                 annotate: dict | None = None) -> None:
        """`annotate` maps a camera name to a callable(rgb) -> rgb.

        An annotated camera produces a SECOND file, `<name>_detect.<container>`,
        from the same frames. The clean feed is always written: a burned-in mark
        cannot be removed later, so the overlay is an extra track to blend or
        drop in an editor, never a decision already taken for you.

        Drawing happens on the writer thread, so an expensive overlay costs
        frames in the overlay clip and never stalls the control loop.
        """
        self.rig = rig
        self.out_dir = Path(out_dir)
        self.quality = quality
        self.container = container
        self.annotate = dict(annotate or {})
        self.tracks: dict[str, CameraTrack] = {}
        self._threads: list[threading.Thread] = []
        self._running = False
        self._t0 = 0.0
        self._started_at = ""
        self._rates: dict[str, float] = {}

    # -- lifecycle --------------------------------------------------------- #

    def start(self) -> None:
        """Begin recording. The rig must already be started."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._rates = self._measure_rates()
        self._t0 = time.monotonic()
        self._started_at = datetime.now(UTC).isoformat()
        self._running = True
        for index, cam in enumerate(self.rig.cameras):
            name = cam.spec.name
            track = CameraTrack(name=name,
                                path=self.out_dir / f"{name}.{self.container}",
                                width=cam.spec.width, height=cam.spec.height)
            self.tracks[name] = track
            overlay = None
            if name in self.annotate:
                overlay = CameraTrack(
                    name=f"{name}_detect",
                    path=self.out_dir / f"{name}_detect.{self.container}",
                    width=cam.spec.width, height=cam.spec.height)
                self.tracks[overlay.name] = overlay
            thread = threading.Thread(
                target=self._record,
                args=(index, cam, track, overlay, self.annotate.get(name)),
                name=f"rec-{name}", daemon=True)
            thread.start()
            self._threads.append(thread)
        _log.info("Recording %s to %s", ", ".join(self.tracks), self.out_dir)

    def stop(self, extra: dict | None = None) -> dict:
        """Stop, flush, and write run.json. Returns the summary it wrote."""
        self._running = False
        for thread in self._threads:
            thread.join(timeout=10.0)
        self._threads = []

        summary = {
            "started_at": self._started_at,
            "duration_s": round(time.monotonic() - self._t0, 3),
            "container": self.container,
            "codec": "mjpeg",
            "quality": self.quality,
            "cameras": {
                t.name: {
                    "file": t.path.name,
                    "frames": t.frames,
                    "dropped": t.dropped,
                    # Stamped in the file vs actually achieved. Equal means the
                    # clip plays at true speed; a gap is the speed change the
                    # editor needs to apply.
                    "fps_header": t.header_fps,
                    "fps_measured": round(t.measured_fps, 2),
                    "width": t.width,
                    "height": t.height,
                }
                for t in self.tracks.values()
            },
            # Per-frame offsets from `started_at`. Rounded to a millisecond --
            # far finer than a 40 fps frame period, and it keeps the file from
            # being mostly float noise.
            "frame_times": {t.name: [round(x, 3) for x in t.times]
                            for t in self.tracks.values()},
        }
        if extra:
            summary.update(extra)
        (self.out_dir / "run.json").write_text(json.dumps(summary, indent=2))
        return summary

    # -- worker ------------------------------------------------------------ #

    def _measure_rates(self, window: float = 0.5) -> dict[str, float]:
        """Steady-state fps per camera, sampled over a short window.

        NOT `Camera.measured_fps`: that averages over the camera's whole life,
        including the first second after open when delivery has not settled and
        the rig was running with nothing consuming it. Measured here, that read
        38.95 fps against a true 40.0 -- a 2.5% error, which is a quarter-second
        of drift over a ten-second clip and enough to pull a cut out of sync.

        Costs one `window` before recording starts. That happens before the
        rollout loop, so it buys a little head room on the clip rather than
        losing any of the motion.
        """
        first = {c.spec.name: c.snapshot()[2] for c in self.rig.cameras}
        t0 = time.monotonic()
        time.sleep(window)
        elapsed = time.monotonic() - t0
        rates = {}
        for cam in self.rig.cameras:
            name = cam.spec.name
            captured = cam.snapshot()[2] - first[name]
            rate = captured / elapsed if elapsed > 0 else 0.0
            if rate < 1.0:
                _log.warning("%s: measured %.1f fps over %.1fs; stamping the "
                             "requested %d instead", name, rate, window, cam.spec.fps)
                rate = float(cam.spec.fps)
            rates[name] = round(rate, 2)
        return rates

    def _open(self, track: CameraTrack, spec, header_fps: float):
        import cv2  # noqa: PLC0415

        track.header_fps = header_fps
        writer = cv2.VideoWriter(
            str(track.path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            header_fps,
            (spec.width, spec.height),
        )
        if not writer.isOpened():
            _log.error("Could not open a writer for %s at %s", spec.name, track.path)
            return None
        try:
            writer.set(cv2.VIDEOWRITER_PROP_QUALITY, float(self.quality))
        except Exception:  # noqa: BLE001 -- not every build exposes the property
            pass
        return writer

    def _record(self, index: int, cam, track: CameraTrack,
                overlay: CameraTrack | None = None, draw=None) -> None:
        import cv2  # noqa: PLC0415

        spec = cam.spec
        header_fps = self._rates[spec.name]
        writer = self._open(track, spec, header_fps)
        if writer is None:
            return
        # The overlay is a SECOND file from the same frames, never a
        # replacement: a burned-in mark cannot be removed later, so the clean
        # feed has to survive. In an editor they are two aligned tracks and the
        # overlay is a choice rather than a decision already made.
        writer2 = self._open(overlay, spec, header_fps) if overlay is not None else None

        # Poll well above the capture rate so a new frame is picked up promptly
        # rather than aliasing against it. Sequence numbers make the extra polls
        # nearly free: a repeat costs one lock acquisition and is skipped, not
        # written.
        period = 1.0 / (max(spec.fps, 1) * 4.0)
        last_seq = -1
        try:
            while self._running:
                frame, _age, seq = cam.snapshot()
                if frame is None or seq == last_seq:
                    time.sleep(period)
                    continue
                if last_seq >= 0 and seq > last_seq + 1:
                    # The camera produced frames faster than this thread consumed
                    # them. Count it: a silently short clip is worse than a
                    # noted one.
                    track.dropped += seq - last_seq - 1
                last_seq = seq
                # The rig hands out RGB; VideoWriter wants BGR. Converting here
                # keeps it off the control loop, and copies the frame so the
                # capture thread is free to overwrite its buffer.
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                track.frames += 1
                track.times.append(time.monotonic() - self._t0)
                if writer2 is not None:
                    writer2.write(cv2.cvtColor(draw(frame), cv2.COLOR_RGB2BGR))
                    overlay.frames += 1
                    overlay.times.append(track.times[-1])
                # No sleep here: poll again at once. Sleeping after a write
                # spends the encode time AND the period before looking, which
                # is enough to miss the next frame at 40 fps. Sleep only when
                # there was nothing new to take.
        finally:
            writer.release()
            if writer2 is not None:
                writer2.release()
            if track.dropped:
                _log.warning("%s: %d frame(s) dropped while writing",
                             spec.name, track.dropped)


def summarise(summary: dict) -> str:
    """One-line-per-camera report, for a console."""
    lines = []
    for name, cam in summary.get("cameras", {}).items():
        drift = abs(cam.get("fps_header", 0) - cam["fps_measured"])
        flag = ""
        if cam["dropped"]:
            flag += f"  \033[33m{cam['dropped']} dropped\033[0m"
        if drift > 0.5:
            flag += (f"  \033[33mheader {cam['fps_header']:.1f} vs actual "
                     f"{cam['fps_measured']:.1f} -- retime in the editor\033[0m")
        lines.append(
            f"  {name:8s} {cam['frames']:5d} frames  {cam['fps_measured']:5.1f} fps  "
            f"{cam['width']}x{cam['height']}  -> {cam['file']}{flag}"
        )
    return "\n".join(lines)

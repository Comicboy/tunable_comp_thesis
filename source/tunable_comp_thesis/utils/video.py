"""Streaming mp4 writer for frames rendered by an env (teleop_lift.py, rollout_policy.py).

Every frame is piped to an ffmpeg process as soon as it is rendered, so nothing piles up in memory
(at 1280 x 720 one frame is 2.8 MB: a 30 s episode at 50 fps would be over 4 GB if kept as a list).
The ffmpeg binary comes from imageio-ffmpeg (installed with moviepy, which gymnasium's RecordVideo
uses) or from the system; if neither is there: pip install imageio-ffmpeg.

    writer = VideoWriter("logs/.../ep094.mp4", fps=50)
    for ...:
        writer.add(env.unwrapped.render())     # (H, W, 3) uint8
    path = writer.close(rename_to="logs/.../ep094_success.mp4")   # or writer.discard()

The file is written under a temporary name (<name>.part.mp4) and gets its final name only in
close(), so an interrupted run never leaves a file that looks complete.
"""

import os
import shutil
import subprocess

import numpy as np


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # not installed, or its binary missing
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise RuntimeError("no ffmpeg found for writing videos: pip install imageio-ffmpeg")
        return exe


class VideoWriter:
    """H.264 mp4 from RGB frames, one frame per add(), played back at `fps`."""

    def __init__(self, path: str, fps: float):
        self.path = path
        self.fps = fps
        self.frames = 0
        self._tmp = os.path.splitext(path)[0] + ".part.mp4"
        self._proc: subprocess.Popen | None = None
        self._size: tuple[int, int] | None = None
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def _start(self, width: int, height: int):
        self._size = (width, height)
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{self.fps:g}", "-i", "-",
               "-an", "-vcodec", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", self._tmp]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def add(self, frame) -> None:
        if frame is None:
            return
        img = np.ascontiguousarray(np.asarray(frame)[..., :3], dtype=np.uint8)
        h, w = img.shape[:2]
        if self._proc is None:
            # H.264 in yuv420p needs even sizes
            self._start(w - w % 2, h - h % 2)
        img = img[: self._size[1], : self._size[0]]
        self._proc.stdin.write(img.tobytes())
        self.frames += 1

    def close(self, rename_to: str | None = None) -> str | None:
        """Finish the file; returns its path (None if no frame was added)."""
        if self._proc is None:
            return None
        self._proc.stdin.close()
        self._proc.wait()
        self._proc = None
        final = rename_to or self.path
        os.replace(self._tmp, final)
        return final

    def discard(self) -> None:
        if self._proc is not None:
            self._proc.stdin.close()
            self._proc.wait()
            self._proc = None
        if os.path.exists(self._tmp):
            os.remove(self._tmp)

    @property
    def seconds(self) -> float:
        return self.frames / self.fps
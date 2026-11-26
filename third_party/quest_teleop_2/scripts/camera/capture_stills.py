#!/usr/bin/env python
"""High-resolution stills from the rig cameras, for figures.

    python scripts/camera/capture_stills.py --label table_empty
    python scripts/camera/capture_stills.py --label table_arm --camera static

The pipeline runs these cameras at 640x480 because that is what the networks
were trained on. The sensors are 4K, so a figure explaining the pipeline can
show the scene at 3840x2160 while the policy sees a sixth of the linear
resolution -- which is itself worth a sentence in the caption.

Exposure, white balance and focus come from config/cameras.yaml, so a still
looks like what the pipeline sees rather than like a different camera. Nothing
in that file is modified.

The rig must not be running: a camera opens once at a time.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quest_teleop2.config import CamerasCfg  # noqa: E402
from quest_teleop2.vision.camera import Camera  # noqa: E402

GREEN, YELLOW, BOLD, RESET = "\033[32m", "\033[33m", "\033[1m", "\033[0m"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True, help="goes in the filename")
    p.add_argument("--camera", default="", help="one name from cameras.yaml; default both")
    p.add_argument("--width", type=int, default=3840)
    p.add_argument("--height", type=int, default=2160)
    p.add_argument("--average", type=int, default=8,
                   help="frames to average. Kills sensor noise on a STATIC scene; "
                        "use 1 if anything in the frame is moving")
    p.add_argument("--settle", type=float, default=2.0,
                   help="seconds to let exposure and focus settle before capturing")
    p.add_argument("--out", type=Path, default=Path("figures"))
    args = p.parse_args()

    import cv2  # noqa: PLC0415

    specs = CamerasCfg.load().specs([args.camera] if args.camera else None)
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    for spec in specs:
        hi = replace(spec, width=args.width, height=args.height,
                     capture_width=args.width, capture_height=args.height,
                     roi=None, fps=30)
        cam = Camera(hi)
        print(f"\n{BOLD}{spec.name}{RESET}  {args.width}x{args.height}  "
              f"focus {spec.focus}  exposure {spec.exposure}  wb {spec.white_balance}")
        cam.start()
        try:
            time.sleep(args.settle)
            acc, kept = None, 0
            deadline = time.time() + 10.0
            while kept < args.average and time.time() < deadline:
                rgb, age, _seq = cam.snapshot()
                if rgb is None or age > 0.5:
                    time.sleep(0.02)
                    continue
                f = rgb.astype(np.float64)
                acc = f if acc is None else acc + f
                kept += 1
                time.sleep(0.05)
        finally:
            cam.stop()
        if not kept:
            print(f"  {YELLOW}no frames{RESET}")
            continue
        img = np.rint(acc / kept).astype(np.uint8)
        path = args.out / f"{stamp}_{args.label}_{spec.name}.png"
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])
        mb = path.stat().st_size / 1e6
        print(f"  {GREEN}{path}{RESET}  {img.shape[1]}x{img.shape[0]}  "
              f"{kept} frames averaged  {mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

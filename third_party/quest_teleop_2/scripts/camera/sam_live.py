#!/usr/bin/env python
"""Segment the top-down camera with SAM 3 from text prompts. Runs in the sam3 env.

    ~/miniforge3/envs/sam3/bin/python scripts/camera/sam_live.py --prompt cube
    ~/miniforge3/envs/sam3/bin/python scripts/camera/sam_live.py --prompt "white cube" --prompt bucket --once
    ~/miniforge3/envs/sam3/bin/python scripts/camera/sam_live.py --object cube

--object applies the reach's rules to a config/objects.yaml name or any short
phrase ("red mug") and shows what --sam would do: the target it would approach,
or why it would refuse. Nothing moves.

Prints each detection's score, pixel area and table position. The table position
uses config/table.yaml, which is exact only at the calibrated object's height.
q quits the live view; --once saves one annotated frame to figures/.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quest_teleop2.config import CamerasCfg  # noqa: E402
from quest_teleop2.vision.camera import CameraRig  # noqa: E402
from quest_teleop2.vision.sam import (  # noqa: E402
    SERVER_FLOOR,
    candidates,
    decide,
    draw,
    object_spec,
)
from quest_teleop2.vision.table import TableMap  # noqa: E402

COLOURS = [(0, 200, 255), (0, 220, 0), (255, 80, 80), (255, 0, 255), (255, 255, 0)]


def load_model(threshold: float):
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError  # noqa: PLC0415
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", (UserWarning, FutureWarning))
        from sam3.model.sam3_image_processor import Sam3Processor  # noqa: PLC0415
        from sam3.model_builder import build_sam3_image_model  # noqa: PLC0415

    try:
        model = build_sam3_image_model()
    except (GatedRepoError, HfHubHTTPError) as exc:
        raise SystemExit(
            "Cannot download facebook/sam3: request access at "
            "https://huggingface.co/facebook/sam3, then log in with "
            f"~/miniforge3/envs/sam3/bin/hf auth login.\n  ({type(exc).__name__})") from None
    return Sam3Processor(model, confidence_threshold=threshold)


def segment(processor, rgb: np.ndarray, prompts: list[str], bf16: bool) -> list[dict]:
    found = []
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        state = processor.set_image(Image.fromarray(rgb))
        for i, prompt in enumerate(prompts):
            out = processor.set_text_prompt(prompt=prompt, state=state)
            masks = out["masks"].squeeze(1).cpu().numpy()
            for mask, box, score in zip(masks, out["boxes"].float().cpu().numpy(),
                                        out["scores"].float().cpu().numpy()):
                ys, xs = np.nonzero(mask)
                if xs.size:
                    found.append({"prompt": prompt, "colour": COLOURS[i % len(COLOURS)],
                                  "score": float(score), "mask": mask, "box": box,
                                  "px": (float(xs.mean()), float(ys.mean())), "area": int(xs.size)})
    return found


def annotate(rgb: np.ndarray, found: list[dict], table) -> tuple[np.ndarray, list[str]]:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lines = []
    for d in found:
        tint = np.zeros_like(img)
        tint[d["mask"]] = d["colour"]
        img = cv2.addWeighted(img, 1.0, tint, 0.45, 0)
        x0, y0, x1, y1 = (int(v) for v in d["box"])
        cv2.rectangle(img, (x0, y0), (x1, y1), d["colour"], 2)
        where = ""
        if table is not None:
            xy = table.to_table(d["px"])
            where = f" ({xy[0]:+.3f}, {xy[1]:+.3f}) m"
        label = f"{d['prompt']} {d['score']:.2f}{where}"
        cv2.putText(img, label, (x0, max(15, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    d["colour"], 2)
        lines.append(f"{label}  area {d['area']} px")
    return img, lines


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", action="append", default=None, help="text prompt; repeat for more")
    p.add_argument("--threshold", type=float, default=0.5, help="minimum detection score")
    p.add_argument("--camera", default="static")
    p.add_argument("--once", action="store_true", help="save one annotated frame and exit")
    p.add_argument("--fp32", action="store_true", help="full precision instead of bfloat16")
    p.add_argument("--object", default=None,
                   help="a config/objects.yaml name or any short phrase: show the reach's "
                        "decision for it")
    p.add_argument("--sam-pick", choices=("one", "best"), default="one")
    args = p.parse_args()
    prompts = args.prompt or ["cube"]
    if not torch.cuda.is_available():
        print("No CUDA GPU visible.")
        return 1

    try:
        table = TableMap.load()
    except (FileNotFoundError, ValueError):
        table = None
    spec = None
    if args.object is not None:
        if table is None:
            print("--object needs a table calibration: scripts/camera/calibrate_table.py")
            return 1
        spec, _ = object_spec(args.object, need_height=False)
        prompts, args.threshold = [spec.prompt], SERVER_FLOOR
    print("Loading SAM 3...", flush=True)
    t = time.perf_counter()
    processor = load_model(args.threshold)
    print(f"  loaded in {time.perf_counter() - t:.0f} s on {torch.cuda.get_device_name(0)}")

    rig = CameraRig(CamerasCfg.load().specs([args.camera]))
    rig.start(lock=True)
    try:
        time.sleep(1.0)
        last_print = 0.0
        while True:
            frames, _age = rig.grab()
            t = time.perf_counter()
            found = segment(processor, frames[0], prompts, bf16=not args.fp32)
            ms = (time.perf_counter() - t) * 1000.0
            if spec is not None:
                decision = decide(candidates(found, table), spec, args.sam_pick)
                img = cv2.cvtColor(draw(frames[0], decision), cv2.COLOR_RGB2BGR)
                lines = [("TARGET " if decision.target is not None else "REFUSED ")
                         + decision.reason]
            else:
                img, lines = annotate(frames[0], found, table)
            cv2.putText(img, f"{ms:.0f} ms", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
            if args.once or time.monotonic() - last_print > 1.0:
                last_print = time.monotonic()
                print(f"[{ms:4.0f} ms] " + ("  |  ".join(lines) if lines else "nothing found"))
            if args.once:
                out = ROOT / "figures" / f"{datetime.now():%Y%m%d_%H%M%S}_sam_{args.camera}.png"
                out.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(out), img)
                print(f"saved {out}")
                return 0
            cv2.imshow("SAM 3 - q quits", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return 0
    finally:
        rig.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())

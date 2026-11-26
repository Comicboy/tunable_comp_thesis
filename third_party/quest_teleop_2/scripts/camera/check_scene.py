#!/usr/bin/env python
"""Check whether the cube positions are distinguishable from a camera viewpoint.

    python scripts/camera/check_scene.py --live           # live cube readout, for aiming
    python scripts/camera/check_scene.py --separability   # the real test
    python scripts/camera/check_scene.py --null           # control: same scene twice

--live is a simple white-blob detector (threshold, connected components) that
prints (x, y) as you move the cube. It is a diagnostic, not what the policy
does, and it serves as a lower bound: if a threshold can localise the cube from
this viewpoint, a ResNet18 can.

--separability captures frames with the cube at each start, embeds them through
the same ImageNet-pretrained ResNet18 the policy uses, and measures whether the
positions form separable clusters before any training. Reports leave-one-out
nearest-neighbour accuracy and the between/within distance ratio. 4/4 with a
healthy margin removes the viewpoint as a suspect.

--null captures the SAME scene N times as if they were different positions.
Whatever separability it reports is the noise floor, not signal.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quest_teleop2.vision.camera import Camera, CameraSpec, list_cameras  # noqa: E402


#: Score below which `find_cube` reports nothing rather than its best guess.
#: It scores by how much darker the surround is, which assumes the object sits
#: on the dark box; slide it onto bare table and the score collapses until some
#: reflection outscores it. Without a floor the answer is then the reflection,
#: and the caller cannot tell. Over 43 genuine detections the worst scored 53.9.
#:
#: Prefer `quest_teleop2.vision.detect.find_object`, which compares against a
#: picture of the empty table and so depends on neither the object's colour nor
#: what it is standing on.
FIND_CUBE_MIN_SCORE = 40.0


#: Smallest fraction of its ROTATED bounding box a candidate may fill.
#:
#: Measured against `cv2.minAreaRect`, not the axis-aligned box. That is the
#: whole point: a square fills 1/(|cos|+|sin|)^2 of its AXIS-ALIGNED box, which
#: falls to 0.50 at 45 degrees, so the old test threw away any cube turned more
#: than ~27 degrees however white and however plainly in view. Against the
#: rotated box a square fills 1.0 at every angle, so this now measures what it
#: was always meant to measure -- raggedness, not orientation.
#:
#: 0.55 still passes a circle (pi/4 = 0.785) and a hexagon, so the cone and
#: cylinder are not excluded by it either.
MIN_FILL = 0.55

#: Long/short side of the rotated box. Also rotation-invariant, unlike w/h of
#: the axis-aligned box, which for a rotated rectangle swings between its true
#: aspect and 1.0 as it turns.
MAX_ASPECT = 3.0


def candidates(frame: np.ndarray) -> list[dict]:
    """Every blob the mask found, with the measurements each filter tests.

    Returned whether accepted or rejected, with `reason` naming the first
    filter that killed it and empty when it survived. `find_cube` reports one
    answer; this says why the others were not it, which is the only way to tell
    "the object is not visible" from "the object was filtered out".

    Also reports `yaw`, the rotated box's angle, which is what a reorientation
    task needs and what the axis-aligned box could never provide.
    """
    import cv2  # noqa: PLC0415

    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 0, 170), (180, 60, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    H, W = grey.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out: list[dict] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        (_cx, _cy), (rw, rh), angle = cv2.minAreaRect(contour)
        long_side, short_side = max(rw, rh), max(min(rw, rh), 1e-6)
        fill = area / max(rw * rh, 1e-6)
        aspect = long_side / short_side

        pad = 12
        y0, y1 = max(0, y - pad), min(H, y + h + pad)
        x0, x1 = max(0, x - pad), min(W, x + w + pad)
        ring = grey[y0:y1, x0:x1].astype(np.float32)
        inner = grey[y:y + h, x:x + w].astype(np.float32)
        surround = (ring.sum() - inner.sum()) / max(ring.size - inner.size, 1)
        score = float(inner.mean() - surround) if inner.size else 0.0

        reason = ""
        if not 200 < area < 8000:
            reason = f"area {area:.0f} outside 200..8000"
        elif aspect > MAX_ASPECT:
            reason = f"aspect {aspect:.2f} above {MAX_ASPECT}"
        # Measured on this scene: the cube sat close enough to the bright wall
        # strip along the top of the frame to merge with it into one component
        # anchored at (0,0). A cube touching the frame edge is a framing
        # problem regardless.
        elif x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            reason = "touches the frame border"
        elif fill < MIN_FILL:
            reason = f"fill {fill:.2f} below {MIN_FILL} (ragged, not merely turned)"

        out.append({"box": (int(x), int(y), int(w), int(h)), "area": int(area),
                    "aspect": float(aspect), "fill": float(fill),
                    "yaw": float(angle), "score": score, "reason": reason})
    return out


def find_cube(frame: np.ndarray,
              min_score: float = FIND_CUBE_MIN_SCORE
              ) -> tuple[tuple[int, int, int, int] | None, float]:
    """Bright, low-saturation blob on a dark surround. Returns (x,y,w,h), score.

    Returns ``(None, score)`` when the best candidate is below `min_score`, so
    a caller can tell "I do not see it" from "here is my best guess". Pass
    ``min_score=0`` for the old always-answer behaviour.

    Use `candidates()` to see what was rejected and why.
    """
    best, best_score = None, -1.0
    for c in candidates(frame):
        if c["reason"]:
            continue
        if c["score"] > best_score:
            best, best_score = c["box"], c["score"]
    if best is not None and best_score < min_score:
        return None, best_score
    return best, best_score


def run_compare(cam: Camera, n: int) -> int:
    """What each detector says about the same frames, in table millimetres.

    The reach test cannot answer this: it sends the arm to whatever the
    detector reported and then measures how well the arm got there, so any
    detector scores the same. The only way to see the disagreement is to ask
    both about one frame.

    It matters because the two segment different things -- the brightness path
    takes only the pale object, background subtraction takes the object and
    whatever travels underneath it -- so their centroids differ by however far
    that shifts the balance. A dataset recorded under one and approached under
    the other is offset by exactly this, and approach_offset is what absorbs it.
    """
    import numpy as np  # noqa: PLC0415

    from quest_teleop2.vision.detect import BackgroundModel, find_object
    from quest_teleop2.vision.table import TableMap

    try:
        table = TableMap.load()
    except (FileNotFoundError, ValueError) as exc:
        print(f"\033[31m{exc}\033[0m")
        return 1
    try:
        background = BackgroundModel.load()
    except (FileNotFoundError, ValueError) as exc:
        print(f"\033[31mNo background to compare against: {exc}\033[0m")
        return 1

    print(f"\n\033[1mDetector comparison\033[0m   {n} frames, object still\n")
    print(f"  {'':4} {'brightness':>22}  {'background':>22}   {'difference':>16}")
    rows = []
    while len(rows) < n:
        frame, _age = cam.latest()
        if frame is None:
            continue
        bright_box, bright_score = find_cube(frame)
        bg_box, bg_score = find_object(frame, background)
        if bright_box is None or bg_box is None:
            which = "brightness" if bright_box is None else "background"
            print(f"  {len(rows):4d} {which} saw nothing "
                  f"(scores {bright_score:.0f} / {bg_score:.0f})")
            rows.append(None)
            continue

        def centre(box):
            x, y, w, h = box
            return table.to_table((x + w / 2.0, y + h / 2.0))

        a, b = centre(bright_box), centre(bg_box)
        rows.append(b - a)
        print(f"  {len(rows) - 1:4d} ({a[0]:+.3f}, {a[1]:+.3f}) s{bright_score:5.1f}"
              f"  ({b[0]:+.3f}, {b[1]:+.3f}) s{bg_score:5.1f}"
              f"   ({(b - a)[0] * 1000:+5.1f}, {(b - a)[1] * 1000:+5.1f}) mm")

    good = [r for r in rows if r is not None]
    if not good:
        print("\n  \033[31mNo frame where both detectors saw the object.\033[0m")
        return 1
    D = np.stack(good)
    print(f"\n  \033[1mbackground minus brightness, over {len(good)} frames\033[0m")
    print(f"    x  mean {D[:, 0].mean() * 1000:+6.1f} mm   std {D[:, 0].std() * 1000:5.1f}")
    print(f"    y  mean {D[:, 1].mean() * 1000:+6.1f} mm   std {D[:, 1].std() * 1000:5.1f}")
    shift = np.linalg.norm(D.mean(axis=0))
    print("\n  A dataset recorded under the brightness detector is approached")
    print(f"  {shift * 1000:.1f} mm off if you now use background subtraction.")
    if shift < 0.005:
        print("  \033[32mUnder 5 mm: the two agree, keep the demonstrations.\033[0m")
    elif shift < 0.020:
        print("  \033[33mAdd this to approach_offset and the old demos stay usable --\033[0m")
        print("  \033[33mit is well inside the +-39 mm the policy trains to absorb.\033[0m")
        print(f"    --offset-x {table.approach_offset[0] - D[:, 0].mean():+.3f} "
              f"--offset-y {table.approach_offset[1] - D[:, 1].mean():+.3f} --save-offset")
    else:
        print("  \033[31mOver 20 mm. Correctable by offset, but large enough that it\033[0m")
        print("  \033[31mis worth understanding before trusting it.\033[0m")
    return 0


def run_diagnose(cam: Camera, scale: float) -> int:
    """Draw every candidate blob and say which filter rejected it.

    `--live` answers "is it found"; this answers "why not". A cube can be the
    whitest thing in frame, plainly on its dark box, and still be dropped by a
    shape filter -- which from the outside looks identical to not seeing it.
    """
    import cv2  # noqa: PLC0415

    print("\n\033[1mDetector diagnosis\033[0m   move the cube; q to quit")
    print("  Green = accepted.  Red = rejected, with the filter that killed it.\n")
    cv2.namedWindow("diagnose", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("diagnose", int(640 * scale), int(480 * scale))
    last = ""
    try:
        while True:
            frame, _age = cam.latest()
            if frame is None:
                continue
            found = candidates(frame)
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if scale != 1.0:
                img = cv2.resize(img, None, fx=scale, fy=scale)

            accepted = [c for c in found if not c["reason"]]
            for c in found:
                x, y, w, h = c["box"]
                ok = not c["reason"]
                colour = (80, 255, 80) if ok else (80, 80, 255)
                s = scale
                cv2.rectangle(img, (int(x * s), int(y * s)),
                              (int((x + w) * s), int((y + h) * s)), colour, 2)
                label = f"{c['score']:.0f}" if ok else c["reason"].split(" (")[0]
                cv2.putText(img, label, (int(x * s), max(12, int(y * s) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42 * scale, colour, 1,
                            cv2.LINE_AA)

            head = (f"{len(found)} blob(s), {len(accepted)} accepted"
                    if found else "no blobs in the mask at all")
            cv2.rectangle(img, (0, 0), (int(460 * scale), int(28 * scale)), (0, 0, 0), -1)
            cv2.putText(img, head, (8, int(20 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * scale, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.imshow("diagnose", img)

            # Print only on change, or the terminal is unreadable at 30 fps.
            summary = " | ".join(
                f"{c['area']}px fill {c['fill']:.2f} " + (c["reason"] or f"OK score {c['score']:.0f}")
                for c in sorted(found, key=lambda c: -c["area"])[:4]
            )
            if summary != last:
                print(f"  {summary or 'nothing in the mask'}")
                last = summary
            if (cv2.waitKey(20) & 0xFF) in (ord("q"), 27):
                break
    finally:
        cv2.destroyAllWindows()
    return 0


def run_live(cam: Camera, scale: float) -> int:
    import cv2  # noqa: PLC0415

    print("\n\033[1mLive cube readout\033[0m   move the cube; q to quit")
    print("  A dumb detector, not the policy. It answers: is the cube")
    print("  unambiguously visible from this viewpoint?\n")
    cv2.namedWindow("scene", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("scene", int(640 * scale), int(480 * scale))
    misses = seen = 0
    try:
        while True:
            frame, age = cam.latest()
            if frame is None:
                continue
            box, score = find_cube(frame)
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if scale != 1.0:
                img = cv2.resize(img, None, fx=scale, fy=scale)
            seen += 1
            if box is None:
                misses += 1
                text, colour = "cube NOT FOUND", (80, 80, 255)
            else:
                x, y, w, h = box
                cx, cy = x + w // 2, y + h // 2
                s = scale
                cv2.rectangle(img, (int(x * s), int(y * s)),
                              (int((x + w) * s), int((y + h) * s)), (80, 255, 80), 2)
                cv2.drawMarker(img, (int(cx * s), int(cy * s)), (80, 255, 80),
                               cv2.MARKER_CROSS, 18, 2)
                text = f"cube ({cx:3d},{cy:3d})  {w}x{h}px  contrast {score:5.1f}"
                colour = (80, 255, 80)
            cv2.rectangle(img, (0, 0), (int(420 * scale), int(52 * scale)), (0, 0, 0), -1)
            cv2.putText(img, text, (8, int(22 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * scale, colour, 1, cv2.LINE_AA)
            cv2.putText(img, f"found {100 * (seen - misses) / max(seen, 1):5.1f}% of frames",
                        (8, int(44 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45 * scale, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.imshow("scene", img)
            if (cv2.waitKey(20) & 0xFF) in (ord("q"), 27):
                break
    finally:
        cv2.destroyAllWindows()
    rate = 100 * (seen - misses) / max(seen, 1)
    print(f"\n  found the cube in {rate:.1f}% of frames")
    if rate < 90:
        print("  \033[33mPatchy.\033[0m A threshold is not the policy, so this is not fatal -- but")
        print("  poor contrast here usually means poor contrast for the backbone too.")
    return 0


def run_separability(cam: Camera, n_starts: int, per_pos: int, null: bool = False,
                     tag: str = "") -> int:
    """Embed each position through ACT's own backbone and test cluster separation."""
    import cv2  # noqa: PLC0415
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The same backbone and weights ACT fine-tunes from. Testing with anything
    # else would answer a different question.
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net = net.to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    feats, labels, shots = [], [], []
    label = "NULL CONTROL" if null else "Separability"
    print(f"\n\033[1m{label}\033[0m   {n_starts} sets x {per_pos} frames")
    if null:
        # The instructions are mutually exclusive, so print exactly one. Printing
        # the generic "move it slightly" line above a "change NOTHING" banner
        # produced a null run in which the cube was moved -- which then read as
        # 9.76x separability and appeared to invalidate a perfectly good test.
        print("  \033[1mDo NOT move anything.\033[0m Not the cube, not the box, not the")
        print("  camera. Leave the scene alone and press ENTER four times.")
        print("  Whatever separability comes out is drift, and sets the floor the")
        print("  real test must clear.\n")
    else:
        print("  Put the cube on a start, then press ENTER. Move it slightly between")
        print("  frames the way it really varies -- a test on one rigid placement")
        print("  measures nothing.\n")
    for pos in range(n_starts):
        if null:
            input(f"  capture set {pos} (change NOTHING), then ENTER... ")
        else:
            input(f"  cube on start s{pos}, then ENTER... ")
        for k in range(per_pos):
            time.sleep(0.35)
            frame, _ = cam.latest()
            shots.append((pos, frame.copy()))
            x = torch.from_numpy(frame).to(device).permute(2, 0, 1)[None].float().div_(255)
            x = (x - mean) / std
            with torch.no_grad():
                f = net(x)[0].cpu().numpy()
            feats.append(f / np.linalg.norm(f))
            labels.append(pos)
        print(f"    captured {per_pos}")

    F = np.stack(feats)
    y = np.array(labels)
    D = 1.0 - F @ F.T          # cosine distance, features are unit norm
    np.fill_diagonal(D, np.inf)

    nn = y[np.argmin(D, axis=1)]
    acc = float((nn == y).mean())
    within = float(np.mean([D[i, (y == y[i]) & (np.arange(len(y)) != i)].mean() for i in range(len(y))]))
    between = float(np.mean([D[i, y != y[i]].mean() for i in range(len(y))]))

    print(f"\n  leave-one-out nearest neighbour: \033[1m{acc:.0%}\033[0m "
          f"(chance {1 / n_starts:.0%})")
    print(f"  mean distance within a position : {within:.4f}")
    print(f"  mean distance between positions : {between:.4f}")
    print(f"  separation ratio between/within : \033[1m{between / max(within, 1e-9):.2f}x\033[0m")

    print("\n  per-position accuracy:")
    for pos in range(n_starts):
        m = y == pos
        print(f"    s{pos}  {float((nn[m] == pos).mean()):.0%}")

    _attribution(shots, n_starts)

    name = ("null" if null else "sep") + (f"_{tag}" if tag else "")
    out = Path("plots/cameras") / name
    out.mkdir(parents=True, exist_ok=True)
    for pos in range(n_starts):
        img = next(f for p, f in shots if p == pos)
        cv2.imwrite(str(out / f"s{pos}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"\n  one frame per position written to {out}/")

    import json  # noqa: PLC0415

    summary = {"tag": tag or "untagged", "null": null, "accuracy": acc,
               "within": within, "between": between,
               "ratio": between / max(within, 1e-9), "n_starts": n_starts,
               "per_position": per_pos}
    (out / "result.json").write_text(json.dumps(summary, indent=2))

    # Rank every placement measured so far, so the comparison is one glance.
    runs = []
    for res in sorted(Path("plots/cameras").glob("sep*/result.json")):
        try:
            runs.append(json.loads(res.read_text()))
        except Exception:  # noqa: BLE001
            continue
    if len(runs) > 1:
        print("\n  \033[1mcamera placements measured so far\033[0m")
        print("    tag                acc    between    ratio")
        for r in sorted(runs, key=lambda r: -r["between"]):
            star = "  <-- best separation" if r is max(runs, key=lambda x: x["between"]) else ""
            print(f"    {r['tag']:16s} {r['accuracy']:5.0%}   {r['between']:.4f}   "
                  f"{r['ratio']:5.2f}x{star}")
        print("\n    Compare BETWEEN, not just the ratio: the ratio also moves when your")
        print("    placement jitter changes, which is not a property of the camera.")

    print()
    if acc >= 0.95 and between / max(within, 1e-9) > 1.5:
        print("  \033[32mClearly separable.\033[0m The information is in the image and reaches")
        print("  the backbone untrained. Viewpoint is not your problem.")
    elif acc >= 0.8:
        print("  \033[33mSeparable but not cleanly.\033[0m Training will likely still work -- the")
        print("  backbone gets fine-tuned -- but the margin is thinner than it should be.")
        print("  Check the per-position numbers above for which pair is being confused.")
    else:
        print("  \033[31mNot separable.\033[0m An untrained backbone cannot tell these apart, so")
        print("  the policy starts from nothing. Move the camera or the positions before")
        print("  recording: no amount of training fixes a viewpoint that lacks the signal.")
    return 0


def _attribution(shots, n_starts: int) -> None:
    """Which parts of the frame drive the separation -- the object, or the scene?

    A whole-frame embedding proves the scenes differ, not that the CUBE is what
    differs. This recomputes on ResNet18's 15x20 spatial map (the same grid ACT
    tokenises) and reports where the between-position variance concentrates. If
    it sits on the marks, the signal is the thing you moved. If it is spread
    over the edges, something else drifted.
    """
    import cv2  # noqa: PLC0415
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    body = torch.nn.Sequential(*(list(net.children())[:-2])).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    per_pos = []
    for pos in range(n_starts):
        frames = [f for p, f in shots if p == pos]
        acc = None
        for frame in frames:
            x = torch.from_numpy(frame).to(device).permute(2, 0, 1)[None].float().div_(255)
            with torch.no_grad():
                f = body((x - mean) / std)[0]
            f = torch.nn.functional.normalize(f, dim=0).cpu().numpy()
            acc = f if acc is None else acc + f
        per_pos.append(acc / len(frames))

    stack = np.stack(per_pos)
    var = ((stack - stack.mean(0, keepdims=True)) ** 2).sum(1).mean(0)
    var = var / max(var.max(), 1e-12)
    h, w = var.shape

    print(f"\n  where the positions differ ({h}x{w} tokens, 32x32 px each):")
    for r in range(h):
        print("    " + "".join(" .:-=+*#@"[min(8, int(v * 8.999))] for v in var[r]))

    edge = np.zeros_like(var, dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    edge_share = float(var[edge].sum() / max(var.sum(), 1e-12))
    print(f"\n  share of the difference sitting on the frame border: {edge_share:.0%}")
    if edge_share > 0.25:
        print("  \033[33mA lot of it is at the edges.\033[0m That is where background and the arm")
        print("  live, so some of this separability may not be the object you moved.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true",
                        help="live cube readout, for aiming")
    parser.add_argument("--compare", type=int, nargs="?", const=20, metavar="N",
                        help="report what BOTH detectors say about the same frames, "
                             "in table coordinates. The difference is what changes "
                             "when you switch, and what approach_offset must absorb")
    parser.add_argument("--diagnose", action="store_true",
                        help="draw every candidate blob and name the filter that "
                             "rejected it. Use when the object is plainly visible "
                             "and still not found")
    parser.add_argument("--separability", action="store_true",
                        help="the clustering test")
    parser.add_argument("--null", action="store_true",
                        help="control: capture the SAME scene N times as if they were "
                             "different positions. Separability here is drift, not signal, "
                             "and sets the floor the real test must clear.")
    parser.add_argument("--serial", default="",
                        help="one camera only; default is every camera found")
    parser.add_argument("--starts", type=int, default=4,
                        help="how many cube positions to test")
    parser.add_argument("--per-position", type=int, default=8,
                        help="frames captured per position")
    parser.add_argument("--scale", type=float, default=2.0,
                        help="display magnification")
    parser.add_argument("--tag", default="",
                        help="name this camera position, e.g. --tag front. Results and frames "
                             "are kept per tag so two placements can be compared numerically "
                             "instead of argued about.")
    args = parser.parse_args()

    if not (args.live or args.diagnose or args.separability or args.null
            or args.compare):
        parser.error("choose --live, --diagnose, --separability or --null")

    found = list_cameras()
    if not found:
        print("\033[31mNo Logitech Brio found.\033[0m")
        return 1
    serial = args.serial or found[0][0]

    cam = Camera(CameraSpec(name="static", serial=serial))
    # lock=False keeps whatever focus/exposure was tuned by hand.
    cam.start(lock=False)
    try:
        time.sleep(1.0)
        if args.compare:
            return run_compare(cam, args.compare)
        if args.diagnose:
            return run_diagnose(cam, args.scale)
        if args.live:
            return run_live(cam, args.scale)
        return run_separability(cam, args.starts, args.per_position,
                                null=args.null, tag=args.tag)
    finally:
        cam.stop()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Live camera view for aiming a mount and tuning focus and exposure.

    python scripts/camera/camera_live.py                      # every camera found
    python scripts/camera/camera_live.py --tile               # both in one window
    python scripts/camera/camera_live.py --tile --focus 0,120 # start at a known focus
    python scripts/camera/camera_live.py --grid 4             # overlay a 4x4 guide

The readout shows capture rate, frame age, and sharpness (variance of the
Laplacian -- higher is sharper, and more reliable than judging focus by eye at
640x480).

Control locks are re-applied at every start, so a value tuned last session
resets to the spec default. Pass --focus/--exposure/--wb to start from a known
point: one number for every camera, or one per camera in listing order. On quit
the script prints the command to reopen where you left off.

Keys
    f / F     focus       -/+ (step 5, 0 = infinity)
    e / E     exposure    -/+ (units of 100 us; lower is darker and faster)
    w / W     white bal   -/+ Kelvin
    TAB       switch which camera the keys drive
    a         toggle autofocus, to find a distance and read it off
    i         toggle the info panel
    g         toggle the alignment grid
    1 2 3 4   stamp the cube's current position as taped position 0..3. Shows
              the closest pair in pixels: two marks close together in the IMAGE
              are two the policy will confuse.
    c         clear the stamps
    SPACE     save a snapshot to plots/cameras/
    q / ESC   quit and print the settings
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quest_teleop2.vision.camera import Camera, CameraSpec, list_cameras  # noqa: E402
from quest_teleop2.vision.display import (  # noqa: E402
    GAP,
    MARGIN,
    closest_pair,
    overlay,
    sharpness,
    tile,
)

from scripts.camera.check_scene import find_cube  # noqa: E402

TILE_WIN = "cameras"

STEP = {"focus": 5, "exposure": 25, "white_balance": 250}
LIMITS = {"focus": (0, 255), "exposure": (3, 2047), "white_balance": (2000, 7500)}
CTRL = {"focus": "focus_absolute", "exposure": "exposure_time_absolute",
        "white_balance": "white_balance_temperature"}


def _set(dev: str, ctrl: str, value: int) -> None:
    subprocess.run(["v4l2-ctl", "-d", dev, f"--set-ctrl={ctrl}={value}"],
                   capture_output=True, timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serial", default="", help="one camera; default is every Brio found")
    parser.add_argument("--grid", type=int, default=0, help="overlay an NxN alignment grid")
    parser.add_argument("--out", type=Path, default=Path("plots/cameras"),
                        help="directory for snapshots")
    parser.add_argument("--focus", default=None,
                        help="initial focus_absolute, 0..255 (0 = infinity). One value for "
                             "all cameras, or comma-separated per camera in listing order.")
    parser.add_argument("--exposure", default=None,
                        help="initial exposure_time_absolute, units of 100 us.")
    parser.add_argument("--wb", default=None, help="initial white_balance_temperature, Kelvin.")
    parser.add_argument("--tile", action="store_true",
                        help="both cameras in one window, side by side, instead of a "
                             "window each. TAB moves the keys between them.")
    parser.add_argument("--info", default=None, action=argparse.BooleanOptionalAction,
                        help="start with the info panel shown. Default: shown for one "
                             "window per camera, HIDDEN under --tile, where the panel "
                             "covers the part of the frame you are aiming. 'i' toggles.")
    parser.add_argument("--scale", type=float, default=None,
                        help="display magnification. Default 2.0 for one window per "
                             "camera, 1.5 under --tile (two 640x480 panels -> 1920x720). "
                             "Display only -- capture and recording stay native.")
    args = parser.parse_args()

    import cv2  # noqa: PLC0415

    found = list_cameras()
    if not found:
        print("\033[31mNo Logitech Brio found.\033[0m")
        return 1
    serials = [args.serial] if args.serial else [s for s, _ in found]

    tiled = args.tile and len(serials) > 1
    if args.scale is None:
        args.scale = 1.5 if tiled else 2.0
    show_info = args.info if args.info is not None else not tiled

    specs = [CameraSpec(name=s, serial=s) for s in serials]

    def apply(flag: str | None, field: str) -> None:
        """One value for every camera, or one per camera in listing order."""
        if flag is None:
            return
        vals = [int(v) for v in str(flag).split(",")]
        if len(vals) == 1:
            vals *= len(specs)
        if len(vals) != len(specs):
            raise SystemExit(f"--{field}: got {len(vals)} values for {len(specs)} cameras")
        lo, hi = LIMITS[field]
        for spec, v in zip(specs, vals):
            setattr(spec, field, int(np.clip(v, lo, hi)))

    apply(args.focus, "focus")
    apply(args.exposure, "exposure")
    apply(args.wb, "white_balance")

    cams: list[Camera] = []
    try:
        for spec in specs:
            cam = Camera(spec)
            cam.start(lock=True)
            cams.append(cam)
    except Exception as exc:
        for cam in cams:
            cam.stop()
        print(f"\033[31m{exc}\033[0m")
        return 1

    print(f"\n\033[1mLive: {', '.join(serials)}\033[0m")
    print(__doc__.split("Keys\n")[1])
    print("  Aim at the workspace, then raise 'sharp' with f/F. Bright green is sharpest so far.")
    if not show_info:
        print("  \033[1mInfo panel hidden\033[0m -- press 'i' to show fps, sharpness and focus.")
    print()

    # WINDOW_NORMAL so a window can also be dragged to any size afterwards.
    if tiled:
        cv2.namedWindow(TILE_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(TILE_WIN,
                         int(sum(c.spec.width for c in cams) * args.scale) + GAP * (len(cams) - 1),
                         int(max(c.spec.height for c in cams) * args.scale))
        print("  TAB moves the keys between panels; the driven one is outlined.\n")
    else:
        for cam in cams:
            cv2.namedWindow(f"cam {cam.spec.serial}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"cam {cam.spec.serial}",
                             int(cam.spec.width * args.scale), int(cam.spec.height * args.scale))

    auto = {s: False for s in serials}
    best = {s: 0.0 for s in serials}
    marks: dict[str, dict[int, tuple[int, int]]] = {s: {} for s in serials}
    grid = args.grid
    active = 0  # which camera the keys apply to

    try:
        while True:
            panels: list[np.ndarray | None] = [None] * len(cams)
            for i, cam in enumerate(cams):
                frame, age = cam.latest()
                if frame is None:
                    continue
                held = marks[cam.spec.serial]
                lines: list[tuple[str, tuple[int, int, int]]] = []
                if show_info:
                    sharp = sharpness(frame)
                    best[cam.spec.serial] = max(best[cam.spec.serial], sharp)
                    peak = best[cam.spec.serial]
                    # Green once within 5% of the sharpest seen -- an absolute
                    # threshold would be meaningless, the scale depends on the scene.
                    colour = (80, 255, 80) if sharp >= 0.95 * peak else (80, 200, 255)
                    sel = " [keys]" if i == active and len(cams) > 1 else ""
                    lines = [
                        (f"{cam.spec.serial}{sel}  {cam.measured_fps:4.1f} fps  "
                         f"age {age * 1000:4.1f} ms", (255, 255, 255)),
                        (f"sharp {sharp:7.1f}   best {peak:7.1f}", colour),
                        (f"focus {cam.spec.focus:3d}"
                         f"{'  AUTO' if auto[cam.spec.serial] else ''}   "
                         f"exp {cam.spec.exposure:4d}   wb {cam.spec.white_balance}",
                         (255, 255, 255)),
                    ]
                    lines += _mark_lines(held, cam.spec)
                panels[i] = overlay(
                    frame, lines, scale=args.scale, grid=grid, marks=held,
                    # Outline the panel the keys drive -- with two dark scenes
                    # the "[keys]" tag alone is easy to lose, and it is gone
                    # entirely once the info panel is hidden.
                    border=(80, 255, 80) if tiled and i == active else None,
                )
                if not tiled:
                    cv2.imshow(f"cam {cam.spec.serial}", panels[i])

            if tiled:
                composite = tile(panels)
                if composite is not None:
                    cv2.imshow(TILE_WIN, composite)

            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("g"):
                grid = 0 if grid else 3
            if key == ord("i"):
                show_info = not show_info
            if key == ord("\t") and len(cams) > 1:
                active = (active + 1) % len(cams)
            if key in (ord("1"), ord("2"), ord("3"), ord("4")):
                idx = key - ord("1")
                for c in cams:
                    f, _ = c.latest()
                    if f is None:
                        continue
                    box, _score = find_cube(f)
                    if box is None:
                        print(f"  \033[33mposition {idx}: no cube found on "
                              f"{c.spec.serial} -- not stamped\033[0m")
                        continue
                    x, y, bw, bh = box
                    marks[c.spec.serial][idx] = (x + bw // 2, y + bh // 2)
                    print(f"  position {idx} on {c.spec.serial}: "
                          f"({x + bw // 2}, {y + bh // 2}) px")
            if key == ord("c"):
                for c in cams:
                    marks[c.spec.serial].clear()
                print("  stamps cleared")

            cam = cams[active]
            spec, dev = cam.spec, str(cam.spec.device())

            def bump(field: str, sign: int, spec: CameraSpec = spec, dev: str = dev) -> None:
                lo, hi = LIMITS[field]
                value = int(np.clip(getattr(spec, field) + sign * STEP[field], lo, hi))
                setattr(spec, field, value)
                _set(dev, CTRL[field], value)
                best[spec.serial] = 0.0  # the scene changed; re-baseline 'best'

            if key == ord("f"):
                bump("focus", -1)
            elif key == ord("F"):
                bump("focus", +1)
            elif key == ord("e"):
                bump("exposure", -1)
            elif key == ord("E"):
                bump("exposure", +1)
            elif key == ord("w"):
                bump("white_balance", -1)
            elif key == ord("W"):
                bump("white_balance", +1)
            elif key == ord("a"):
                auto[spec.serial] = not auto[spec.serial]
                _set(dev, "focus_automatic_continuous", int(auto[spec.serial]))
                if not auto[spec.serial]:
                    # Read back where autofocus settled and keep it as the manual
                    # value -- the usual way to find a good focus is to let the
                    # camera do it once, then pin it.
                    out = subprocess.run(["v4l2-ctl", "-d", dev, "--get-ctrl=focus_absolute"],
                                         capture_output=True, text=True, timeout=5).stdout
                    if ":" in out:
                        spec.focus = int(out.split(":")[1].strip().split()[0])
                    _set(dev, "focus_absolute", spec.focus)
            elif key == ord(" "):
                args.out.mkdir(parents=True, exist_ok=True)
                for c in cams:
                    f, _ = c.latest()
                    if f is not None:
                        p = args.out / f"{c.spec.serial}.jpg"
                        cv2.imwrite(str(p), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                        print(f"  saved {p}")
    finally:
        cv2.destroyAllWindows()
        for cam in cams:
            cam.stop()

    print("\n\033[1mSettings — paste into config/cameras.yaml\033[0m")
    for cam in cams:
        s = cam.spec
        print(f'  - {{name: <wrist|static>, serial: "{s.serial}", focus: {s.focus}, '
              f"exposure: {s.exposure}, white_balance: {s.white_balance}}}")
    print("\n  reopen with these values:")
    print(f"    python scripts/camera/camera_live.py{' --tile' if tiled else ''}"
          f" --focus {','.join(str(c.spec.focus) for c in cams)}"
          f" --exposure {','.join(str(c.spec.exposure) for c in cams)}"
          f" --wb {','.join(str(c.spec.white_balance) for c in cams)}")
    for cam in cams:
        held = marks[cam.spec.serial]
        if len(held) > 1:
            gap, pair = closest_pair(held)
            print(f"\n  {cam.spec.serial}: marks at "
                  + ", ".join(f"{i}=({x},{y})" for i, (x, y) in sorted(held.items()))
                  + f"\n    closest pair {pair} = {gap:.1f} px")
    print("\n  Exposure also caps frame rate: 250 = 25 ms = 40 fps ceiling.")
    return 0


def _mark_lines(held: dict[int, tuple[int, int]],
                spec: CameraSpec) -> list[tuple[str, tuple[int, int, int]]]:
    """One readout line summarising the stamped positions, if there are any."""
    if not held:
        return []
    outside = [i for i, (x, y) in held.items()
               if not (MARGIN <= x <= spec.width - MARGIN
                       and MARGIN <= y <= spec.height - MARGIN)]
    if outside:
        return [(f"marks {sorted(held)}  OUTSIDE MARGIN: {outside}", (60, 60, 255))]
    if len(held) > 1:
        gap, pair = closest_pair(held)
        return [(f"marks {sorted(held)}  closest {pair} {gap:5.1f} px",
                 (80, 255, 80) if gap >= 60 else (80, 200, 255))]
    return [(f"marks {sorted(held)}", (255, 255, 255))]


if __name__ == "__main__":
    raise SystemExit(main())

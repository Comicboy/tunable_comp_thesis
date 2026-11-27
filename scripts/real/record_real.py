#!/usr/bin/env python3
"""Record lift demonstrations on the real UR5e + Wuji Hand 2, with our fingertip admittance.

    python scripts/real/record_real.py --stiffness 150 500                      # practice, K hidden
    python scripts/real/record_real.py --stiffness 150 500 --show-k             # practice, K shown
    python scripts/real/record_real.py --stiffness 150 500 --record             # training demos
    python scripts/real/record_real.py --stiffness 150 500 --record --split val # validation demos

The real-robot counterpart of the simulation's scripts/teleop/teleop_lift.py, built on the
supervisor's quest_teleop2 (used unchanged, third_party/quest_teleop_2). His TeleopSession runs
the 125 Hz loop, the wrist retargeting, the safety limiter and the recording; ours plug in:
  RealTeleopSession   the episode flow (tunable_comp_real/session.py)
  TactileFingertipAdmittance   glove -> fingertip forces -> our admittance, every tick
  MeasuredWujiHand2   his hand driver, recording measured (never commanded) angles
  StiffnessRecorders  one file per K, delete last, the hidden weighted K draw (episodes.py)

Episode (agreed 2026-11-26, notes_real_setup.md section 8):
  1. home: the fingers park, the arm goes to home_q, the glove relearns its rest (touch nothing
     for 2 s), a K is drawn and hidden;
  2. place the cube on its marked spot;
  3. left pinch + release: recording starts;
  4. grasp and lift: once the cube is 15 cm up for 3 s the demo is saved by itself;
  5. put the cube down, let go, press h for the next episode.

Keys: h home / next episode, r discard + home, d delete the last saved demo, a re-anchor,
      g relearn the glove rest, q quit. Left-hand pinch freezes the arm while you reposition.
Time limit and a lost right hand discard the episode and go home.

Files: one per K, <demo dir>/lift_K<K>.hdf5 (the supervisor's HDF5 layout plus our keys), default
logs/real/demos/ (train) or logs/real/demos_val/ (val); recording stops being offered once every
K has --demos-per-k demos. Run scripts/real/test_session.py first after any change.

Start-up order: cameras, Quest server, arm, hand, then the glove. The glove must connect after the
hand: the hand SDK, starting up, breaks a glove connection that already exists in the same process
(found 2026-11-27). The script then waits until glove frames arrive, and stops with a message if not.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from quest_teleop2.config import REPO_ROOT as QT2_ROOT
from quest_teleop2.quest.quest_input import DualHandInput
from quest_teleop2.quest.webxr_server import WebXRServer
from quest_teleop2.robot.safety import SafetyAbort
from quest_teleop2.robot.ur5e_arm import UR5eArm
from quest_teleop2.tactile.contact import FingerRegions
from quest_teleop2.tactile.glove import GloveClient
from quest_teleop2.teleop.hand_retarget import HandRetargeter, urdf_limits
from tunable_comp_real.admittance import AdmittanceParams, DynamicAdmittance
from tunable_comp_real.episodes import (FORMAT, DeletableRecorder, StiffnessRecorders, SuccessDetector,
                                        SuccessParams, demo_path, k_name)
from tunable_comp_real.hand_driver import MeasuredWujiHand2
from tunable_comp_real.hand_model import HandModel
from tunable_comp_real.session import RealTeleopSession
from tunable_comp_real.tactile_admittance import GloveCalibration, TactileFingertipAdmittance
from tunable_comp_real.wrist_pivot import pivot_offset


def _load_supervisor_common():
    """The supervisor's scripts/_common.py (argument helpers, URDF and retarget-config paths,
    camera rig, banner), loaded from his repo by file path: it is outside his package, and a plain
    `import scripts._common` could resolve to our own scripts/ folder."""
    path = QT2_ROOT / "scripts" / "_common.py"
    spec = importlib.util.spec_from_file_location("quest_teleop2_scripts_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = _load_supervisor_common()

RED, YELLOW, GREEN, BOLD, RESET = "\033[31m", "\033[33m", "\033[32m", "\033[1m", "\033[0m"
TASK = "real_lift_fixed_spot"
DEMO_DIRS = {"train": Path("logs/real/demos"), "val": Path("logs/real/demos_val")}
#: Seconds to park the fingers at the neutral pose (as his record_grasps.PARK_RAMP_S).
PARK_S = 1.0

CONTROLS = """\
  Right hand   wrist drives the robot hand's wrist (relative to where you were when the clutch
               engaged), fingers drive the fingers
  Left hand    pinch + release at home: start the episode
               pinch during an episode: freeze the arm while you reposition, release to go on
  Keyboard     h  home (next episode)      r  discard + home      d  delete the last saved demo
               a  re-anchor                g  relearn the glove rest      q  quit
  An episode ends by itself: saved once the cube is 15 cm up for 3 s, discarded on the time
  limit or when the right hand is lost. Safety is the physical e-stop; the hand has none."""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("episodes")
    g.add_argument("--stiffness", type=float, nargs="+", required=True, metavar="K",
                   help="fingertip stiffnesses [N/m]; one is drawn per episode, hidden (inf = admittance off)")
    g.add_argument("--record", action="store_true", help="save successful demos (otherwise practice)")
    g.add_argument("--split", choices=("train", "val"), default="train")
    g.add_argument("--demo-dir", type=Path, default=None,
                   help="folder of the files lift_K<K>.hdf5 (default logs/real/demos[_val])")
    g.add_argument("--demos-per-k", type=int, default=None,
                   help="demos wanted per K (default 20 train, 10 val); the draw favours the K that lack demos")
    g.add_argument("--show-k", action="store_true", help="print each episode's K (practice only)")
    g.add_argument("--object", default="cube", help="stored with every demo")
    g.add_argument("--cube-spot", default="A", help="label of the marked cube spot, stored with every demo")
    g.add_argument("--seed", type=int, default=None, help="seed of the K draw (default: random)")
    g = ap.add_argument_group("success (the proxy of the sim's success term)")
    g.add_argument("--contact-threshold", type=float, default=0.5,
                   help="fingertip force counting as contact, in the glove's force unit "
                        "(N once calibrated; glove units before: set from test grasps)")
    g.add_argument("--lift", type=float, default=0.15, help="lift of the TCP since the grasp formed [m]")
    g.add_argument("--hold", type=float, default=3.0, help="seconds grasped and lifted")
    g.add_argument("--time-limit", type=float, default=60.0, help="an episode running longer is discarded [s]")
    g = ap.add_argument_group("admittance and glove")
    g.add_argument("--law", choices=("dynamic", "static"), default="dynamic")
    g.add_argument("--mass", type=float, default=AdmittanceParams.mass, help="admittance mass M [kg]")
    g.add_argument("--max-offset", type=float, default=AdmittanceParams.max_offset, help="e_max [m]")
    g.add_argument("--regions", type=Path, default=Path("config/real/glove_tip_regions.yaml"),
                   help="fingertip map (scripts/real/glove_tip_regions.py)")
    g.add_argument("--noise-floor", type=Path, default=Path("config/real/glove_noise.npz"),
                   help="per-cell glove floor (his glove_noise.py); used when the file exists")
    g.add_argument("--calibration", type=Path, default=Path("config/real/glove_calibration.yaml"),
                   help="glove units -> N per finger; used when the file exists, glove units otherwise")
    g.add_argument("--glove-python", type=Path, default=None,
                   help="python of the env that reads the glove (default: his GLOVE_PYTHON)")
    g = ap.add_argument_group("teleoperation")
    g.add_argument("--pivot", choices=("wrist", "tcp"), default="wrist",
                   help="robot point that copies your wrist: the hand's wrist (default, as the sim's "
                        "--ik_point wrist) or the pendant TCP (his original mapping)")
    g.add_argument("--wrist-offset", type=float, default=0.0, metavar="M",
                   help="the hand's base (its wrist), metres out from the flange face along the flange "
                        "axis; measure with a ruler (default 0 = the flange face)")
    g = ap.add_argument_group("hand and cameras")
    g.add_argument("--hand-cutoff", type=float, default=0.0,
                   help="the driver's command low-pass [Hz]; 0 = off (agreed default, so the admittance "
                        "output reaches the motors directly), 5 = his default")
    g.add_argument("--serial", default="", help="hand serial, if more than one")
    g.add_argument("--cameras", default="",
                   help="comma-separated camera names from his config/cameras.yaml; empty = all, none = none")
    g.add_argument("--yes", action="store_true", help="skip the confirmation before the arm moves")
    common.add_hand_args(ap)
    common.add_common_args(ap, teleop=True, task=True, port=True)
    return ap.parse_args()


def check_args(args, dt: float) -> str | None:
    """Everything that can be refused before any hardware is touched; a message, or None."""
    ks = [float(k) for k in args.stiffness]
    if len(set(ks)) != len(ks) or min(ks) <= 0:
        return f"--stiffness needs distinct positive values, got {ks}"
    k_max = DynamicAdmittance(AdmittanceParams(mass=args.mass), dt).max_stable_stiffness
    if args.law == "dynamic" and any(not math.isinf(k) and k > k_max for k in ks):
        return f"K above the stability limit {k_max:.0f} N/m at M = {args.mass} kg, dt = {dt * 1000:.0f} ms"
    if args.record and args.show_k:
        return "--show-k is for practice only: recorded demos must not know their K"
    if args.pivot == "wrist" and not 0.0 <= args.wrist_offset <= 0.25:
        return f"--wrist-offset {args.wrist_offset} m: expected 0..0.25 m out from the flange face (metres, not mm?)"
    if getattr(args, "left", False):
        return "--left: our hand model and fingertip map are for the right Hand 2"
    if not args.regions.is_file():
        return f"fingertip map not found: {args.regions} (scripts/real/glove_tip_regions.py writes it)"
    if not common.hand_urdf(args).is_file():
        return f"URDF not found: {common.hand_urdf(args)}"
    return None


def make_recorders(args, task_cfg, camera_names, hand_side: str, action_hz: float):
    """One DeletableRecorder per K (refuses files of another split, K or format)."""
    folder = args.demo_dir or DEMO_DIRS[args.split]
    target = args.demos_per_k or (20 if args.split == "train" else 10)
    recorders = {}
    for k in (float(v) for v in args.stiffness):
        recorders[k] = DeletableRecorder(
            demo_path(folder, k), task_cfg,
            {"format": FORMAT, "split": args.split, "stiffness": k, "task": TASK},
            record_hand_joints=True, hand_side=hand_side, action_hz=action_hz,
            camera_names=camera_names, more_obs_keys=TactileFingertipAdmittance.OBS_KEYS,
            more_extra_keys=TactileFingertipAdmittance.EXTRA_KEYS)
    return StiffnessRecorders(recorders, target_per_k=target), folder, target


def main() -> int:
    args = parse_args()
    common.setup_logging(args.verbose)
    robot_cfg, teleop_cfg, task_cfg = common.load_configs(args)
    dt = robot_cfg.control.dt

    # ---- refusals before any hardware ------------------------------------------------------------
    problem = check_args(args, dt)
    if problem is None and (teleop_cfg.input.mode != "hands" or not teleop_cfg.input.pinch_freezes):
        problem = ("the start gate needs config/teleop.yaml input.mode 'hands' and clutch "
                   "'left_pinch_release'")
    if problem:
        print(f"{RED}{problem}{RESET}")
        return 1
    try:
        rig, camera_names = common.build_camera_rig(args.cameras)
    except (FileNotFoundError, ValueError) as exc:
        print(f"{RED}{exc}{RESET}")
        return 1
    side = common.hand_side(args)
    recorder = folder = target = None
    if args.record:
        try:
            recorder, folder, target = make_recorders(args, task_cfg, camera_names, side,
                                                      1.0 / robot_cfg.control.record_dt)
        except (OSError, ValueError) as exc:
            print(f"{RED}{exc}{RESET}")
            return 1
        if recorder.targets_reached:
            print(f"{YELLOW}every K already has {target} demos in {folder} ({recorder.summary()}); "
                  f"nothing to record (raise --demos-per-k, or use another --split / --demo-dir){RESET}")
            return 0

    # ---- the hand, its retargeting, the model ------------------------------------------------------
    urdf = common.hand_urdf(args)
    limits = urdf_limits(urdf)
    retarget_cfg = common.retarget_config(args)
    try:
        retargeter = HandRetargeter(optimizer="adaptive", limits=limits, config=retarget_cfg,
                                    hand_side=common.solve_side(args), mirror=common.mirror_left(args),
                                    human_limits=common.human_limits(args))
    except Exception as exc:  # noqa: BLE001 -- the retargeting stack reports its own reasons
        print(f"{RED}Retargeter failed to load: {exc}{RESET}")
        return 1
    lp_alpha = (yaml.safe_load(Path(retarget_cfg).read_text()) or {}).get("retarget", {}).get("lp_alpha")
    hand_kwargs = {"serial_number": args.serial, "handedness": side, "cutoff_hz": args.hand_cutoff}
    if args.hand_current is not None:
        hand_kwargs["effort_limit_a"] = float(args.hand_current)
    gripper = MeasuredWujiHand2(limits=limits, **hand_kwargs)
    retarget = lambda joints: gripper.to_normalised(retargeter(joints))  # noqa: E731 (as his record_grasps)
    model = HandModel(urdf)

    # ---- glove and admittance (the glove is connected later, after the hand) -------------------------
    params = AdmittanceParams(law=args.law, mass=args.mass, max_offset=args.max_offset)
    calibration = GloveCalibration.load(args.calibration)
    floor = args.noise_floor if args.noise_floor.is_file() else None
    tips = FingerRegions.load(args.regions)
    glove = GloveClient(**({"python": args.glove_python} if args.glove_python else {}))

    # ---- the session's parts (as his _common.build_session) ------------------------------------------
    quest = DualHandInput(teleop_cfg.gripper, pinch_freezes=teleop_cfg.input.pinch_freezes,
                          drive_side=common.drive_side(args))
    server = WebXRServer(teleop_cfg, quest)
    arm = UR5eArm(robot_cfg)
    episode_attrs = {
        "object_id": args.object, "cube_spot": args.cube_spot, "split": args.split,
        "hand_cutoff_hz": float(args.hand_cutoff),
        "hand_current_a": float(args.hand_current) if args.hand_current is not None else float("nan"),
        "retarget_config": str(retarget_cfg),
        "retarget_lp_alpha": float(lp_alpha) if lp_alpha is not None else float("nan"),
        "home_q": np.asarray(robot_cfg.home_q, dtype=float),
        "record_hz": float(1.0 / robot_cfg.control.record_dt),
        "teleop_pivot": args.pivot,
        "teleop_wrist_offset_m": float(args.wrist_offset) if args.pivot == "wrist" else float("nan"),
    }

    # ---- what is about to happen ---------------------------------------------------------------------
    ks = ", ".join(k_name(float(k)) for k in args.stiffness)
    print(f"\n{BOLD}{'Recording' if args.record else 'Practice'}: lift from home, cube on spot "
          f"{args.cube_spot}{RESET}")
    print(f"  K             one of {ks} N/m per episode, {'shown' if args.show_k else 'HIDDEN'}")
    if args.record:
        print(f"  files         {folder}/lift_K<K>.hdf5 ({args.split}), {target} demos per K wanted; "
              f"in them now: {recorder.summary()}")
    print(f"  admittance    {args.law}, M {args.mass:g} kg, e_max {1000 * args.max_offset:g} mm; force in "
          f"{calibration.unit}{'' if calibration.path else ' (no calibration yet)'}; noise floor "
          f"{floor or 'none'}")
    print(f"  success       thumb + one finger > {args.contact_threshold:g} {calibration.unit}, lifted "
          f"{100 * args.lift:g} cm for {args.hold:g} s; time limit {args.time_limit:g} s")
    print(f"  hand          driver low-pass {args.hand_cutoff:g} Hz, retargeter lp_alpha {lp_alpha}")
    print("  wrist         " + (f"pivot at the hand's wrist, {1000 * args.wrist_offset:g} mm out from the flange"
                               if args.pivot == "wrist" else "pivot at the pendant TCP (his mapping)"))
    print(f"  cameras       {', '.join(camera_names) if camera_names else 'none'}")
    print(f"\n{CONTROLS}\n")
    if not args.yes and not common.confirm("The fingers will park and the arm will go home. Proceed?"):
        return 1

    # ---- run (order as his record_grasps.py, except the glove: after the hand) ----------------------
    hand_filter = None
    try:
        if rig is not None:
            rig.start(lock=True)
            print(f"Cameras started: {', '.join(rig.names)}")
        server.start()
        common.print_quest_banner(teleop_cfg.server.port)
        info = arm.connect()
        print(f"Connected. PolyScope {info['polyscope']}, {info['safety_status']}.")
        if not common.connect_hand(gripper):
            return 1
        print(f"Hand connected. {gripper}")
        # The glove only now: the hand SDK, starting up, breaks a glove connection that already
        # exists in this process (found 2026-11-27: lost within 1 s of hand.connect(); with the hand
        # first, the glove streamed undisturbed). His record_grasps connects the glove first.
        glove.start()
        wait_for_glove(glove)
        print(f"Glove streaming: {glove.info.get('serial')}, {glove.rate_hz:.0f} Hz.")
        hand_filter = TactileFingertipAdmittance(
            glove, gripper, model, tips, params, dt=dt, calibration=calibration,
            regions_path=args.regions, floor_path=floor,
            ticks_per_record=robot_cfg.control.record_decimation)
        session = RealTeleopSession(
            arm=arm, quest=quest, server=server, robot_cfg=robot_cfg, teleop_cfg=teleop_cfg,
            task_cfg=task_cfg, gripper=gripper, recorder=recorder, hand_retargeter=retarget,
            cameras=rig, hand_filter=hand_filter, k_values=[float(k) for k in args.stiffness],
            success=SuccessDetector(SuccessParams(contact_threshold=args.contact_threshold,
                                                  lift_m=args.lift, hold_s=args.hold)),
            rng=np.random.default_rng(args.seed), time_limit_s=args.time_limit, park_s=PARK_S,
            show_k=args.show_k, episode_attrs=episode_attrs)
        tool = getattr(arm, "_tool", None)    # the pendant's TCP offset, read by connect()
        if tool is not None:
            session.episode_attrs["tcp_offset_matrix"] = np.asarray(tool, dtype=float)
        if args.pivot == "wrist":
            if tool is None:
                print(f"{RED}The pendant's TCP offset was not read; cannot place the wrist pivot.{RESET}")
                return 1
            d = pivot_offset(tool, args.wrist_offset)
            session.retargeter.set_offset(d)
            print(f"Wrist pivot: TCP {1000 * tool[2, 3]:.0f} mm out, pivot {1000 * args.wrist_offset:.0f} mm "
                  f"out -> d = {np.round(1000 * d, 1).tolist()} mm (TCP relative to the pivot)")
        # Park the fingers before the arm moves and before servoing: ramp_to blocks.
        gripper.ramp_to(gripper.neutral_pose, seconds=PARK_S)
        arm.move_home()
        arm.start_servo()
        session.prepare()
        session.run()
        return 0
    except SafetyAbort as exc:
        print(f"\n{RED}SAFETY ABORT:{RESET} {exc}")
        return 1
    except (RuntimeError, TimeoutError, OSError) as exc:
        print(f"\n{RED}{exc}{RESET}")
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        # Hand first: it has no e-stop, and arm shutdown can block (as his record_grasps).
        try:
            gripper.disconnect()
        except Exception as exc:  # noqa: BLE001
            print(f"{RED}hand disconnect failed: {exc}{RESET}")
        arm.close()
        server.stop()
        if rig is not None:
            rig.stop()
        glove.close()
        if hand_filter is not None:
            stats = hand_filter.tick_stats()
            print(f"\nServo ticks: {stats['ticks']}, mean {1000 * stats['mean_interval']:.2f} ms, "
                  f"late (> 12 ms) {stats['late_ticks']}; hand repeated reads {gripper.repeated_reads}")
        if recorder is not None:
            print(f"Demos in {folder}: {recorder.summary() if args.show_k or recorder.targets_reached else recorder.n_saved}")
        print("Disconnected.")


def wait_for_glove(glove: GloveClient, seconds: float = 3.0, min_frames: int = 50) -> None:
    """Refuse to go on unless glove frames are actually arriving."""
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end and glove.frames < min_frames and not glove.lost:
        time.sleep(0.05)
    if glove.lost or glove.frames < min_frames:
        raise RuntimeError(
            f"The glove is {'lost' if glove.lost else 'silent'} ({glove.frames} frames in {seconds:g} s). "
            "To see the glove server's own messages, start it by hand in another terminal and run again:\n"
            '  "$(python -c \'from quest_teleop2.tactile.glove import GLOVE_PYTHON; print(GLOVE_PYTHON)\')" '
            '-I "$HIS/scripts/glove/glove_server.py"')


if __name__ == "__main__":
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
    sys.exit(main())
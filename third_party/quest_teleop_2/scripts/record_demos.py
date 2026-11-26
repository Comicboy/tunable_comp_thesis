#!/usr/bin/env python
"""Teleoperate and record demonstrations to an HDF5 dataset.

    python scripts/record_demos.py --out data/demos.hdf5 --start 0 --goal 0
    python scripts/record_demos.py --out data/demos.hdf5 --cameras none   # state only
    python scripts/record_demos.py --teach                                # capture task poses

Recording an existing --out appends to it and continues the numbering.

Right hand drives by default; --left uses the left Wuji Hand and swaps the
operator's roles.

Left controller, since you cannot reach the laptop while wearing the headset:

    thumbstick L/R   cycle start position     thumbstick U/D   cycle goal
    trigger          start episode            X  save          Y  discard
    grip (hold)      clutch                   thumbstick push  re-anchor

Per demonstration: select start and goal, place the object, hold the clutch,
press trigger, perform the motion, press X.

--teach jogs the arm to each start and goal and writes the TCP poses into
config/task.yaml. Those poses are for reference only; the policy sees the
one-hot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quest_teleop2.config import CONFIG_DIR  # noqa: E402
from quest_teleop2.data.recorder import EpisodeRecorder  # noqa: E402
from quest_teleop2.quest.quest_input import Event  # noqa: E402
from quest_teleop2.robot.rtde_io import RobotState  # noqa: E402
from quest_teleop2.robot.gripper import WujiHand2Gripper  # noqa: E402
from quest_teleop2.robot.safety import SafetyAbort  # noqa: E402
from quest_teleop2.teleop.hand_retarget import HandRetargeter, urdf_limits  # noqa: E402
from scripts._common import (  # noqa: E402
    retarget_config,
    # noqa: E402,
    add_common_args,
    build_hand,
    add_hand_args,
    build_camera_rig,
    build_session,
    confirm,
    connect_hand,
    controls_help,
    drive_side,
    hand_side,
    hand_urdf,
    human_limits,
    load_configs,
    mirror_left,
    print_quest_banner,
    setup_logging,
    solve_side,
)



class TeachMode:
    """Walks through every start and goal, capturing the current TCP pose."""

    def __init__(self, task_cfg, server, path: Path) -> None:
        self.task = task_cfg
        self.server = server
        self.path = path
        self.points = [("start", i, n) for i, n in enumerate(task_cfg.starts)]
        self.points += [("goal", i, n) for i, n in enumerate(task_cfg.goals)]
        self.cursor = 0
        self.captured: dict[tuple[str, str], list[float]] = {}

    @property
    def current(self) -> tuple[str, int, str]:
        return self.points[self.cursor]

    def label(self) -> str:
        kind, _, name = self.current
        mark = "captured" if (kind, name) in self.captured else "empty"
        return f"[{self.cursor + 1}/{len(self.points)}] {kind} {name} ({mark})"

    def handle(self, event: Event, state: RobotState) -> bool:
        """Controller-mode path. Maps buttons onto the same actions as keys."""
        mapping = {
            Event.CYCLE_START: "cycle_start",
            Event.CYCLE_GOAL: "cycle_goal",
            Event.SAVE: "save",
            Event.DISCARD: "discard",
            Event.START_EPISODE: "start",
        }
        action = mapping.get(event)
        return self.handle_key(action, state) if action else False

    def handle_key(self, action: str, state: RobotState) -> bool:
        """Keyboard path -- what actually runs in hands-only mode.

        Returning True consumes the key so the session's normal episode
        handling does not also act on it.
        """
        if action in ("cycle_start", "cycle_goal"):
            step = 1 if action == "cycle_start" else -1
            self.cursor = (self.cursor + step) % len(self.points)
            self.server.banner(self.label())
            print(f"\r  -> {self.label():60}", end="", flush=True)
            return True
        if action == "save":
            kind, _, name = self.current
            self.captured[(kind, name)] = [round(float(v), 5) for v in state.actual_tcp_pose]
            self.server.banner(f"captured {kind} {name}", "ok")
            print(f"\n  captured {kind} {name}: {np.round(state.actual_tcp_pose, 4).tolist()}")
            if len(self.captured) == len(self.points):
                self.server.banner("all points captured", "ok")
                print("  all points captured -- press q to write them to task.yaml")
            self.cursor = (self.cursor + 1) % len(self.points)
            return True
        if action == "discard":
            kind, _, name = self.current
            if self.captured.pop((kind, name), None) is not None:
                self.server.banner(f"cleared {kind} {name}", "bad")
                print(f"\n  cleared {kind} {name}")
            return True
        if action == "start":
            self.server.banner(self.label())
            return True
        return False  # engage / reanchor / quit fall through to the session

    def write(self) -> bool:
        if not self.captured:
            print("\nNothing captured; task.yaml left unchanged.")
            return False
        data = yaml.safe_load(self.path.read_text()) or {}
        data["start_poses"] = {
            n: self.captured[("start", n)] for n in self.task.starts if ("start", n) in self.captured
        }
        data["goal_poses"] = {
            n: self.captured[("goal", n)] for n in self.task.goals if ("goal", n) in self.captured
        }
        self.path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=None))
        missing = [f"{k} {n}" for (k, n) in
                   [("start", s) for s in self.task.starts] + [("goal", g) for g in self.task.goals]
                   if (k, n) not in self.captured]
        print(f"\nWrote {len(self.captured)} pose(s) to {self.path}")
        if missing:
            print(f"\033[33mStill missing:\033[0m {', '.join(missing)}")
        return True


#: Seconds to ramp the fingers to the parking pose at startup.
HAND_PARK_S = 2.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/demos.hdf5"), help="HDF5 output")
    parser.add_argument("--no-hand", action="store_true",
                        help="record arm only, leaving the hand de-energised")
    parser.add_argument("--optimizer", default="vector", choices=("vector", "adaptive"),
                        help="hand retargeting solver")
    parser.add_argument("--cutoff", type=float, default=WujiHand2Gripper.DEFAULT_CUTOFF_HZ,
                        help="hand low-pass cutoff, Hz")
    add_hand_args(parser)
    parser.add_argument("--serial", default="", help="hand serial, if more than one")
    parser.add_argument("--no-home", action="store_true",
                        help="skip the move to the home pose at startup")
    parser.add_argument("--yes", action="store_true",
                        help="skip the confirmation prompt")
    parser.add_argument(
        "--record-hand", action=argparse.BooleanOptionalAction, default=True,
        help="store the 25 raw tracked hand joints alongside the 20 commanded "
        "ones. On by default; --no-record-hand turns it off. Not the same as "
        "--no-hand, which de-energises the hand entirely",
    )
    parser.add_argument("--start", type=int, default=0,
                        help="initial start index, so a per-variant session cannot be "
                             "mislabelled by miscounting [ presses")
    parser.add_argument("--goal", type=int, default=0, help="initial goal index")
    parser.add_argument("--teach", action="store_true", help="capture start/goal poses into task.yaml")
    parser.add_argument(
        "--cameras",
        default="",
        help="comma-separated camera names from config/cameras.yaml; empty means "
        "all of them, 'none' records state-only. Order always comes from the "
        "config file, never from this flag.",
    )
    parser.add_argument(
        "--no-auto-home",
        action="store_true",
        help="do not return to the home pose when an episode starts. On by "
        "default: identical initial conditions are what make a dataset learnable.",
    )
    add_common_args(parser, teleop=True, task=True, port=True)
    args = parser.parse_args()
    setup_logging(args.verbose)
    robot_cfg, teleop_cfg, task_cfg = load_configs(args)

    # Checked here, before any hardware is energised. h5py only fails when the
    # first episode is SAVED, which is after a demonstration has been performed
    # -- so a directory passed to --out would cost you the take, not just the
    # command.
    if not args.teach:
        if args.out.is_dir():
            print(f"\033[31m--out {args.out} is a directory.\033[0m Pass a file:")
            print(f"  --out {args.out.as_posix().rstrip('/')}/demos.hdf5")
            return 1
        if args.out.suffix != ".hdf5":
            print(f"\033[31m--out {args.out} does not end in .hdf5.\033[0m")
            return 1

    teach: TeachMode | None = None
    recorder: EpisodeRecorder | None = None
    gripper = None
    retarget = None
    rig = None
    camera_names: tuple[str, ...] = ()

    # Teach mode only jogs the arm to capture TCP poses; nothing is recorded, so
    # there is nothing for a camera to be attached to.
    if not args.teach:
        try:
            rig, camera_names = build_camera_rig(args.cameras)
        except (FileNotFoundError, ValueError) as exc:
            print(f"\033[31m{exc}\033[0m")
            return 1

    # Teach mode only jogs the arm to capture poses, so the hand stays out
    # of it -- no reason to energise 20 joints to record a TCP.
    if not args.no_hand and not args.teach:
        if not hand_urdf(args).exists():
            print(f"\033[31mURDF not found: {hand_urdf(args)}\033[0m")
            return 1
        limits = urdf_limits(hand_urdf(args))
        if args.optimizer != "adaptive":
            print(f"\033[33mOnly an adaptive config exists for this hand; "
                  f"ignoring --optimizer {args.optimizer}.\033[0m")
        try:
            # config= is not optional. Without it the retargeter falls back to
            # the optimiser-keyed default, which is the FIRST GENERATION's AVP
            # config pointing at the first generation's hand model.
            retargeter = HandRetargeter(optimizer="adaptive", limits=limits,
                                        config=retarget_config(args),
                                        hand_side=solve_side(args),
                                        mirror=mirror_left(args),
                                        human_limits=human_limits(args))
        except Exception as exc:
            print(f"\033[31mRetargeter failed to load: {exc}\033[0m")
            return 1
        gripper = build_hand(args, limits=limits, serial_number=args.serial,
                             cutoff_hz=args.cutoff)
        retarget = lambda joints: gripper.to_normalised(retargeter(joints))  # noqa: E731

    if args.teach:
        print("\n\033[1mTeach mode\033[0m -- capturing start/goal poses. Nothing is recorded.")
        print(f"  Move the arm with your {side} hand ({gate} pinch to freeze and "
              "reposition), then:")
        print("    SPACE   engage / disengage motion")
        print("    s       capture the current TCP as this point")
        print("    r       clear this point")
        print("    [  ]    next / previous point")
        print("    q       finish and write the poses to task.yaml\n")
    else:
        print(f"\n\033[1mRecording demonstrations to {args.out}\033[0m")
        print(f"  variants: {task_cfg.n_starts} starts x {task_cfg.n_goals} goals")
        print(f"  action rate: {1 / robot_cfg.control.record_dt:.1f} Hz "
              f"(1 action every {robot_cfg.control.record_decimation} servo ticks "
              f"at {robot_cfg.control.frequency_hz:.0f} Hz)")
        print(f"  starting on \033[1m{task_cfg.starts[args.start]} -> "
              f"{task_cfg.goals[args.goal]}\033[0m  ([ and ] still cycle)")
        if camera_names:
            print(f"  cameras: \033[1m{', '.join(camera_names)}\033[0m "
                  "(this order is baked into the file)")
        else:
            print("  cameras: \033[33mnone\033[0m -- state-only dataset")
        side = drive_side(args)
        gate = "left" if side == "right" else "right"
        print(f"\n{controls_help(side)}")

    if not args.yes and not confirm("The arm will move when you hold the clutch. Proceed?"):
        print("Aborted.")
        return 1

    if not args.teach:
        recorder = EpisodeRecorder(
            args.out, task_cfg, record_hand_joints=args.record_hand,
            hand_side=hand_side(args),
            action_hz=1.0 / robot_cfg.control.record_dt,
            camera_names=camera_names,
        )
        if recorder.n_saved:
            print(f"  appending to {recorder.n_saved} existing demos")

    if not 0 <= args.start < task_cfg.n_starts:
        print(f"\033[31m--start {args.start} out of range "
              f"(0..{task_cfg.n_starts - 1})\033[0m")
        return 1
    if not 0 <= args.goal < task_cfg.n_goals:
        print(f"\033[31m--goal {args.goal} out of range "
              f"(0..{task_cfg.n_goals - 1})\033[0m")
        return 1

    arm, server, session = build_session(
        args, recorder=recorder, gripper=gripper, hand_retargeter=retarget, cameras=rig
    )
    session.selection.start_idx = args.start
    session.selection.goal_idx = args.goal

    if args.teach:
        teach = TeachMode(task_cfg, server, args.task_config or (CONFIG_DIR / "task.yaml"))
        session.event_hook = teach.handle
        session.key_hook = teach.handle_key

    try:
        if rig is not None:
            # Before the arm, so a busy or unplugged camera fails while nothing
            # is energised. lock=True pins exposure/white balance/focus, which
            # is what keeps appearance from drifting across a 120-demo session.
            rig.start(lock=True)
            print(f"Cameras started: {', '.join(rig.names)}")

        server.start()
        print_quest_banner(teleop_cfg.server.port)

        info = arm.connect()
        print(f"Connected. PolyScope {info['polyscope']}, {info['safety_status']}.")
        if gripper is not None:
            if not connect_hand(gripper):
                return 1
            print(f"Hand connected. {gripper}")
            # Before the arm moves, and before servoing: ramp_to blocks, and
            # the RTDE watchdog times out after 0.1 s.
            gripper.ramp_to(gripper.neutral_pose, seconds=HAND_PARK_S)
        if not args.no_home:
            arm.move_home()
        arm.start_servo()

        if teach is not None:
            server.update_hud(task=teach.label(), episode="teach mode")
        print("\nPut the headset on and open the page.\n")
        session.run()
        return 0

    except SafetyAbort as exc:
        print(f"\n\033[31mSAFETY ABORT:\033[0m {exc}")
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        # Hand first: it has no e-stop, and arm shutdown can block.
        if gripper is not None:
            try:
                gripper.disconnect()
            except Exception as exc:  # noqa: BLE001
                print(f"\033[31mhand disconnect failed: {exc}\033[0m")
        arm.close()
        server.stop()
        if rig is not None:
            rig.stop()
        if teach is not None:
            teach.write()
        if recorder is not None:
            print(f"\n{recorder.summary()}")
            if recorder.n_saved:
                print(f"Next: python scripts/replay_demo.py --dataset {args.out} --demo 0")
        print("Disconnected.")


if __name__ == "__main__":
    raise SystemExit(main())

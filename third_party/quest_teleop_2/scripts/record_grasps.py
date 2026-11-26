#!/usr/bin/env python
"""Record manipulation demonstrations, each starting from a randomised approach.

    python scripts/record_grasps.py --out data/right/grasp/cube.hdf5 --object cube
    python scripts/record_grasps.py --out data/right/grasp/cube.hdf5 --noise-xy 0.03
    python scripts/record_grasps.py --out data/right/p_and_p/cube.hdf5 --object cube
    python scripts/record_grasps.py --out data/right/reorient/full.hdf5 --object cube --from-home

`--from-home` skips the scripted approach: recording starts at home with the
fingers parked and you teleoperate the whole task, as an end-to-end baseline.

Press `t` and the robot homes, finds the object with the top-down camera, and
moves itself to a pre-grasp pose with a deliberate error injected. Recording
starts the moment it arrives and runs until you save: the approach is not
yours to perform, everything after it is. A grasp ends at the lift; a
pick-and-place carries on through the transport, the place and the release.

The injected error is the point of the exercise rather than a nuisance. A
policy trained only on approaches that arrived perfectly has never seen the
states it will actually face, because at deployment the approach carries
calibration and detection error. How much error it can absorb is the result;
the noise is its x-axis, and it is logged per episode.

Controls are the ordinary ones, with one addition:

    SPACE   engage / disengage motion (starts DISENGAGED)
    t       home, approach, and start recording
    s       save as a SUCCESS
    f       save as a FAILURE -- a genuine miss is data, not waste
    r       discard a spoiled take (tracking dropped, wrong object)
    h       home now      a  re-anchor      q  quit

Motion stays engaged after a save, on purpose: the fingers are not
backdrivable, so you need it live to open the hand and release the object.
Freeze with the gate pinch to reposition, exactly as in ordinary recording.

The fingers ramp to the neutral pose during every approach, so each episode
starts from the same hand configuration, and they ease back onto your tracked
pose over about a second once the arm arrives. They map your hand absolutely
and have no anchor to drop, so without that easing the whole pose would arrive
on a single tick.

Per episode: t, engage, perform the task, then s or f. Where you stop is where
the policy will stop, so stop at the same place every time.

The start and goal one-hots are the session's, cycled with `[` and `]` before
`t`. They are recorded per episode, but no policy here takes them as input, so
a goal that moves is invisible unless a camera can see it.

Record failures. An advantage-weighted update needs negatives to have anything
to contrast a good action against, and a dataset of nothing but successes gives
it none.

Needs config/table.yaml, including a template pose. Run calibrate_table.py
first, and check the approach with:

    python scripts/troubleshooting/scripted_grasp.py --reach-only --view
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quest_teleop2.data.recorder import EpisodeRecorder  # noqa: E402
from quest_teleop2.data.schema import ADMITTANCE_EXTRA_KEYS, TACTILE_OBS_KEYS  # noqa: E402
from quest_teleop2.robot.gripper import WujiHand2Gripper  # noqa: E402
from quest_teleop2.robot.rtde_io import RobotState  # noqa: E402
from quest_teleop2.robot.safety import SafetyAbort  # noqa: E402
from quest_teleop2.teleop.hand_retarget import HandRetargeter, urdf_limits  # noqa: E402
from quest_teleop2.vision.table import (  # noqa: E402
    TableMap, active_detector, locate_cube, pre_grasp_pose,
)
from scripts._common import (  # noqa: E402
    retarget_config,  # noqa: E402
    APPROACH_DEFAULTS,
    add_common_args,
    add_sam_args,
    approach_settings,
    locator_clash,
    sam_object,
    start_sam,
    stop_sam,
    build_hand,
    add_glove_args,
    add_hand_args,
    build_camera_rig,
    build_hand_filter,
    build_session,
    close_hand_filter,
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
from scripts.troubleshooting.scripted_grasp import _floored  # noqa: E402

GREEN, YELLOW, RED, BOLD, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[1m", "\033[0m"

#: Seconds to ramp the fingers to the neutral pose before the arm travels. Runs
#: inside approach()'s stopped window, where taking time is safe.
PARK_RAMP_S = 1.0

#: Seconds to blend the fingers from neutral back onto the operator's tracked
#: pose once the loop resumes. The hand maps absolutely and has no anchor, so
#: without this the whole pose arrives on one tick.
EASE_S = 0.8


class PendingAttrs:
    """Wraps the recorder so per-episode facts survive `start_episode`.

    The approach knows the object's position and the injected noise, but it
    runs BEFORE the episode exists, and `start_episode` clears any attributes
    set earlier. Holding them here and applying them the instant recording
    begins is what keeps that ordering from silently dropping them -- and every
    one of these is unrecoverable once the file is written.
    """

    def __init__(self, inner: EpisodeRecorder, goals=None) -> None:
        self._inner = inner
        self.pending: dict[str, object] = {}
        self.goals = goals

    def start_episode(self, start_idx: int, goal_idx: int) -> None:
        self._inner.start_episode(start_idx, goal_idx)
        attrs = dict(self.pending)
        if self.goals is not None:
            xy = np.asarray(self.goals[goal_idx], dtype=float)
            attrs["goal_xy_m"] = xy
            print(f"  goal {goal_idx}: ({xy[0]:+.3f}, {xy[1]:+.3f}) m")
        if attrs:
            self._inner.set_episode_attrs(**attrs)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


class GraspApproach:
    """Homes, locates the object, and drives to a noisy pre-grasp pose.

    Hooked onto the session's `t` key. Returning True from `handle_key`
    consumes the key, which is how a failed approach declines to start an
    episode at all -- better than recording one that begins from nowhere in
    particular.
    """

    def __init__(self, session, arm, gripper, cam, table: TableMap,
                 recorder: PendingAttrs, args: argparse.Namespace) -> None:
        self.session = session
        self.arm = arm
        self.gripper = gripper
        self.cam = cam
        self.table = table
        self.recorder = recorder
        self.args = args
        self.template = table.require_template()
        self.grasp_z = float(self.template[2])
        self.rng = np.random.default_rng(args.seed)
        self.n = 0

    def handle_key(self, action: str, state: RobotState) -> bool:
        # Saving does NOT disengage: the fingers are not backdrivable, so
        # motion stays live for the operator to let go.
        if action != "start":
            return False
        if self.recorder.is_recording:
            return False  # let the session say "already recording"
        return not self.run()

    def run(self) -> bool:
        session, arm = self.session, self.arm
        # LOOK BEFORE MOVING. Homing first meant a failed detection still cost
        # a full arm move, which is the one case where the robot should do
        # nothing at all -- the reason it failed might be that the object is
        # somewhere the calibration cannot vouch for.
        frame, _age, _seq = self.cam.snapshot()
        found = None if frame is None else locate_cube(frame, self.table)
        if found is None:
            print(f"\n  {RED}object not found -- not moving.{RESET}")
            session.server.banner("object not found", "bad")
            print("   The reason is logged above: too dim to trust, outside the")
            print("   calibrated region, or genuinely not there. If the ARM is")
            print("   blocking the camera, press h to home and then t again.")
            return False
        cube_xy, score = found

        # Home first so the moveL below is a predictable straight line rather
        # than a sweep from wherever teleoperation ended.
        session._stale_state = True
        print("\n  homing...", end="", flush=True)
        session.server.banner("homing", "idle")
        home = arm.rehome()
        if getattr(self.args, "from_home", False):
            return self._park_at_home(home, cube_xy, score)

        # Upward only. Symmetric z error would put some approaches below the
        # height the template was taught at, and that is the one direction with
        # a table in it.
        noise = np.array([self.rng.uniform(-self.args.noise_xy, self.args.noise_xy),
                          self.rng.uniform(-self.args.noise_xy, self.args.noise_xy),
                          self.rng.uniform(0.0, self.args.noise_z)])
        aim = self.table.aim(cube_xy)
        raise_m = float(getattr(self.args, "approach_raise_m", 0.0))
        target = _floored(
            pre_grasp_pose(aim, self.template, self.grasp_z + self.args.standoff + raise_m,
                           noise),
            self.grasp_z)
        print(f" approaching ({cube_xy[0]:+.3f}, {cube_xy[1]:+.3f}) "
              f"+ ({noise[0] * 1000:+.0f}, {noise[1] * 1000:+.0f}, "
              f"{noise[2] * 1000:+.0f}) mm...", end="", flush=True)
        session.server.banner("approaching", "idle")
        # Park at neutral so every episode starts from the same configuration:
        # the observation carries the 20 joint positions, so a leftover pose
        # would be a per-episode hidden variable. Ramped inside approach()'s
        # stopped window, the only place servoing is not live.
        state = arm.approach(
            target,
            before_move=lambda: self.gripper.ramp_to(self.gripper.neutral_pose,
                                                     seconds=PARK_RAMP_S),
            vertical=bool(getattr(self.args, "approach_from_above", False)))

        self._resume(state)

        self.n += 1
        self.recorder.pending = {
            "object_id": self.args.object,
            "approach_noise_m": noise.astype(float),
            "object_xy_m": np.asarray(cube_xy, dtype=float),
            "aim_xy_m": np.asarray(aim, dtype=float),
            "arrived_tcp_pose": np.asarray(state.actual_tcp_pose, dtype=float),
            "detector_score": float(score),
            "detector": active_detector(),
            "standoff_m": float(self.args.standoff),
            "noise_xy_m": float(self.args.noise_xy),
            "noise_z_m": float(self.args.noise_z),
            "approach_raise_m": raise_m,
            "hand_current_a": float(getattr(self.gripper, "current_limit", None) or 0.0),
        }
        print(f" {GREEN}ready.{RESET}  engage and grasp; s saves, f saves a miss.")
        session.server.banner("REC - grasp now", "ok")
        return True

    def _resume(self, state: RobotState) -> None:
        # The TCP moved and the operator's hand did not: drop the anchor, ease the fingers.
        self.session.retargeter.clear()
        self.session.limiter.reset(state.actual_tcp_pose)
        self.session.pinch.reset()
        self.session.ease_gripper(EASE_S)

    def _park_at_home(self, home: RobotState, cube_xy, score) -> bool:
        # Fingers park only once the arm is home, with servoing stopped.
        self.arm.stop_servo()
        self.gripper.ramp_to(self.gripper.neutral_pose, seconds=PARK_RAMP_S)
        self.arm.start_servo()
        self._resume(home)
        self.n += 1
        self.recorder.pending = {
            "object_id": self.args.object,
            "object_xy_m": np.asarray(cube_xy, dtype=float),
            "detector_score": float(score),
            "detector": active_detector(),
            "from_home": True,
            "hand_current_a": float(getattr(self.gripper, "current_limit", None) or 0.0),
        }
        print(f" cube at ({cube_xy[0]:+.3f}, {cube_xy[1]:+.3f}).  {GREEN}ready at home.{RESET}  "
              f"engage and do the whole task; s saves, f saves a miss.")
        self.session.server.banner("REC - whole task", "ok")
        return True


def glove_recorded(path: Path) -> bool | None:
    """Whether the episodes already in `path` carry the glove's data; None for no episodes."""
    if not path.exists():
        return None
    import h5py  # noqa: PLC0415

    with h5py.File(path, "r") as f:
        demos = list(f["data"].values()) if "data" in f else []
        return None if not demos else "tactile" in demos[0]["obs"]


def resolve_approach(args: argparse.Namespace) -> str:
    """Fill unset approach settings from the file being appended to, else the defaults,
    and refuse a value that would put two settings in one file. Returns the source."""
    stored, source = APPROACH_DEFAULTS, "the defaults"
    if args.out.exists() and not getattr(args, "from_home", False):
        try:
            stored, source = approach_settings(args.out), str(args.out)
        except (KeyError, OSError):
            pass
    for flag, value in zip(("standoff", "noise_xy", "noise_z"), stored):
        given = getattr(args, flag)
        if given is None:
            setattr(args, flag, float(value))
        elif source != "the defaults" and abs(given - value) > 1e-6:
            name = "--" + flag.replace("_", "-")
            raise ValueError(f"{args.out} was recorded with {name} {value:g}; {given:g} would put "
                             "two approach settings in one file. Leave it out, or record to a "
                             "new file.")
    return source


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/grasp/cube.hdf5"),
                        help="HDF5 output; recording an existing file appends")
    parser.add_argument("--object", default="cube",
                        help="object identifier, stored on every episode. Results "
                             "are reported per object and this cannot be "
                             "reconstructed afterwards")
    parser.add_argument("--standoff", type=float, default=None,
                        help="metres above the template pose to approach at. Default: what "
                             f"--out was recorded with, else {APPROACH_DEFAULTS[0]}")
    parser.add_argument("--noise-xy", type=float, default=None,
                        help="uniform +-metres of injected approach error in x and y. "
                             f"Default: as --out, else {APPROACH_DEFAULTS[1]}")
    parser.add_argument("--noise-z", type=float, default=None,
                        help="metres of injected error in z, UPWARD ONLY. Default: as --out, "
                             f"else {APPROACH_DEFAULTS[2]}")
    parser.add_argument("--seed", type=int, default=0, help="noise seed")
    parser.add_argument("--table", type=Path, default=None,
                        help="override config/table.yaml")
    parser.add_argument("--cameras", default="",
                        help="comma-separated camera names; empty means all of "
                             "them. Order always comes from config/cameras.yaml")
    parser.add_argument("--optimizer", default="vector", choices=("vector", "adaptive"),
                        help="hand retargeting solver")
    parser.add_argument("--cutoff", type=float, default=WujiHand2Gripper.DEFAULT_CUTOFF_HZ,
                        help="hand low-pass cutoff, Hz")
    parser.add_argument("--serial", default="", help="hand serial, if more than one")
    parser.add_argument(
        "--record-hand", action=argparse.BooleanOptionalAction, default=True,
        help="store the 25 raw tracked hand joints alongside the 20 commanded ones")
    parser.add_argument("--goal-xy", nargs="+", metavar="X,Y", default=None,
                        help="goal positions in metres, indexed by the ] key. "
                             "Logged per episode as goal_xy_m; required for "
                             "goal-relative training")
    parser.add_argument("--from-home", action="store_true",
                        help="no scripted approach: record from home and teleoperate "
                             "the whole task (end-to-end baseline)")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation")
    add_sam_args(parser)
    add_hand_args(parser)
    add_glove_args(parser)
    add_common_args(parser, teleop=True, task=True, port=True)
    args = parser.parse_args()
    setup_logging(args.verbose)
    robot_cfg, teleop_cfg, task_cfg = load_configs(args)

    # Checked before any hardware is energised. h5py only fails when the first
    # episode is SAVED, which is after a demonstration has been performed.
    if args.out.is_dir():
        print(f"{RED}--out {args.out} is a directory.{RESET} Pass a file.")
        return 1
    if args.out.suffix != ".hdf5":
        print(f"{RED}--out {args.out} does not end in .hdf5.{RESET}")
        return 1
    try:
        settings_from = resolve_approach(args)
        sam_spec = sam_object(args)
    except (OSError, ValueError) as exc:
        print(f"{RED}{exc}{RESET}")
        return 1
    clash = locator_clash(args, args.out) if args.out.exists() else None
    if clash:
        print(f"{RED}{clash} Record to a new file, or pass "
              f"{'--no-sam' if args.sam else '--sam'} to carry this one on.{RESET}")
        return 1

    try:
        table = TableMap.load(args.table)
        table.require_template()
    except (FileNotFoundError, ValueError) as exc:
        print(f"{RED}{exc}{RESET}")
        return 1

    try:
        rig, camera_names = build_camera_rig(args.cameras)
    except (FileNotFoundError, ValueError) as exc:
        print(f"{RED}{exc}{RESET}")
        return 1
    if rig is None or table.camera not in camera_names:
        print(f"{RED}The top-down camera {table.camera!r} must be recorded too -- "
              f"the approach cannot find the object without it.{RESET}")
        return 1
    detect = camera_names.index(table.camera)

    if not hand_urdf(args).exists():
        print(f"{RED}URDF not found: {hand_urdf(args)}{RESET}")
        return 1
    limits = urdf_limits(hand_urdf(args))
    try:
        if args.optimizer != "adaptive":
            print(f"{YELLOW}Only an adaptive config exists for this hand; "
                  f"ignoring --optimizer {args.optimizer}.{RESET}")
        retargeter = HandRetargeter(optimizer="adaptive", limits=limits,
                                    config=retarget_config(args),
                                    hand_side=solve_side(args),
                                    mirror=mirror_left(args),
                                    human_limits=human_limits(args))
    except Exception as exc:  # noqa: BLE001
        print(f"{RED}Retargeter failed to load: {exc}{RESET}")
        return 1
    gripper = build_hand(args, limits=limits, serial_number=args.serial,
                         cutoff_hz=args.cutoff)
    retarget = lambda joints: gripper.to_normalised(retargeter(joints))  # noqa: E731

    goals = None
    if args.goal_xy is not None:
        try:
            goals = [np.array([float(v) for v in s.split(",")], dtype=float)
                     for s in args.goal_xy]
        except ValueError:
            print(f"{RED}--goal-xy takes X,Y pairs in metres, e.g. 0.65,0.52{RESET}")
            return 1
        if any(g.shape != (2,) for g in goals):
            print(f"{RED}--goal-xy takes X,Y pairs in metres, e.g. 0.65,0.52{RESET}")
            return 1
        if len(goals) < task_cfg.n_goals:
            print(f"{RED}task.yaml declares {task_cfg.n_goals} goals but only "
                  f"{len(goals)} --goal-xy given; ] would cycle past the end.{RESET}")
            return 1
        box = robot_cfg.safety.workspace
        for i, g in enumerate(goals):
            if not (box.x[0] <= g[0] <= box.x[1] and box.y[0] <= g[1] <= box.y[1]):
                print(f"{RED}goal {i} ({g[0]:+.3f}, {g[1]:+.3f}) is outside the "
                      f"workspace x{box.x} y{box.y}{RESET}")
                return 1

    print(f"\n{BOLD}Recording grasps to {args.out}{RESET}")
    print(f"  object            {args.object}   hand: {hand_side(args)}")
    if goals is None:
        print(f"  {YELLOW}no --goal-xy: goal_xy_m will NOT be logged, and the "
              f"file cannot be used for goal-relative training{RESET}")
    else:
        print("  goals             " + "  ".join(
            f"[{i}] ({g[0]:+.3f}, {g[1]:+.3f})" for i, g in enumerate(goals))
            + "   ] cycles")
    print(f"  calibration RMS   {table.rms_error_m * 1000:.1f} mm")
    if args.from_home:
        print(f"  {BOLD}from home{RESET}         no scripted approach; you teleoperate "
              f"the whole task")
    else:
        print(f"  approach at       z={float(table.template_pose[2]) + args.standoff:.3f}"
              f"   offset ({table.approach_offset[0] * 1000:+.0f}, "
              f"{table.approach_offset[1] * 1000:+.0f}) mm")
        print(f"  injected noise    +-{args.noise_xy * 1000:.0f} mm xy, "
              f"+{args.noise_z * 1000:.0f} mm z (upward only)")
        print(f"  settings from     {settings_from}")
    print(f"  cameras           {', '.join(camera_names)} "
          f"(this order is baked into the file)")
    print(f"\n{controls_help(drive_side(args))}")

    glove_wanted = args.glove or args.stiffness is not None
    held = glove_recorded(args.out)
    if held is not None and held != glove_wanted:
        print(f"{RED}{args.out} holds episodes recorded {'with' if held else 'without'} the "
              f"glove; record to a new file, or {'add --glove' if held else 'drop --glove and --stiffness'}."
              f"{RESET}")
        return 1
    try:
        hand_filter = build_hand_filter(args, robot_cfg.control.dt)
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"{RED}Glove: {exc}{RESET}")
        return 1
    if hand_filter is not None:
        k = hand_filter.params
        print(f"  glove             {hand_filter.glove.info.get('serial')}, "
              + (f"admittance K {k.stiffness:g}, M {k.mass:g}, D {k.d:.3g} ({hand_filter.law_name})"
                 if k else "recorded, no admittance")
              + f"; noise floor {'from ' + str(args.glove_noise) if hand_filter.has_floor else 'none'}")
        print("                    learning its rest when the loop starts: touch nothing; g relearns it")

    move = "home" if args.from_home else "home and approach"
    if not args.yes and not confirm(f"The arm will {move} on t. Proceed?"):
        print("Aborted.")
        close_hand_filter(hand_filter)
        return 1

    recorder = PendingAttrs(goals=goals, inner=EpisodeRecorder(
        args.out, task_cfg, record_hand_joints=args.record_hand,
        hand_side=hand_side(args), action_hz=1.0 / robot_cfg.control.record_dt,
        camera_names=camera_names,
        more_obs_keys=TACTILE_OBS_KEYS if hand_filter is not None else (),
        more_extra_keys=ADMITTANCE_EXTRA_KEYS if hand_filter is not None else ()))
    if recorder.n_saved:
        print(f"  appending to {recorder.n_saved} existing demos")

    # The approach does the homing itself, so the session must not also home on
    # episode start -- it would undo the pose that was just reached.
    args.no_auto_home = True
    arm, server, session = build_session(
        args, recorder=recorder, gripper=gripper, hand_retargeter=retarget, cameras=rig,
        hand_filter=hand_filter)

    approach = GraspApproach(session, arm, gripper, rig.cameras[detect], table,
                             recorder, args)
    session.key_hook = approach.handle_key
    session.event_hook = lambda event, state: False

    sam = None
    try:
        rig.start(lock=True)
        print(f"Cameras started: {', '.join(rig.names)}")
        sam = start_sam(sam_spec, arm, args.sam_pick)
        server.start()
        print_quest_banner(teleop_cfg.server.port)
        info = arm.connect()
        print(f"Connected. PolyScope {info['polyscope']}, {info['safety_status']}.")
        if not connect_hand(gripper):
            return 1
        print(f"Hand connected. {gripper}")
        # Park the fingers before the arm moves, and before servoing: ramp_to
        # blocks, and the RTDE watchdog times out after 0.1 s.
        gripper.ramp_to(gripper.neutral_pose, seconds=PARK_RAMP_S)
        arm.move_home()
        arm.start_servo()
        print("\nPut the headset on and open the page. Press t for the first grasp.\n")
        session.run()
        return 0

    except SafetyAbort as exc:
        print(f"\n{RED}SAFETY ABORT:{RESET} {exc}")
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        # Hand first: it has no e-stop, and arm shutdown can block.
        try:
            gripper.disconnect()
        except Exception as exc:  # noqa: BLE001
            print(f"{RED}hand disconnect failed: {exc}{RESET}")
        arm.close()
        stop_sam(sam)
        server.stop()
        rig.stop()
        close_hand_filter(hand_filter)
        print(f"\n{recorder.summary()}")
        print(f"{approach.n} approach(es) made this session.")
        if recorder.n_saved:
            print(f"Next: python scripts/data/inspect_dataset.py --dataset {args.out}")
        print("Disconnected.")


if __name__ == "__main__":
    raise SystemExit(main())

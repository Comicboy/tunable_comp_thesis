"""Shared plumbing for the entry-point scripts."""

from __future__ import annotations

import argparse
import os
import logging
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quest_teleop2.config import RobotCfg, TaskCfg, TeleopCfg  # noqa: E402


class _Formatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[90m",
        logging.INFO: "\033[0m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;31m",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, "\033[0m")
        return f"{color}{record.getMessage()}\033[0m"


#: Directory holding `left.urdf` and `right.urdf` for the Wuji Hand. Absolute
#: because it lives outside this repo. Override with QUEST_TELEOP_URDF_DIR.
#:
#: Do NOT point this at a different hand model to make it resolve: the limits
#: define the 0..1 normalisation stored in every recorded dataset, so changing
#: them silently reinterprets old data.
URDF_DIR = Path(os.environ.get(
    "QUEST_TELEOP_URDF_DIR",
    "/home/gregoriov/Documents/isaac_rl/urdf/wujihand/urdf",
))

#: Kept so `--urdf` has a visible default in --help. Right is the default hand.
DEFAULT_URDF = URDF_DIR / "right.urdf"


def add_hand_args(parser: argparse.ArgumentParser, urdf: bool = True):
    """Add --left, and optionally --urdf. Resolve with `hand_side` / `hand_urdf`.

    `--left` means the LEFT Wuji Hand: it selects left.urdf, tells the
    retargeter which hand it is solving for, and swaps the operator's roles so
    your left hand drives and your right hand pinches to gate motion.
    """
    parser.add_argument("--left", action="store_true",
                        help="the robot is wearing the LEFT Wuji Hand")
    parser.add_argument("--no-human-limits", action="store_true",
                        help="allow the URDF's full hyperextension range. Off by "
                             "default: the URDF lets the middle joints bend 28 deg "
                             "backwards, which no hand does and which makes the "
                             "pinky look broken")
    parser.add_argument("--native-left-solver", action="store_true",
                        help="with --left, drive with your LEFT hand through the "
                             "vendor's left retargeter. Off by default because that "
                             "path is biased (~16 deg of abduction on a flat hand); "
                             "the default mirrors the right solver instead")
    if urdf:
        parser.add_argument("--urdf", type=Path, default=None,
                            help="Wuji Hand URDF, for the joint limits (default: the "
                                 "mounted Hand 2's own <side>.urdf)")
        parser.add_argument("--hand-current", type=float, default=None, metavar="A",
                            help="per-joint current limit for the hand, amps (driver default "
                                 "0.5; the firmware boots at 1.0; Wuji's own teleop and SDK "
                                 "examples use 1.5)")
    return parser


def hand_side(args: argparse.Namespace) -> str:
    """Which Wuji Hand is on the robot: 'left' or 'right'. Picks the URDF."""
    return "left" if getattr(args, "left", False) else "right"


def mirror_left(args: argparse.Namespace) -> bool:
    """True when a left hand is driven by mirroring the right-hand solver.

    The default for --left. The vendor's native left path is biased -- a flat
    hand solves to ~16 deg of abduction against ~7 deg for the right, and the
    error varies with pose, so it cannot be calibrated out. Solving as a right
    hand and negating the four fingers' abduction reproduces the left hand to
    0.164 mm at the fingertips, so the good solver drives the left hardware.

    The operator keeps their normal roles: RIGHT hand drives, LEFT hand gates.
    """
    return getattr(args, "left", False) and not getattr(args, "native_left_solver", False)


def human_limits(args: argparse.Namespace) -> bool:
    """Whether to clamp the retargeter to anatomical joint floors."""
    return not getattr(args, "no_human_limits", False)


def solve_side(args: argparse.Namespace) -> str:
    """The hand the RETARGETER solves for -- 'right' whenever mirroring."""
    return "right" if mirror_left(args) else hand_side(args)


def drive_side(args: argparse.Namespace) -> str:
    """Which of the operator's hands drives. Mirroring keeps it the right one."""
    return "left" if (getattr(args, "left", False) and not mirror_left(args)) else "right"


#: Hand 2 assets live in the vendored retargeting clone rather than
#: `URDF_DIR`, which holds only the first-generation model.
HAND2_URDF_DIR = (Path(__file__).resolve().parent.parent / "wuji-retargeting"
                  / "wuji_retargeting" / "wuji-description" / "hand2" / "hand2_beta1" / "body" / "urdf")

#: Retargeting configs that point the optimiser at the Hand 2 model. Written
#: from the vendor's AVP config plus their Hand 2 asset/link_naming block --
#: the vendor ships no Hand 2 config for WebXR input.
HAND2_RETARGET_DIR = Path(__file__).resolve().parent.parent / "config" / "retarget"


def hand_urdf(args: argparse.Namespace) -> Path:
    """The URDF to use: an explicit --urdf, else the one matching --left.

    The limits in this file define the 0..1 normalisation of every recorded
    action, so pointing it at a different hand model silently reinterprets
    data recorded against this one.
    """
    explicit = getattr(args, "urdf", None)
    if explicit is not None:
        return Path(explicit)
    return HAND2_URDF_DIR / f"{hand_side(args)}.urdf"


def build_hand(args: argparse.Namespace,
               limits: tuple | None = None, **extra):
    """Construct the gripper for the mounted Wuji Hand 2.

    One place that knows how the hand is built, so every script stays free of
    SDK details.

    `limits` defaults to `urdf_limits(hand_urdf(args))`; pass it when the
    caller has already read them (they are not cheap to parse twice).
    """
    from quest_teleop2.robot.gripper import make_gripper  # noqa: PLC0415
    from quest_teleop2.teleop.hand_retarget import urdf_limits  # noqa: PLC0415

    if limits is None:
        limits = urdf_limits(hand_urdf(args))
    extra.setdefault("handedness", hand_side(args))
    if getattr(args, "hand_current", None) is not None:
        extra.setdefault("effort_limit_a", float(args.hand_current))
    return make_gripper("wuji", limits=limits, **extra)


def connect_hand(hand) -> bool:
    """Connect the hand, reporting a failure as a message rather than a crash.

    A hand that is not on the network is an ordinary operational state -- its
    adapter is unplugged, it is still booting, it faulted -- and the operator
    needs to read what to do about it. An unhandled traceback buries that in
    two screens of stack (the SDK logs its own copy as well), and in a script
    like `go_home` it makes a run that already did its main job look like it
    crashed.

    Returns True on success. On failure prints the reason and returns False;
    the caller decides whether that is fatal.
    """
    try:
        hand.connect()
        return True
    except Exception as exc:  # noqa: BLE001 -- the message IS the output
        print(f"\n\033[31mHand not connected:\033[0m {exc}")
        return False


def retarget_config(args: argparse.Namespace) -> Path:
    """The retargeting config that points the optimiser at the Hand 2 model."""
    return HAND2_RETARGET_DIR / f"adaptive_hand2_{solve_side(args)}.yaml"


def setup_logging(verbose: bool = False) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_Formatter())
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, handlers=[handler])
    # aiohttp logs every request at INFO, which drowns the status line.
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    robot: bool = True,
    teleop: bool = False,
    task: bool = False,
    port: bool = False,
) -> argparse.ArgumentParser:
    """Add the config-override flags, in their own --help section.

    Call this LAST, after the script's own arguments: argparse lists options in
    the order they were added, and these belong at the bottom.

    Only ask for what the script reads. A --port on something that never serves
    the headset, or a --task-config on something that never loads task.yaml, is
    noise in --help and a flag that silently does nothing.
    """
    g = parser.add_argument_group("configuration (rarely needed)")
    if robot:
        g.add_argument("--robot-config", type=Path, default=None,
                       help="override config/robot.yaml")
        g.add_argument("--host", type=str, default=None, help="override the robot IP")
    if teleop:
        g.add_argument("--teleop-config", type=Path, default=None,
                       help="override config/teleop.yaml")
    if task:
        g.add_argument("--task-config", type=Path, default=None,
                       help="override config/task.yaml")
    if port:
        g.add_argument("--port", type=int, default=None,
                       help="override the WebXR server port")
    g.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


def load_configs(args: argparse.Namespace) -> tuple[RobotCfg, TeleopCfg, TaskCfg]:
    # getattr throughout: a script only declares the overrides it actually uses.
    robot = RobotCfg.load(getattr(args, "robot_config", None))
    teleop = TeleopCfg.load(getattr(args, "teleop_config", None))
    task = TaskCfg.load(getattr(args, "task_config", None))
    if getattr(args, "host", None):
        robot.connection.host = args.host
    if getattr(args, "port", None):
        teleop.server.port = args.port
    return robot, teleop, task


def print_quest_banner(port: int) -> None:
    print(
        f"""
  ┌──────────────────────────────────────────────────────────────┐
  │  In the headset, open Meta Quest Browser and go to:          │
  │                                                              │
  │       http://localhost:{port:<38}│
  │                                                              │
  │  If it does not load, re-run:  setup/quest_usb.sh {port:<12}│
  └──────────────────────────────────────────────────────────────┘
"""
    )


def confirm(prompt: str) -> bool:
    """Ask before anything that will physically move the arm."""
    try:
        return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def build_session(args: argparse.Namespace, recorder=None, gripper=None,
                  hand_retargeter=None, cameras=None, hand_filter=None):
    """Wire up the full teleoperation stack.

    Returns (arm, server, session). The caller owns the lifecycle: call
    arm.connect() / arm.move_home() / arm.start_servo(), then session.run(),
    and arm.close() plus server.stop() in a finally block.

    `gripper` and `hand_retargeter` are for the Wuji Hand, which needs the URDF
    limits and a retargeting config that this generic builder should not have to
    know about -- teleop_all.py constructs both and passes them in.
    """
    from quest_teleop2.quest.quest_input import DualHandInput, QuestInput
    from quest_teleop2.quest.webxr_server import WebXRServer
    from quest_teleop2.robot.gripper import make_gripper
    from quest_teleop2.robot.ur5e_arm import UR5eArm
    from quest_teleop2.teleop.session import TeleopSession

    robot_cfg, teleop_cfg, task_cfg = load_configs(args)

    if teleop_cfg.input.mode == "hands":
        quest = DualHandInput(teleop_cfg.gripper,
                              pinch_freezes=teleop_cfg.input.pinch_freezes,
                              drive_side=drive_side(args))
    else:
        quest = QuestInput(teleop_cfg.controller)
    server = WebXRServer(teleop_cfg, quest)
    arm = UR5eArm(robot_cfg)
    if gripper is None:
        gripper = make_gripper(getattr(args, "gripper", "none"))

    session = TeleopSession(
        arm=arm,
        quest=quest,
        server=server,
        robot_cfg=robot_cfg,
        teleop_cfg=teleop_cfg,
        task_cfg=task_cfg,
        gripper=gripper,
        recorder=recorder,
        home_before_episode=not getattr(args, "no_auto_home", False) and recorder is not None,
        hand_retargeter=hand_retargeter,
        cameras=cameras,
        hand_filter=hand_filter,
    )
    return arm, server, session


def add_glove_args(parser: argparse.ArgumentParser) -> None:
    """The tactile glove on the robot hand, and the admittance it can drive."""
    from quest_teleop2.tactile.glove import GLOVE_PYTHON  # noqa: PLC0415

    g = parser.add_argument_group("tactile glove and admittance")
    g.add_argument("--glove", action="store_true",
                   help="read the tactile glove on the robot hand and record its pressure "
                        "and per-finger forces; the hand follows the operator unchanged")
    g.add_argument("--stiffness", type=float, default=None, metavar="K",
                   help="drive the hand through an admittance to the glove's contact forces "
                        "with this stiffness (implies --glove); see config/admittance.yaml")
    g.add_argument("--damping", type=float, default=None, metavar="D",
                   help="admittance damping; default damping_ratio x critical, from the config")
    g.add_argument("--mass", type=float, default=None, metavar="M",
                   help="admittance mass; default from the config")
    g.add_argument("--admittance-config", type=Path, default=Path("config/admittance.yaml"))
    g.add_argument("--glove-regions", type=Path, default=Path("config/glove_regions.yaml"),
                   help="which glove cells belong to which finger "
                        "(scripts/glove/glove_regions.py writes it)")
    g.add_argument("--glove-noise", type=Path, default=Path("config/glove_noise.npz"),
                   help="per-cell floor below which the glove reads zero "
                        "(scripts/glove/glove_noise.py writes it; used when present)")
    g.add_argument("--admittance-law", default=None, metavar="MODULE:CLASS",
                   help="another admittance law, e.g. student_law:StudentAdmittance; "
                        "see quest_teleop2/control/admittance.py for what it must provide")
    g.add_argument("--glove-python", type=Path, default=GLOVE_PYTHON,
                   help="python of the environment that reads the glove (wujihandpy >= 1.7)")


def build_hand_filter(args: argparse.Namespace, dt: float):
    """The glove reader and admittance the flags ask for, already reading; None without --glove.

    Starts learning the glove's rest at once, so nothing may touch it for the first 2 s.
    """
    if not (getattr(args, "glove", False) or getattr(args, "stiffness", None) is not None):
        return None
    from quest_teleop2.control.admittance import AdmittanceParams, TactileAdmittance  # noqa: PLC0415
    from quest_teleop2.tactile.contact import FingerRegions  # noqa: PLC0415
    from quest_teleop2.tactile.glove import GloveClient  # noqa: PLC0415

    params = None
    if args.stiffness is not None:
        params = AdmittanceParams.load(args.admittance_config, stiffness=args.stiffness,
                                       damping=args.damping, mass=args.mass)
        params.check()
    regions = FingerRegions.load(args.glove_regions)
    glove = GloveClient(python=args.glove_python).start()
    try:
        hand_filter = TactileAdmittance(glove, regions, params, dt, law=args.admittance_law,
                                        floor_path=args.glove_noise)
    except Exception:
        glove.close()
        raise
    hand_filter.learn_rest()
    return hand_filter


def close_hand_filter(hand_filter) -> None:
    if hand_filter is not None:
        hand_filter.glove.close()


class KeyBus:
    """Keypresses from the terminal and the video window, in one queue.

    With a live view the operator is watching the window, not the terminal, so
    SPACE and q have to work from whichever has focus. Both producers push
    here and the control loop drains it.
    """

    def __init__(self) -> None:
        import threading  # noqa: PLC0415

        self._keys: list[str] = []
        self._lock = threading.Lock()

    def push(self, key: str) -> None:
        with self._lock:
            self._keys.append(key)

    def drain(self) -> list[str]:
        with self._lock:
            out, self._keys = self._keys, []
        return out


def _detector_overlay(shown, frame, table, label: str):
    """What the reach would use, drawn on the top-down tile: SAM 3's last decision
    when it is the locator, otherwise a fresh run of the cube detector."""
    import cv2  # noqa: PLC0415

    from quest_teleop2.vision.table import active_locator  # noqa: PLC0415
    from scripts.camera.check_scene import find_cube  # noqa: PLC0415

    locator = active_locator()
    if locator is not None:
        last = locator.last
        return locator.draw(shown), label + (f"  {locator.name}: " + (
            "no decision yet" if last is None else "target" if last.target is not None
            else "refused"))
    box, score = find_cube(frame)
    if box is None:
        return shown, label + "  no object"
    x, y, w, h = box
    cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 255, 0), 2)
    xy = table.to_table((x + w / 2.0, y + h / 2.0))
    return shown, label + f"  object ({xy[0]:+.3f}, {xy[1]:+.3f})  score {score:.0f}"


def add_sam_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sam", action=argparse.BooleanOptionalAction, default=True,
                        help="find --object with SAM 3 (the default). --object is a "
                             "config/objects.yaml name or any short phrase, quoted. --no-sam "
                             "uses the background detector, as every dataset recorded before "
                             "2026-11-16 did")
    parser.add_argument("--object-height", type=float, default=None, metavar="M",
                        help="object height in metres; needed when --object is a phrase "
                             "not in config/objects.yaml")
    parser.add_argument("--approach-from-above", action="store_true",
                        help="cross at home height, then descend straight onto the object, "
                             "instead of one straight line; for cluttered tables")
    parser.add_argument("--sam-pick", choices=("one", "best"), default="one",
                        help="when several objects match about equally: refuse (one) or "
                             "take the best score")


def dataset_locators(dataset) -> set[str]:
    """How the reach found the object in each episode of a dataset: 'sam3' or 'background'."""
    import h5py  # noqa: PLC0415

    with h5py.File(dataset, "r") as f:
        names = {str(d.attrs["detector"]) for d in (f["data"].values() if "data" in f else [])
                 if "detector" in d.attrs}
    return {"sam3" if n.startswith("sam3") else "background" for n in names}


def locator_clash(args: argparse.Namespace, dataset) -> str | None:
    """A sentence when `dataset` was reached with another locator than this run will use.

    SAM 3 aims at the object, the background detector at the object and its stand together,
    so the two arrive centimetres apart and their episodes do not mix.
    """
    try:
        used = dataset_locators(dataset)
    except OSError:
        return None
    now = "sam3" if getattr(args, "sam", False) else "background"
    if not used or used == {now}:
        return None
    was = "the background detector" if "background" in used else "SAM 3"
    return (f"{dataset} was reached with {was}, and this run uses "
            f"{'SAM 3' if now == 'sam3' else 'the background detector'}: one aims at the object, "
            f"the other at the object and its stand together.")


def sam_object(args: argparse.Namespace):
    """With --sam: the object's spec, after raising args.standoff by its extra height."""
    if not getattr(args, "sam", False):
        return None
    from quest_teleop2.vision.sam import object_spec  # noqa: PLC0415

    spec, ref = object_spec(args.object, getattr(args, "object_height", None))
    args.object = spec.name
    args.approach_raise_m = max(0.0, spec.height_m - ref.height_m)
    print(f"  SAM 3        {spec.name}: '{spec.prompt}', min score {spec.min_score:.2f}"
          + (f", approach raised {args.approach_raise_m * 1000:.0f} mm over the {ref.name}'s"
             if args.approach_raise_m else ""))
    return spec


#: The approach every dataset so far was recorded with: standoff, +-xy and +z noise, m.
APPROACH_DEFAULTS = (0.04, 0.04, 0.02)


def approach_settings(dataset) -> tuple[float, float, float]:
    """The standoff and noise bounds a dataset was recorded with.

    Stored per episode, so the file is the authority rather than whatever is typed
    months later. Noise bounds are read where recorded, otherwise inferred from the
    sampled noise, rounded up to the cm. Mixed settings are refused: there is no
    single right answer to roll out with.
    """
    import h5py  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    with h5py.File(dataset, "r") as f:
        demos = list(f["data"].values()) if "data" in f else []
        stand = {round(float(d.attrs["standoff_m"]), 4) for d in demos
                 if "standoff_m" in d.attrs}
        bounds = {(round(float(d.attrs["noise_xy_m"]), 4), round(float(d.attrs["noise_z_m"]), 4))
                  for d in demos if "noise_xy_m" in d.attrs and "noise_z_m" in d.attrs}
        noise = np.array([d.attrs["approach_noise_m"] for d in demos
                          if "approach_noise_m" in d.attrs])
    if not stand or not len(noise):
        raise KeyError("no standoff_m / approach_noise_m -- recorded before "
                       "they were logged")
    if len(stand) > 1:
        raise ValueError(f"standoff varies across episodes: {sorted(stand)}")
    if len(bounds) > 1:
        raise ValueError(f"noise bounds vary across episodes: {sorted(bounds)}")
    if bounds:
        return (stand.pop(), *bounds.pop())
    return (stand.pop(),
            float(np.ceil(np.abs(noise[:, :2]).max() * 100) / 100),
            float(np.ceil(noise[:, 2].max() * 100) / 100))


def start_sam(spec, arm, pick: str = "one"):
    """Start SAM 3 and make it answer locate_cube. Returns the client for stop_sam."""
    if spec is None:
        return None
    import time  # noqa: PLC0415

    from quest_teleop2.vision.sam import SamClient, SamLocator  # noqa: PLC0415
    from quest_teleop2.vision.table import set_locator  # noqa: PLC0415

    t = time.monotonic()
    print("Starting SAM 3...", flush=True)
    client = SamClient().start()
    set_locator(SamLocator(client, spec, arm=arm, pick=pick))
    print(f"SAM 3 ready in {time.monotonic() - t:.0f} s.")
    return client


def stop_sam(client) -> None:
    from quest_teleop2.vision.table import set_locator  # noqa: PLC0415

    set_locator(None)
    if client is not None:
        client.close()


class Viewer:
    """Tiled camera view, driven from inside the control loop. No thread.

    Two constraints collide. OpenCV's GUI is only reliable on the main thread,
    and the Wuji SDK refuses any call from a thread other than the one that
    constructed it -- so running the robot on a worker to free main for the GUI
    raises `Thread safety violation` the moment the hand is commanded.

    Neither can be threaded around, so nothing is: `pump()` is called from the
    servo loop, which already runs at 125 Hz on the main thread. The price is
    that the picture stalls during blocking moves -- homing and the approach,
    a few seconds an episode -- because nothing is ticking then. The policy
    phase, which is the part worth watching, updates throughout.
    """

    def __init__(self, rig, table, title: str = "cameras", every: int = 5) -> None:
        self.rig = rig
        self.table = table
        self.title = title
        #: Draw one frame in `every` pumps. At a 125 Hz servo loop, 5 gives
        #: 25 fps, which is the camera rate -- drawing faster would re-display
        #: frames that have not changed.
        self.every = max(1, every)
        self._n = 0

    def pump(self) -> list[str]:
        """Draw if due, and return any keys pressed in the window."""
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        self._n += 1
        if self._n % self.every:
            return []
        tiles = []
        for cam, name in zip(self.rig.cameras, self.rig.names):
            frame, age, _seq = cam.snapshot()
            if frame is None:
                continue
            shown = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
            label = f"{name}  {age * 1000:.0f} ms"
            if self.table is not None and name == self.table.camera:
                shown, label = _detector_overlay(shown, frame, self.table, label)
            cv2.putText(shown, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            tiles.append(shown)
        if tiles:
            h = min(im.shape[0] for im in tiles)
            row = [im if im.shape[0] == h else
                   cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
                   for im in tiles]
            cv2.imshow(self.title, np.hstack(row))
        key = cv2.waitKey(1) & 0xFF
        return [chr(key)] if 32 <= key < 127 else []

    def close(self) -> None:
        try:
            import cv2  # noqa: PLC0415

            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001 -- teardown must not mask a real error
            pass


def view_loop(rig, table, worker, stop, title: str = "cameras",
              bus: "KeyBus | None" = None) -> None:
    """Tile every camera while the robot works. MAIN THREAD ONLY.

    OpenCV's GUI is only reliable on the main thread and the robot's moves
    block, so one of the two has to give it up -- the caller runs the robot on
    `worker` and calls this here.

    The detector is re-run for the overlay rather than reusing a cached box, so
    what is drawn is the same call the approach makes on whatever frame is
    newest. A separate opinion painted over the picture would look
    authoritative while being neither what the robot saw nor what it decided.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    while worker.is_alive():
        tiles = []
        for cam, name in zip(rig.cameras, rig.names):
            frame, age, _seq = cam.snapshot()
            if frame is None:
                continue
            shown = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
            label = f"{name}  {age * 1000:.0f} ms"
            if table is not None and name == table.camera:
                shown, label = _detector_overlay(shown, frame, table, label)
            cv2.putText(shown, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            tiles.append(shown)
        if tiles:
            # Cameras may differ in resolution, so match heights before
            # stacking rather than letting hstack refuse.
            h = min(im.shape[0] for im in tiles)
            row = [im if im.shape[0] == h else
                   cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
                   for im in tiles]
            cv2.imshow(title, np.hstack(row))
        key = cv2.waitKey(30) & 0xFF
        if key != 255:
            ch = chr(key) if 32 <= key < 127 else ""
            if ch == "q" and bus is None:
                print("\n  q pressed -- stopping after the current episode")
                stop.set()
            elif ch and bus is not None:
                bus.push(ch)
    cv2.destroyAllWindows()


def start_video(spec: str, rig, table, tag: str):
    """Film the rig, with the object detector drawn on the top-down camera.

    `spec` is a directory, or "auto" for a timestamped one under recordings/.
    Recording runs on its own threads and never blocks the control loop.

    The overlay is deliberate and applies only to the top-down feed: the point
    of that clip is to show what the detector sees. The wrist feed is left clean
    because the point of THAT one is to show what the policy sees, and a box
    drawn over it would be a claim the policy never made.
    """
    from datetime import datetime  # noqa: PLC0415

    from quest_teleop2.vision.recorder import VideoRecorder  # noqa: PLC0415
    from quest_teleop2.vision.table import cube_overlay  # noqa: PLC0415

    out = (Path("recordings") / f"{datetime.now():%Y-%m-%d_%H%M%S}_{tag}"
           if spec == "auto" else Path(spec))
    recorder = VideoRecorder(rig, out, annotate={table.camera: cube_overlay(table)})
    recorder.start()
    print(f"Filming {', '.join(rig.names)} to {out}")
    return recorder


def stop_video(recorder) -> str:
    from quest_teleop2.vision.recorder import summarise  # noqa: PLC0415

    return summarise(recorder.stop())


def build_camera_rig(spec_names: str | None, config_path=None):
    """(rig, names) for --cameras, or (None, ()) when vision is off.

    `spec_names` is a comma-separated list of names from config/cameras.yaml,
    "" or None for every camera in the file, and "none" to record state-only.
    Rig ORDER always comes from the config file, never from the flag -- see
    CamerasCfg.specs.
    """
    from quest_teleop2.config import CamerasCfg
    from quest_teleop2.vision.camera import CameraRig

    if spec_names and spec_names.lower() == "none":
        return None, ()
    cfg = CamerasCfg.load(config_path)
    wanted = [n.strip() for n in spec_names.split(",")] if spec_names else None
    specs = cfg.specs(wanted)
    rig = CameraRig(specs)
    return rig, tuple(s.name for s in specs)


def controls_help(drive_side: str = "right") -> str:
    """The on-screen control summary, worded for whichever hand is driving."""
    gate = "Left" if drive_side == "right" else "Right"
    drive = drive_side.capitalize()
    return f"""\
  {drive} hand (tracked)
    wrist            drives the TCP
    thumb + index    pinch to close the gripper

  {gate} hand (tracked)
    thumb + index    pinch to FREEZE the arm while you reposition your own
                     arm, release to carry on -- like lifting a mouse

  Keyboard
    SPACE            engage / disengage motion  (starts DISENGAGED)
    t                start an episode (returns to home first)
    s                save episode as a SUCCESS
    f                save episode as a FAILURE (a genuine miss is data)
    r                discard episode (a spoiled take, not a miss)
    [  ]             cycle start position / goal
    h                return to the home pose now
    a                re-anchor without moving the arm
    q                quit

  Safety is the physical e-stop, not software.
"""


#: Right-handed control summary, for --help text that cannot know the flag yet.
CONTROLS_HELP = controls_help("right")


class PolicyApi(NamedTuple):
    """How to call this policy's `step`, and what it needs back from the host."""

    history: bool
    """`step` takes the measured pose and a sample clock beside the anchor."""
    rejected: type
    """What it raises for a sample it refuses. The host holds; nothing is mutated."""
    tick: bool
    """`step` takes the host's own control-tick index."""
    check: bool
    """`check()` reports a dead inference worker, and must be polled."""


def planned_hands(pol, ticks: int):
    """The hand the policy plans for now and the next `ticks` ticks, (ticks + 1, 20).

    Read from the chunks an ensembling ACT loader keeps, blended with its own weights, so row 0
    is the hand it just returned. None for a loader that keeps them elsewhere or plans the hand
    relative to its reading.
    """
    import numpy as np  # noqa: PLC0415

    try:
        chunk, age, t, k = pol._chunk, pol._age, int(pol._t) - 1, int(pol.k)
        m, window, newest = float(pol.m), int(pol.window), bool(pol.weight_newest)
    except AttributeError:
        return None
    if t < 0 or getattr(pol, "hand_rel", False) or not getattr(pol, "use_ensemble", True):
        return None
    rows, offs = [], []
    for s in range(k):
        off = t - int(age[s])
        if age[s] < 0 or not 0 <= off < window:
            continue
        rows.append(np.asarray(chunk[s])[np.minimum(off + np.arange(ticks + 1), k - 1), 6:26])
        offs.append(off)
    if not rows:
        return None
    o = np.asarray(offs, dtype=np.float64)
    w = np.exp(-m * (o if newest else o.max() - o))
    return np.clip(np.einsum("s,stj->tj", w / w.sum(), np.stack(rows)), 0.0, 1.0)


def history_policy_api(pol, module):
    """Work out the calling convention of a loader this repo did not write.

    Loaders arrive as a bundle with their checkpoint, so the runner adapts to
    them rather than the other way round. Everything here is read off the
    object: the 20-D loaders take the anchor positionally and nothing else.
    """
    import inspect  # noqa: PLC0415

    params = inspect.signature(pol.step).parameters
    needs = "sample_time_s" in params
    # The bundles name their own refusal: HistoryInputRejected in the ACT one,
    # DPInputRejected in the diffusion one. Both subclass RuntimeError, so the
    # fallback catches them anyway -- but then it catches every other
    # RuntimeError too, including the ones that mean something is broken.
    exc = RuntimeError
    if needs:
        named = [v for k, v in vars(module).items()
                 if k.endswith("InputRejected") and isinstance(v, type)
                 and issubclass(v, BaseException)]
        if len(named) == 1:
            exc = named[0]
    return PolicyApi(history=needs, rejected=exc,
                     tick="tick" in params, check=callable(getattr(pol, "check", None)))

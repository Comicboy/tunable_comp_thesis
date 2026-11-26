"""Typed configuration loaded from config/*.yaml.

These are dataclasses rather than raw dicts because most of what lives here are
safety limits. A typo in a dict key is a KeyError at the moment of use, which on
a real arm may be mid-motion; a typo here fails at load time.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"

T = TypeVar("T")


def _build(cls: type[T], data: dict[str, Any], path: str = "") -> T:
    """Recursively construct a nested dataclass from a dict, rejecting unknown keys."""
    if not is_dataclass(cls):
        return data  # type: ignore[return-value]

    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        where = f" under '{path}'" if path else ""
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)}{where} in config. "
            f"Valid keys: {sorted(known)}"
        )

    # `from __future__ import annotations` makes Field.type a string, so resolve
    # the real types before testing whether a field is itself a dataclass.
    hints = get_type_hints(cls)

    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in data:
            continue
        value = data[name]
        sub = f"{path}.{name}" if path else name
        ftype = hints.get(name)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = _build(ftype, value, sub)  # type: ignore[arg-type]
        else:
            kwargs[name] = value
    return cls(**kwargs)  # type: ignore[return-value]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


# --------------------------------------------------------------------------- #
# robot.yaml
# --------------------------------------------------------------------------- #


@dataclass
class ConnectionCfg:
    host: str = "192.168.1.102"
    rtde_port: int = 30004
    script_port: int = 30002
    dashboard_port: int = 29999


@dataclass
class ServoJCfg:
    lookahead_time: float = 0.1
    gain: float = 300.0


@dataclass
class ControlCfg:
    frequency_hz: float = 125.0
    #: Rate at which actions are recorded and a policy is stepped. The servo
    #: loop stays at `frequency_hz` -- servoj needs that for smoothness and the
    #: RTDE watchdog -- and one action is taken every `record_decimation` ticks.
    #:
    #: 25 Hz sits above DROID (15) and the Berkeley/NYU dexterous-hand systems
    #: (16-20) and below ALOHA/ACT (50). Coarser steps mean a shorter effective
    #: horizon and less compounding error, which matters with ~80 demos.
    record_hz: float = 25.0
    servoj: ServoJCfg = field(default_factory=ServoJCfg)

    @property
    def dt(self) -> float:
        return 1.0 / self.frequency_hz

    @property
    def record_decimation(self) -> int:
        """Servo ticks per recorded action. 125/25 = 5."""
        return max(1, int(round(self.frequency_hz / self.record_hz)))

    @property
    def record_dt(self) -> float:
        """Seconds per recorded action, as actually achieved after rounding."""
        return self.record_decimation * self.dt


@dataclass
class WorkspaceCfg:
    """Commanded-TCP bounds in the base frame, metres.

    x is forward reach from the base, y is side to side, z is height.

    These defaults are a conservative fallback for a key missing from
    robot.yaml, and are deliberately tighter than the tuned values there. Do
    not "sync" them: a typo that drops a key should shrink the box, not widen
    it to whatever the cell happened to allow last.
    """

    x: list[float] = field(default_factory=lambda: [0.15, 0.65])
    y: list[float] = field(default_factory=lambda: [-0.45, 0.45])
    z: list[float] = field(default_factory=lambda: [0.05, 0.55])

    def as_array(self) -> np.ndarray:
        """(3, 2) array of [min, max] per axis."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def validate(self) -> None:
        for axis in ("x", "y", "z"):
            lo, hi = getattr(self, axis)
            if lo >= hi:
                raise ValueError(f"workspace.{axis} = [{lo}, {hi}] is empty or inverted.")


@dataclass
class SafetyCfg:
    workspace: WorkspaceCfg = field(default_factory=WorkspaceCfg)
    max_linear_speed: float = 0.25
    max_angular_speed: float = 1.0
    max_jump_linear: float = 0.05
    max_jump_angular: float = 0.35
    max_joint_speed: float = 1.0
    max_jump_joint: float = 0.10
    #: Ceiling on how fast the COMMANDED joint speed may change, rad/s^2.
    #: max_joint_speed alone bounds velocity but not its derivative, so a policy
    #: whose opening target is far from home gets clamped to full speed on tick
    #: one -- 0 to 1 rad/s in 8 ms is 125 rad/s^2, several times a UR5e's own
    #: joint acceleration limit, and the robot answers with a PROTECTIVE_STOP.
    #: 4.0 reaches full speed in 0.25 s, which servoj's lookahead absorbs.
    max_joint_accel: float = 4.0
    #: Slack on the workspace ABORT check, metres. The box clamps commanded
    #: targets; the measured TCP lags behind them and overshoots slightly at a
    #: boundary. The recorded demonstrations reach y=+0.4503 and z=0.0994
    #: against limits of 0.45 and 0.10 -- so a zero-tolerance abort fires on
    #: sub-millimetre servo lag rather than on a policy actually leaving the
    #: safe region. This bounds the latter without tripping on the former.
    workspace_tolerance: float = 0.02
    #: Software force guard, N; the pendant's own limit cannot go below 100 N.
    max_tcp_force: float = 30.0
    #: Payload the pendant must report, kg.
    payload_kg: float = 1.01
    payload_tolerance_kg: float = 0.2
    quest_timeout_s: float = 0.2
    rtde_timeout_s: float = 0.5

    def validate(self) -> None:
        self.workspace.validate()
        for name in (
            "max_linear_speed",
            "max_angular_speed",
            "max_jump_linear",
            "max_jump_angular",
            "max_joint_speed",
            "max_jump_joint",
            "max_joint_accel",
            "max_tcp_force",
            "payload_kg",
            "payload_tolerance_kg",
            "quest_timeout_s",
            "rtde_timeout_s",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"safety.{name} must be > 0.")
        if self.max_tcp_force > 100.0:
            raise ValueError(
                f"safety.max_tcp_force={self.max_tcp_force} N is above the pendant's own "
                "100 N limit, so the software guard would never act first."
            )
        # A speed cap above what a human can safely stand next to is almost
        # certainly a typo (m/s vs mm/s), so refuse it rather than obey it.
        if self.max_linear_speed > 1.0:
            raise ValueError(
                f"safety.max_linear_speed={self.max_linear_speed} m/s is above the 1.0 m/s "
                "sanity ceiling for hand teleoperation. Edit the ceiling in config.py "
                "deliberately if you really want this."
            )
        if self.max_joint_speed > 3.14:
            raise ValueError(
                f"safety.max_joint_speed={self.max_joint_speed} rad/s exceeds the sanity "
                "ceiling for a shared workspace."
            )


@dataclass
class RobotCfg:
    connection: ConnectionCfg = field(default_factory=ConnectionCfg)
    control: ControlCfg = field(default_factory=ControlCfg)
    safety: SafetyCfg = field(default_factory=SafetyCfg)
    home_q: list[float] = field(default_factory=lambda: [0.0, -1.57, -1.57, -1.57, 1.57, 0.0])
    home_speed: float = 0.3

    @classmethod
    def load(cls, path: Path | None = None) -> RobotCfg:
        cfg = _build(cls, _load_yaml(path or CONFIG_DIR / "robot.yaml"))
        cfg.validate()
        return cfg

    def validate(self) -> None:
        self.safety.validate()
        if len(self.home_q) != 6:
            raise ValueError(f"home_q must have 6 entries, got {len(self.home_q)}.")
        if not 1.0 <= self.control.frequency_hz <= 500.0:
            raise ValueError("control.frequency_hz must be in [1, 500].")
        if self.control.frequency_hz > 250.0:
            raise ValueError(
                f"control.frequency_hz={self.control.frequency_hz} is too fast to hold "
                "from Python on a non-realtime kernel -- including the generic Linux "
                "kernel this box boots for CUDA. The UR5e bus runs at 500 Hz and "
                "ur_rtde can drive it, but that needs its own tuning. Use 125."
            )


# --------------------------------------------------------------------------- #
# teleop.yaml
# --------------------------------------------------------------------------- #


@dataclass
class ServerCfg:
    port: int = 8080
    send_hz: int = 60


@dataclass
class RetargetCfg:
    xr_to_base: list[list[float]] = field(
        default_factory=lambda: [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
    )
    yaw_offset_deg: float = 0.0
    pos_scale: float = 1.0
    rot_scale: float = 1.0
    deadband_m: float = 0.002
    deadband_rad: float = 0.01

    def rotation_matrix(self) -> np.ndarray:
        """XR -> base change of basis, including the yaw offset."""
        m = np.array(self.xr_to_base, dtype=np.float64)
        if m.shape != (3, 3):
            raise ValueError(f"retarget.xr_to_base must be 3x3, got {m.shape}.")
        # A change of basis has to be a proper rotation, or the mapping mirrors
        # the operator's motion and the arm moves the wrong way.
        if not np.allclose(m @ m.T, np.eye(3), atol=1e-6):
            raise ValueError("retarget.xr_to_base is not orthonormal.")
        if not np.isclose(np.linalg.det(m), 1.0, atol=1e-6):
            raise ValueError(
                f"retarget.xr_to_base has determinant {np.linalg.det(m):.3f}, not +1: "
                "it encodes a reflection, which would mirror your hand motion."
            )
        yaw = np.deg2rad(self.yaw_offset_deg)
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ m


@dataclass
class GripperCfg:
    pinch_close_m: float = 0.030
    pinch_open_m: float = 0.045

    def validate(self) -> None:
        if self.pinch_close_m >= self.pinch_open_m:
            raise ValueError(
                f"gripper.pinch_close_m ({self.pinch_close_m}) must be < "
                f"pinch_open_m ({self.pinch_open_m}); the gap is the hysteresis band."
            )


@dataclass
class ButtonMap:
    trigger: int = 0
    squeeze: int = 1
    thumbstick: int = 3
    a: int = 4
    b: int = 5


@dataclass
class AxisMap:
    thumbstick_x: int = 2
    thumbstick_y: int = 3


@dataclass
class ControllerCfg:
    handedness: str = "left"
    buttons: ButtonMap = field(default_factory=ButtonMap)
    axes: AxisMap = field(default_factory=AxisMap)
    press_threshold: float = 0.6

    @property
    def tracked_hand(self) -> str:
        """The hand that is NOT holding the controller."""
        return "right" if self.handedness == "left" else "left"

    def validate(self) -> None:
        if self.handedness not in ("left", "right"):
            raise ValueError(f"controller.handedness must be 'left' or 'right', got {self.handedness!r}.")


@dataclass
class InputCfg:
    """Which Quest inputs drive the robot.

    This headset reports two tracked hands and no controller in a WebXR
    session, so 'hands' is the working configuration; 'controller' is retained
    for hardware where Meta's Multimodal mode reaches WebXR.
    """

    CLUTCH_MODES = ("left_pinch_release", "left_pinch_hold", "always_on", "controller")

    mode: str = "hands"
    clutch: str = "left_pinch_release"
    #: Motion stays off until the operator explicitly engages it. Without this,
    #: a 'release'-style clutch would have the arm following the instant hand
    #: tracking appears, before the operator is ready.
    start_engaged: bool = False

    @property
    def pinch_freezes(self) -> bool:
        """True when a left pinch STOPS motion rather than enabling it."""
        return self.clutch == "left_pinch_release"

    def validate(self) -> None:
        if self.mode not in ("hands", "controller"):
            raise ValueError(f"input.mode must be 'hands' or 'controller', got {self.mode!r}.")
        if self.clutch not in self.CLUTCH_MODES:
            raise ValueError(
                f"input.clutch must be one of {self.CLUTCH_MODES}, got {self.clutch!r}."
            )
        if self.clutch == "controller" and self.mode != "controller":
            raise ValueError("input.clutch='controller' requires input.mode='controller'.")
        if self.mode == "controller" and self.clutch.startswith("left_pinch"):
            raise ValueError(
                f"input.clutch={self.clutch!r} needs a tracked left hand, which "
                "input.mode='controller' replaces with a controller."
            )


@dataclass
class TeleopCfg:
    server: ServerCfg = field(default_factory=ServerCfg)
    input: InputCfg = field(default_factory=InputCfg)
    retarget: RetargetCfg = field(default_factory=RetargetCfg)
    gripper: GripperCfg = field(default_factory=GripperCfg)
    controller: ControllerCfg = field(default_factory=ControllerCfg)

    @classmethod
    def load(cls, path: Path | None = None) -> TeleopCfg:
        cfg = _build(cls, _load_yaml(path or CONFIG_DIR / "teleop.yaml"))
        cfg.validate()
        return cfg

    def validate(self) -> None:
        self.retarget.rotation_matrix()  # raises if malformed
        self.gripper.validate()
        self.controller.validate()
        self.input.validate()


# --------------------------------------------------------------------------- #
# task.yaml
# --------------------------------------------------------------------------- #


@dataclass
class TaskCfg:
    name: str = "pick_place"
    starts: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    # Written by `record_demos.py --teach`, read by nothing. Declared because
    # _build rejects unknown keys, so task.yaml would fail to load without them.
    start_poses: dict[str, list[float]] = field(default_factory=dict)
    goal_poses: dict[str, list[float]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> TaskCfg:
        cfg = _build(cls, _load_yaml(path or CONFIG_DIR / "task.yaml"))
        cfg.validate()
        return cfg

    @property
    def n_starts(self) -> int:
        return len(self.starts)

    @property
    def n_goals(self) -> int:
        return len(self.goals)

    @property
    def obs_dim(self) -> int:
        """Fixed part (joint_pos + joint_vel + gripper) + the two one-hots.

        Delegates to the schema rather than restating the width. This used to
        hardcode 13 and silently disagreed with `data/schema.py` the moment the
        end-effector grew from 1 DoF to 20.
        """
        from .data.schema import obs_dim  # noqa: PLC0415 -- avoids an import cycle

        return obs_dim(self.n_starts, self.n_goals)

    def validate(self) -> None:
        if not self.starts or not self.goals:
            raise ValueError("task.yaml must define at least one start and one goal.")
        if len(set(self.starts)) != len(self.starts):
            raise ValueError(f"Duplicate start names in task.yaml: {self.starts}")
        if len(set(self.goals)) != len(self.goals):
            raise ValueError(f"Duplicate goal names in task.yaml: {self.goals}")

    def one_hot(self, kind: str, index: int) -> np.ndarray:
        n = self.n_starts if kind == "start" else self.n_goals
        if not 0 <= index < n:
            raise IndexError(f"{kind} index {index} out of range [0, {n}).")
        v = np.zeros(n, dtype=np.float32)
        v[index] = 1.0
        return v


# --------------------------------------------------------------------------- #
# cameras.yaml
# --------------------------------------------------------------------------- #


@dataclass
class CamerasCfg:
    """The camera rig. Order in the file IS rig order, and it is baked in.

    A dataset stores images under `obs/images/<name>` and lists the names in
    `env_args.camera_names`; a checkpoint records the same order in its meta.
    Reordering this file after recording therefore silently swaps the model's
    inputs -- the reason nothing here is keyed off `/dev/videoN`, which is
    assignment order and changes on replug.
    """

    cameras: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None = None) -> CamerasCfg:
        cfg = _build(cls, _load_yaml(path or CONFIG_DIR / "cameras.yaml"))
        cfg.validate()
        return cfg

    @property
    def names(self) -> list[str]:
        return [str(c.get("name", "")) for c in self.cameras]

    def validate(self) -> None:
        if not self.cameras:
            raise ValueError("cameras.yaml defines no cameras.")
        for entry in self.cameras:
            for key in ("name", "serial"):
                if not entry.get(key):
                    raise ValueError(f"Every camera needs a {key!r}: {entry}")
        for label, values in (("name", self.names),
                              ("serial", [c["serial"] for c in self.cameras])):
            if len(set(values)) != len(values):
                raise ValueError(f"Duplicate camera {label}s in cameras.yaml: {values}")

    def specs(self, names: list[str] | None = None) -> list[CameraSpec]:
        """CameraSpecs in RIG order, optionally filtered to `names`.

        Filtering preserves the order in the file rather than the order asked
        for. `--cameras wrist,static` and `--cameras static,wrist` must produce
        the same dataset layout, or the flag becomes a way to silently record an
        incompatible file.
        """
        from .vision.camera import CameraSpec  # noqa: PLC0415 -- keeps config light

        wanted = self.names if names is None else list(names)
        unknown = [n for n in wanted if n not in self.names]
        if unknown:
            raise ValueError(
                f"Unknown camera(s) {unknown}. cameras.yaml defines {self.names}."
            )
        return [_build(CameraSpec, entry) for entry in self.cameras
                if entry["name"] in wanted]


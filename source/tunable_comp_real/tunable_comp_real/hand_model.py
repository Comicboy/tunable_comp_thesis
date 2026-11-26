"""Kinematics of the Wuji Hand 2 (right hand) from its URDF.

Forward kinematics, fingertip Jacobians and pad normals for our Cartesian fingertip admittance,
and the conversion between the hand's normalised joint commands (0..1, what the supervisor's
driver sends and records) and radians, following his driver's rule (quest_teleop2.robot.gripper.
NormalisedHand): 0 = the joint's lower limit, 1 = its upper limit, where the lower limit of every
flexion joint except the thumb's saddle is raised to 0 rad (no hyperextension; his MIN_JOINT_RAD),
and values are clipped to 0..1. On the robot, prefer the connected gripper's own to_radians /
to_normalised; these copies are for code without a gripper object (offline analysis, viewer).

numpy only (no pinocchio, no Isaac Lab), so it runs in the `teleop` env next to the supervisor's
package. Everything is expressed in the hand's root frame `r_wrist`.

Joint order = the device order used by the supervisor's code: finger-major, thumb .. little, and
per finger the four joints in chain order from the palm outwards:
    thumb:  cmc_flex, cmc_abd, mcp, ip
    others: mcp_flex, mcp_abd, pip, dip
i.e. joint 2 of every finger is its abduction (sideways) joint, as in his admittance weights.
(The Isaac model of the original Wuji hand used breadth-first order; nothing here uses it.)

Geometry of the URDF (checked 2026-11-25): every finger extends along the -z axis of its links,
flexion turns about the link's +x axis and moves the fingertip towards +y; the thumb's base sits
on the +y side of the palm. So the palm side of every link is its +y side, and the outward normal
of a fingertip pad is the distal link's +y axis (PAD_AXIS). To be confirmed on the hand.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "little")
#: Link-name prefix of each finger in the URDF.
URDF_PREFIX = {"thumb": "r_thumb", "index": "r_index_finger", "middle": "r_middle_finger",
               "ring": "r_ring_finger", "little": "r_pinky"}
ROOT_LINK = "r_wrist"
N_JOINTS = 20
#: Outward pad normal in the distal link's frame (palm side = +y, see the module docstring).
PAD_AXIS = np.array([0.0, 1.0, 0.0])


def default_urdf() -> Path:
    """The right Hand 2 URDF shipped with the supervisor's code (third_party/quest_teleop_2)."""
    import quest_teleop2  # noqa: PLC0415 -- only to locate his folder

    root = Path(quest_teleop2.__file__).resolve().parents[1]
    return root / "wuji-retargeting" / "wuji_retargeting" / "wuji-description" / "hand2" / "hand2_beta1" \
        / "body" / "urdf" / "right.urdf"


def _min_joint_rad() -> np.ndarray:
    """The supervisor's hyperextension floor per joint (20,), -inf where there is none."""
    from quest_teleop2.robot.gripper import MIN_JOINT_RAD  # noqa: PLC0415

    return np.asarray(MIN_JOINT_RAD, dtype=float).reshape(N_JOINTS)


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF rpy (fixed axes x, y, z): R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation by `angle` about the unit vector `axis` (Rodrigues)."""
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


@dataclass
class Step:
    """One URDF joint along a finger's chain: fixed transform to the joint, then (if revolute) a
    rotation about `axis` by the joint angle."""

    name: str
    origin_R: np.ndarray   # (3, 3)
    origin_p: np.ndarray   # (3,)
    axis: np.ndarray | None  # (3,) unit, in the joint frame; None for a fixed joint


@dataclass
class FingerPose:
    """Forward kinematics of one finger, all in the r_wrist frame."""

    joint_pos: np.ndarray   # (4, 3) origin of each of the finger's 4 joints
    joint_axis: np.ndarray  # (4, 3) unit rotation axis of each joint
    tip_pos: np.ndarray     # (3,) origin of the fingertip frame (*_tip)
    tip_R: np.ndarray       # (3, 3) orientation of the fingertip frame (= distal link)

    @property
    def pad_normal(self) -> np.ndarray:
        """(3,) outward normal of the fingertip pad."""
        return self.tip_R @ PAD_AXIS


class HandModel:
    """Kinematic model of the right Hand 2. `q` is always (20,) radians in device order."""

    def __init__(self, urdf: str | Path | None = None) -> None:
        self.urdf = Path(urdf) if urdf is not None else default_urdf()
        joints = self._parse(self.urdf)
        self.chains: dict[str, list[Step]] = {}
        names, lower, upper = [], [], []
        for finger in FINGERS:
            chain = self._chain(joints, f"{URDF_PREFIX[finger]}_tip")
            revolute = [s for s in chain if s.axis is not None]
            if len(revolute) != 4:
                raise ValueError(f"{finger}: expected 4 revolute joints, found {len(revolute)} in {self.urdf}")
            self.chains[finger] = chain
            for s in revolute:
                names.append(s.name)
                lower.append(joints[s.name]["lower"])
                upper.append(joints[s.name]["upper"])
        # Centre of mass of each distal link, in the fingertip frame (the `point` convention of
        # jacobian()): the contact point PhysX used in sim (tip link COM), here from the URDF.
        coms = self._parse_coms(self.urdf)
        self.distal_com: dict[str, np.ndarray] = {}
        for finger in FINGERS:
            tip_joint = self.chains[finger][-1]               # fixed joint distal link -> *_tip
            distal = joints[tip_joint.name]["parent"]
            self.distal_com[finger] = tip_joint.origin_R.T @ (coms[distal] - tip_joint.origin_p)
        self.joint_names = tuple(names)
        #: URDF joint limits [rad], device order.
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        #: Lower ends of the 0..1 command range [rad]: the URDF's, raised to the supervisor's
        #: hyperextension floor (his MIN_JOINT_RAD, imported so the two cannot drift apart).
        self.command_lower = np.maximum(self.lower, _min_joint_rad())

    # -- URDF ------------------------------------------------------------- #

    @staticmethod
    def _parse_coms(path: Path) -> dict[str, np.ndarray]:
        """Centre of mass of every link in its own frame (URDF <inertial><origin xyz>)."""
        coms = {}
        for link in ET.parse(path).getroot().iter("link"):
            origin = link.find("inertial/origin")
            if origin is not None:
                coms[link.get("name")] = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
        return coms

    @staticmethod
    def _parse(path: Path) -> dict[str, dict]:
        joints = {}
        for j in ET.parse(path).getroot().iter("joint"):
            origin = j.find("origin")
            xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()]) if origin is not None else np.zeros(3)
            rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()] if origin is not None else [0.0] * 3
            axis_el, limit = j.find("axis"), j.find("limit")
            axis = np.array([float(v) for v in axis_el.get("xyz").split()]) if axis_el is not None else None
            joints[j.get("name")] = {
                "type": j.get("type"), "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
                "R": rpy_matrix(*rpy), "p": xyz,
                "axis": axis / np.linalg.norm(axis) if axis is not None else None,
                "lower": float(limit.get("lower")) if limit is not None else np.nan,
                "upper": float(limit.get("upper")) if limit is not None else np.nan,
            }
        return joints

    @staticmethod
    def _chain(joints: dict[str, dict], tip_link: str) -> list[Step]:
        """Joints from the root link to `tip_link`, in order."""
        by_child = {j["child"]: (name, j) for name, j in joints.items()}
        steps, link = [], tip_link
        while link != ROOT_LINK:
            if link not in by_child:
                raise ValueError(f"link {link!r} is not connected to {ROOT_LINK!r}")
            name, j = by_child[link]
            steps.append(Step(name, j["R"], j["p"], j["axis"] if j["type"] == "revolute" else None))
            link = j["parent"]
        return steps[::-1]

    # -- normalised commands <-> radians ---------------------------------- #

    def to_radians(self, u: np.ndarray) -> np.ndarray:
        """Normalised joint values (20,) to radians (20,), as NormalisedHand.to_radians."""
        span = self.upper - self.command_lower
        q = self.command_lower + np.asarray(u, dtype=float).reshape(N_JOINTS) * span
        return np.clip(q, self.command_lower, self.upper)

    def to_normalised(self, q: np.ndarray) -> np.ndarray:
        """Radians (20,) to normalised joint values (20,) in 0..1, as NormalisedHand.to_normalised."""
        span = self.upper - self.command_lower
        return np.clip((np.asarray(q, dtype=float).reshape(N_JOINTS) - self.command_lower) / span, 0.0, 1.0)

    # -- kinematics ------------------------------------------------------- #

    def finger_slice(self, finger: str) -> slice:
        i = FINGERS.index(finger)
        return slice(4 * i, 4 * i + 4)

    def finger(self, q: np.ndarray, finger: str) -> FingerPose:
        """Forward kinematics of one finger at the joint angles q (20,) [rad]."""
        qf = np.asarray(q, dtype=float)[self.finger_slice(finger)]
        R, p = np.eye(3), np.zeros(3)
        pos, axes, k = [], [], 0
        for s in self.chains[finger]:
            p = p + R @ s.origin_p
            R = R @ s.origin_R
            if s.axis is not None:
                pos.append(p.copy())
                axes.append(R @ s.axis)
                R = R @ axis_angle(s.axis, qf[k])
                k += 1
        return FingerPose(np.array(pos), np.array(axes), p, R)

    def fk(self, q: np.ndarray) -> dict[str, FingerPose]:
        """Forward kinematics of all five fingers."""
        return {f: self.finger(q, f) for f in FINGERS}

    def jacobian(self, q: np.ndarray, finger: str, point: np.ndarray | None = None) -> np.ndarray:
        """(3, 4) translational Jacobian of a point on the finger's distal link with respect to the
        finger's own four joints, in the r_wrist frame. `point` is given in the distal (fingertip)
        frame relative to the fingertip origin; None = the fingertip origin itself."""
        pose = self.finger(q, finger)
        x = pose.tip_pos if point is None else pose.tip_pos + pose.tip_R @ np.asarray(point, dtype=float)
        # Revolute joint i moves the point with velocity axis_i x (x - origin_i) per rad.
        return np.stack([np.cross(a, x - o) for a, o in zip(pose.joint_axis, pose.joint_pos)], axis=1)
"""UR5 + right Wuji Hand: environment for the scripted lift experiment.

Same scene as the lift task, with these changes that make it scriptable:

* the arm takes the pose of the grasp point (in the robot base frame) and
  differential IK turns it into joint targets, so a script can say
  "put the hand there" instead of computing joint angles;
* the hand takes absolute finger joint references (not offsets from the default pose);
* everything random or episode-ending is switched off: the cube always starts at
  the same place, no early termination, one long episode;
* the cube and the finger links get a known friction coefficient (FRICTION).
"""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from tunable_comp_thesis.tasks.manager_based.lift_compliant import mdp

from .joint_pos_env_cfg import UR5WujiCubeLiftEnvCfg

EE_OFFSET = (0.04, 0.0, 0.08)
"""Grasp point in the palm frame [m]; the same offset as the ee_frame marker."""

FRICTION = 1.0
"""Static and dynamic friction of the cube and the finger links (rubber-like fingertips).
PhysX combines the two touching materials by averaging, so the contact gets this value too."""


def _friction_event(asset_cfg: SceneEntityCfg) -> EventTerm:
    """Sets a fixed friction on the asset's bodies once, at startup ("randomize" over a range of one value)."""
    return EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": asset_cfg,
            "static_friction_range": (FRICTION, FRICTION),
            "dynamic_friction_range": (FRICTION, FRICTION),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )


@configclass
class UR5WujiCubeScriptedLiftEnvCfg(UR5WujiCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # arm: absolute pose (x, y, z, qw, qx, qy, qz) of the grasp point in the base frame
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_joint", "wrist_.*"],
            body_name="right_palm_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=EE_OFFSET),
        )

        # hand: action = absolute joint angles [rad]; stiffness and controller are set by the script
        self.actions.hand_action.use_default_offset = False

        # a small scene: one env per stiffness value
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5

        # deterministic: cube always at its spawn position, nothing ends the episode early
        self.events.reset_object_position = None

        # known friction on both sides of the grasp contacts
        self.events.cube_friction = _friction_event(SceneEntityCfg("object"))
        self.events.finger_friction = _friction_event(SceneEntityCfg("robot", body_names="right_finger.*"))
        self.terminations.object_dropping = None
        self.episode_length_s = 20.0

        # no policy here: no observation noise, no goal marker
        self.observations.policy.enable_corruption = False
        self.commands.object_pose.debug_vis = False
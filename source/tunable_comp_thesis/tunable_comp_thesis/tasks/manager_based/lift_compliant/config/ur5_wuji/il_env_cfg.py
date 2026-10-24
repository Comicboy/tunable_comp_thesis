"""UR5 + right Wuji Hand: environment for imitation learning (teleoperation, demos, policy rollouts).

Same scene and contact sensing as the lift task, with the choices made for imitation learning:

* start pose: the UR5_CFG joint angles except wrist_2_joint = +pi/2, so the fingers point forward
  (base +x) with the palm facing down; the grasp point starts at about (0.495, 0.109, 0.375) m in
  the base frame, ~35 cm above the table. The robot model itself (assets/ur5.py) is not changed:
  the scripted lift and its results keep the old pose;
* the cube spawns at rest in a 10 x 10 cm square under the hand (CUBE_CENTER +- CUBE_HALF_RANGE),
  faces aligned with the hand (no yaw);
* action, all absolute joint positions [rad], 26 numbers:
      arm  (6):  joint targets, in ARM_JOINTS order (the teleop script runs its own IK;
                 a policy outputs them directly)
      hand (20): finger references q_ref of the FingertipAdmittanceAction (the admittance adds
                 its offset on top), in the action term's joint order;
* known friction (mu = 1) on the cube and the finger links, as in the scripted lift;
* one stiffness K per environment for a whole episode (hand_action.stiffness_values, set by the
  script that creates the env).

Run it with physics on the CPU (--device cpu): with one env the GPU is several times slower, and
demos and policy rollouts must use the same physics device (see claude/notes_imitation_learning.md,
section 6).
"""

import math

from isaaclab.utils import configclass

from tunable_comp_thesis.tasks.manager_based.lift_compliant import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from .joint_pos_env_cfg import UR5WujiCubeLiftEnvCfg
from .scripted_lift_env_cfg import EE_OFFSET, _friction_event  # noqa: F401 (EE_OFFSET re-exported for scripts)

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
"""Arm joints in the order of the arm action (and of the recorded arm actions)."""

START_JOINTS = {"wrist_2_joint": math.pi / 2}
"""Start angles that differ from UR5_CFG [rad]: fingers forward (+x), palm down."""

CUBE_CENTER = (0.50, 0.11)
"""Centre of the cube's spawn square in the base frame [m], under the grasp point at the start."""

CUBE_HALF_RANGE = 0.05
"""Half side of the spawn square [m]: x in [0.45, 0.55], y in [0.06, 0.16]."""

CUBE_REST_Z = 0.022
"""Spawn height of the cube centre [m]: 1 mm above its rest height (42 mm cube on the table at z = 0),
so it does not fall at the start of every episode (the lift task spawns it at 0.055 and lets it drop)."""

EPISODE_LENGTH_S = 60.0
"""Time limit of an episode [s]; long enough for one teleoperated grasp-and-lift."""


@configclass
class UR5WujiCubeLiftILEnvCfg(UR5WujiCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --- start pose: change only the listed joints of the robot's default pose ---------------
        self.scene.robot.init_state = self.scene.robot.init_state.replace(
            joint_pos={**self.scene.robot.init_state.joint_pos, **START_JOINTS}
        )

        # --- actions: absolute joint positions for the arm and the hand --------------------------
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=ARM_JOINTS,
            preserve_order=True,  # keep the ARM_JOINTS order instead of the articulation's
            scale=1.0,
            use_default_offset=False,  # action = target angle, not an offset from the start pose
        )
        self.actions.hand_action.use_default_offset = False  # absolute q_ref, like the scripted lift

        # --- cube: at rest under the hand, randomised in a 10 x 10 cm square at every reset ------
        self.scene.object.init_state.pos = (CUBE_CENTER[0], CUBE_CENTER[1], CUBE_REST_Z)
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                # offsets from init_state.pos (env frame = robot base frame here)
                "pose_range": {"x": (-CUBE_HALF_RANGE, CUBE_HALF_RANGE), "y": (-CUBE_HALF_RANGE, CUBE_HALF_RANGE)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
            },
        )

        # --- known friction on both sides of the grasp contacts ----------------------------------
        self.events.cube_friction = _friction_event(SceneEntityCfg("object"))
        self.events.finger_friction = _friction_event(SceneEntityCfg("robot", body_names="right_finger.*"))

        # --- episodes: long time limit; the "cube fell off the table" termination stays ----------
        self.episode_length_s = EPISODE_LENGTH_S

        # --- one env by default (teleop); scripts can ask for more ------------------------------
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        # --- no observation noise, no markers ---------------------------------------------------
        # (the observations themselves are still the lift task's; the IL state comes in a later step)
        self.observations.policy.enable_corruption = False
        self.commands.object_pose.debug_vis = False
        self.scene.ee_frame.debug_vis = False
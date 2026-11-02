"""UR5 + right Wuji Hand: environment for imitation learning (teleoperation, demos, policy rollouts).

Same scene and contact sensing as the lift task, with the choices made for imitation learning:

* start pose (START_JOINTS): fingers forward (base +x), palm facing down, grasp point at
  (0.495, 0.109, 0.200) m in the base frame, i.e. 20 cm above the table (~16 cm above the cube's top).
  Found with scripts/tools/find_init_pose.py --z 0.20 from the UR5_CFG pose with wrist_2 = +pi/2.
  The robot model itself (assets/ur5.py) is not changed: the scripted lift and its results keep the
  old pose;
* the cube spawns at rest in a 10 x 10 cm square (CUBE_CENTER +- CUBE_HALF_RANGE), 10 cm in front of
  the hand's start position (the hand approaches forward and down; further from the base, so the arm
  is not folded up while grasping), faces aligned with the hand (no yaw);
* action, all absolute joint positions [rad], 26 numbers:
      arm  (6):  joint targets, in ARM_JOINTS order (the teleop script runs its own IK;
                 a policy outputs them directly)
      hand (20): finger joint commands q_cmd, sent to the drives as they are, in the hand action
                 term's joint order;
* the policy performs the fingertip admittance itself: by default the hand action term's admittance
  law is bypassed (admittance_enabled = False), so its action goes to the drives unchanged, while the
  term still smooths the contact forces and reports K for the observations. Only the teleoperation
  (recording) switches the law on: there the operator sends references q_ref, the admittance turns
  them into q_cmd = q_ref + J^+ e (e = f / K), and q_cmd is recorded as the action;
* known friction (mu = 1) on the cube and the finger links, as in the scripted lift;
* one stiffness K per environment for a whole episode (hand_action.stiffness_values, set by the
  script that creates the env);
* observations ("policy" group, ILPolicyCfg, 51 numbers, in this order):
      arm joint angles (6) and velocities (6), ARM_JOINTS order       [rad], [rad/s]
      hand joint angles (20), the hand action's joint order            [rad]   (no hand velocities)
      fingertip forces from the cube (15 = 5 tips x 3), base frame     [N]
      cube position (3), base frame                                    [m]
      log10(K) (1)                                                     (150 N/m -> 2.18, 500 -> 2.70)
* success: the cube's bottom face is held at least SUCCESS_HEIGHT above the table (5 mm tolerance,
  no upper limit) for SUCCESS_HOLD_S without a break; the episode then ends ("success" termination
  term, time_out=False). A light-green plane marks the target height and turns darker green while
  the hold timer runs. The other endings (time limit, cube off the table) are failures.

Run it with physics on the CPU (--device cpu): with one env the GPU is several times slower, and
demos and policy rollouts must use the same physics device (see claude/notes_imitation_learning.md,
section 6).
"""

import math

from isaaclab.utils import configclass

from tunable_comp_thesis.tasks.manager_based.lift_compliant import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
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

START_JOINTS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.6090,
    "elbow_joint": 2.0846,
    "wrist_1_joint": -0.4751,
    "wrist_2_joint": math.pi / 2,
    "wrist_3_joint": 0.0,
}
"""Arm start angles [rad] (override UR5_CFG): fingers forward (+x), palm down, grasp point 20 cm above
the table. From find_init_pose.py --z 0.20 (2026-10-24): reached within 0.46 mm / 0.03 deg; pan and
wrist_3 came out as -0.0000 / -0.0002 and are set to 0. Lift + elbow + wrist_1 ≈ 0 (0.0005), so the palm stays
horizontal (the three axes are parallel)."""

CUBE_CENTER = (0.60, 0.11)
"""Centre of the cube's spawn square in the base frame [m]: in line with the grasp point at the start
(y = 0.11) and 10 cm in front of it (x = 0.495 -> 0.60). Moved from x = 0.50 on 2026-10-27 so the arm
is more stretched while grasping and its joints do not get close to each other."""

CUBE_HALF_RANGE = 0.05
"""Half side of the spawn square [m]: x in [0.55, 0.65], y in [0.06, 0.16]."""

CUBE_HALF_SIZE = 0.021
"""Half the cube's edge [m] (42 mm cube): its centre rests at z = 0.021 in the base frame (table top at 0)."""

CUBE_REST_Z = 0.022
"""Spawn height of the cube centre [m]: 1 mm above its rest height (42 mm cube on the table at z = 0),
so it does not fall at the start of every episode (the lift task spawns it at 0.055 and lets it drop)."""

EPISODE_LENGTH_S = 60.0
"""Time limit of an episode [s]; long enough for one teleoperated grasp-and-lift."""

SUCCESS_HEIGHT = 0.15
"""Target height of the cube's bottom face above the table [m] (decided 2026-10-27): cube centre at
0.171 m in the base frame, just below the grasp point's start height (0.199 m)."""

SUCCESS_TOLERANCE = 0.005
"""[m] The cube still counts as above the target if it sags this far below it (keeps the timer running)."""

SUCCESS_HOLD_S = 3.0
"""[s] How long the cube must stay above the target height without a break (150 env steps at 50 Hz)."""

STIFFNESS_VALUES = (150.0, 500.0)
"""The two stiffnesses K [N/m] of the demo sets (decided 2026-10-27): 150 = the softest value that
does not oscillate in the scripted lift, 500 = clearly stiffer thumb (the fingers' own drives are
~120-160 N/m, so mainly the thumb changes). One K per episode, set by the script that creates the env."""


@configclass
class ILPolicyCfg(ObsGroup):
    """The state the policy sees (and that is recorded), concatenated in this order (51 numbers)."""

    arm_joint_pos = ObsTerm(
        func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS, preserve_order=True)}
    )
    arm_joint_vel = ObsTerm(
        func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS, preserve_order=True)}
    )
    # same joint selection as the hand action (regex, articulation order), so the order matches q_ref
    hand_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["right_finger.*"])})
    fingertip_forces = ObsTerm(func=mdp.fingertip_forces, params={"action_name": "hand_action"})
    cube_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
    stiffness = ObsTerm(func=mdp.stiffness_log10, params={"action_name": "hand_action"})

    def __post_init__(self):
        self.enable_corruption = False  # no noise: the demos record what really happened
        self.concatenate_terms = True


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
        self.actions.hand_action.use_default_offset = False  # absolute joint positions, like the scripted lift
        # the policy performs the admittance itself: the law is off unless a recording script turns it on
        self.actions.hand_action.admittance_enabled = False

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

        # --- success: cube held above the target height for SUCCESS_HOLD_S -> episode ends ------
        self.terminations.success = DoneTerm(
            func=mdp.CubeHeldAboveHeight,
            params={
                "height": SUCCESS_HEIGHT,
                "half_size": CUBE_HALF_SIZE,
                "hold_s": SUCCESS_HOLD_S,
                "tolerance": SUCCESS_TOLERANCE,
                "show_plane": True,  # scripts that need no picture (training, tests) may switch it off
                "plane_center": CUBE_CENTER,
                "plane_size": (0.30, 0.30),
            },
            time_out=False,
        )

        # --- one env by default (teleop); scripts can ask for more ------------------------------
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        # --- the IL state (replaces the lift task's RL observations) -----------------------------
        self.observations.policy = ILPolicyCfg()

        # --- no markers --------------------------------------------------------------------------
        self.commands.object_pose.debug_vis = False
        self.scene.ee_frame.debug_vis = False
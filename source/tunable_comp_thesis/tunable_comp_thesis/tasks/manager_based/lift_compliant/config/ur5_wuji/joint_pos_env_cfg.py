"""UR5 + Wuji Hand (right) configuration for the compliant lift task.

This is the file that specifies the robot related information for the task (lift_env_cfg.py): the robot, its action terms, the
body the pose command tracks, the object, the fingertip contact sensors and the
end-effector frame.
"""

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from tunable_comp_thesis.tasks.manager_based.lift_compliant import mdp
from tunable_comp_thesis.tasks.manager_based.lift_compliant.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from tunable_comp_thesis.assets import UR5_WUJI_RIGHT_CFG  # isort: skip

@configclass
class UR5WujiCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent (sets decimation, sim.dt, PhysX settings)
        super().__post_init__()

        # UR5 arm with the right Wuji Hand on the flange (26 joints, 33 bodies)
        # TODO: Set a task-specific pre-grasp pose (end-effector ~10 cm above the cube spawn
        # centre, palm down) before training. See scripts/tools/find_init_pose.py.
        self.scene.robot = UR5_WUJI_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 6 joint position targets, as offsets from the default pose.
        # TODO: This should be replaced by the admittance controller!
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_joint", "wrist_.*"],
            scale=0.5,
            use_default_offset=True,
        )

        # 20 finger joint position targets
        # since a 20-DoF hand has no single "closed" state as the franka gripper had
        self.actions.hand_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["right_finger.*"],
            scale=1.0,
            use_default_offset=True,
        )

        # Body the object pose command is tracked against
        self.commands.object_pose.body_name = "right_palm_link"

        # Object to lift (same cube as the Franka's)
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Good to remember: sensors take USD prim paths, anything using body_name/joint_names takes articulation names.
        # The two differ for the hand because of the WujiHand Xform in between

        # Fingertip contact sensors with one regex matches all five tip links.
        # IMPORTANT: Requires activate_contact_sensors=True in UR5_WUJI_RIGHT_CFG, or they read zeros!
        self.scene.fingertip_contacts = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/WujiHand/right_finger._tip_link",
            update_period=0.0,
            history_length=6,
        )

        # End-effector frame: grasp point relative to the robot base
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,  # on while checking the offset; set False for training
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/WujiHand/right_palm_link",
                    name="end_effector",
                    # It's important to set the grasp centre right because the reaching reward (object_ee_distance) measures to this frame,
                    # so a wrong offset teaches the policy to bring the wrong point to the cube.
                    offset=OffsetCfg(pos=[0.04, 0.0, 0.08]),   # +x: out of the palm face, z: up the hand toward the fingers
                ),
            ],
        )

"""The end effector frame where the grasp centre should be located was tuned by looking at the simulation
and trying to place the ee marker into the centre of the ball (a bit towards the palm) formed by the curled fingers. For that the initial state of the hand had to be modified to this:
joint_pos={f"{side}_finger.*_joint1": 0.8,
                        f"{side}_finger.*_joint3": 0.8,
                        f"{side}_finger.*_joint4": 0.8},
"""

@configclass
class UR5WujiCubeLiftEnvCfg_PLAY(UR5WujiCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for the Universal Robots UR5 robot.

The following configuration parameters are available:

* :obj:`UR5_CFG`: The UR5 arm without an end effector.

The joints are left at the drive gains authored in the USD, which approximate an
ideal position source.

Sources:
- https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/blob/ros2/config/ur5/joint_limits.yaml
- https://www.universal-robots.com/media/50573/ur5_bz.pdf
"""

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .paths import DATA_DIR

##
# Configuration
##
UR5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(DATA_DIR / "ur5" / "ur5.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=True,
    ),
    # Initial joint angles in rad, applied at spawn and on every reset.
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    # Joint model used for the actuators (PD controller): tau = Kp (q_target - q) + Kd (qd_target - qd), computed
    # Gains found in the usd file (originally stored in degrees):
    #   arm joints    Kp = 57779 N m/rad, Kd = 229 N m s/rad
    #   wrist joints  Kp = 21762 N m/rad, Kd =  87 N m s/rad
    actuators={
        "shoulder_elbow": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_joint"],
            effort_limit_sim=150.0,     # Nm
            velocity_limit_sim=np.pi,   # rad/s
            stiffness=None,             # keep the USD's gains
            damping=None,
        ),
        "arm_wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"],
            effort_limit_sim=28.0,      # Nm
            velocity_limit_sim=np.pi,   # rad/s
            stiffness=None,             # keep the USD's gains
            damping=None,
        ),
    },
)
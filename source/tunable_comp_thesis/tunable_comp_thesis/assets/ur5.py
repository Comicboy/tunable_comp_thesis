# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for the Universal Robots UR5 robot

Sources:
- https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/blob/ros2/config/ur5/joint_limits.yaml
- https://www.universal-robots.com/media/50573/ur5_bz.pdf


"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import numpy as np

##
# Configuration
##

# TODO: Figure out how to derive the damping and stiffness through simulation (if possible) or measure it somehow with the real robot. => Check the textbook for formulas and stuff

UR5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur5/ur5.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg( # This specifies the initial state i.e. all the joint angles of the robot when it is spawned in rad
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    # This describes how the joints behave, all of them are modelled as a PD controller with a spring-damper built into the joints with tau = Kp (q_target - q) + Kd (qd_target - qd)
    actuators={
        "shoulder_elbow": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_joint"],
            effort_limit_sim=150.0,   # Nm
            velocity_limit_sim=np.pi,  # rad/s
            stiffness=800.0,          # NA should be designed as Kp = w^2 J
            damping=40.0,             # NA should be designed as Kd = 2 zeta w J
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"],
            effort_limit_sim=28.0,    # Nm
            velocity_limit_sim=3.14,  # rad/s
            stiffness=400.0,          # placeholder same as above
            damping=20.0,             # placeholder same as above
        ),
    },
)
"""
WujiHand configuration file

Source: https://github.com/wuji-technology/isaaclab-sim/blob/main/run_sim.py
"""


import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


from .paths import DATA_DIR

WUJI_HAND_LEFT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(DATA_DIR / "wuji_hand" / "left" / "wujihand.usd"),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, # It is set to true because the fingers can (and kinda should) collide with each other and the palm
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={"left_finger.*_joint1": 0.06}, # The first joint of each winger is slightly spread, everything else is 0
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["left_finger.*_joint.*"],
            stiffness=None, # Wuji includes the sys identified values for stiffness and damping so we take thouse from  the usd file
            damping=None,
        ),
    },
)

WUJI_HAND_RIGHT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(DATA_DIR / "wuji_hand" / "right" / "wujihand.usd"),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, # It is set to true because the fingers can (and kinda should) collide with each other and the palm
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={"right_finger.*_joint1": 0.06}, # The first joint of each finger is slightly spread, everything else is 0
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["right_finger.*_joint.*"],
            stiffness=None, # Wuji includes the sys identified values for stiffness and damping so we take those from  the usd file
            damping=None,
        ),
    },
)
"""Configuration for the Universal Robots UR5 robot equipped with Wuji hand

The following configuration parameters are available:

* :obj:`UR5_WUJI_RIGHT_CFG`: UR5 with the right Wuji Hand.
* :obj:`UR5_WUJI_LEFT_CFG`: UR5 with the left Wuji Hand.

Both require a combined USD in data/, built manually

Sources:
- https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/blob/ros2/config/ur5/joint_limits.yaml
- https://www.universal-robots.com/media/50573/ur5_bz.pdf
- Source: https://github.com/wuji-technology/isaaclab-sim/blob/main/run_sim.py


"""

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from .paths import DATA_DIR
from .ur5 import UR5_CFG
from .wuji_hand import WUJI_HAND_LEFT_CFG, WUJI_HAND_RIGHT_CFG

def _make_ur5_wuji_cfg(side: str, hand_cfg: ArticulationCfg) -> ArticulationCfg:
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(DATA_DIR / f"ur5_wuji_{side}.usd"),
            rigid_props=UR5_CFG.spawn.rigid_props,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
            ),
            activate_contact_sensors=True
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                **UR5_CFG.init_state.joint_pos,
                **hand_cfg.init_state.joint_pos,
            },
        ),
        actuators={**UR5_CFG.actuators, **hand_cfg.actuators},
    )


UR5_WUJI_RIGHT_CFG = _make_ur5_wuji_cfg("right", WUJI_HAND_RIGHT_CFG)
UR5_WUJI_LEFT_CFG = _make_ur5_wuji_cfg("left", WUJI_HAND_LEFT_CFG)
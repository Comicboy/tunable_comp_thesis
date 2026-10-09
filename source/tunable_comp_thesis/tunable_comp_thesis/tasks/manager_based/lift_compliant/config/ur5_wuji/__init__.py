"""UR5 + right Wuji Hand configurations of the compliant lift task."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Lift-Cube-UR5Wuji-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR5WujiCubeLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR5WujiLiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cube-UR5Wuji-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR5WujiCubeLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR5WujiLiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

# Scripted lift: arm in task space (differential IK), hand with fingertip admittance.
gym.register(
    id="Isaac-Lift-Cube-UR5Wuji-Scripted-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.scripted_lift_env_cfg:UR5WujiCubeScriptedLiftEnvCfg"},
    disable_env_checker=True,
)
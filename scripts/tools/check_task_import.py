from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import gymnasium as gym

# Import directly, so any error is raised instead of swallowed.
import tunable_comp_thesis.tasks.manager_based.lift_compliant.config.ur5_wuji  # noqa: F401

print("registered:", [e for e in gym.registry if "UR5Wuji" in e])

app.close()
import os; os._exit(0)
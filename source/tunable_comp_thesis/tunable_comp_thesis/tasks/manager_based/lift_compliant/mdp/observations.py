"""Observation terms for imitation learning that Isaac Lab does not provide.

The joint angles, joint velocities and the cube position come from Isaac Lab's own terms
(mdp.joint_pos, mdp.joint_vel, mdp.object_position_in_robot_root_frame); these two are specific
to the fingertip admittance:

* fingertip_forces: the contact force on each fingertip, exactly the (smoothed) force the admittance
  law acts on;
* stiffness_log10: the episode's stiffness K, which the policy gets as part of its state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .fingertip_admittance_action import FingertipAdmittanceAction

__all__ = ["fingertip_forces", "stiffness_log10"]


def fingertip_forces(
    env: ManagerBasedRLEnv, action_name: str = "hand_action", robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Contact force on each fingertip from the cube, robot base frame [N], (N, 15) = 5 tips x (fx, fy, fz).

    Exactly the force the admittance law acts on: the hand action's tip_forces_filtered (thumb first,
    finger1..finger5), as of the last physics step. Both "filters" apply:
      * object filter (which contacts count): the five contact sensors only report the force from
        the cube, not from the table or the other fingers;
      * time filter (smoothing): the controller's EMA, f_filt <- alpha f + (1 - alpha) f_filt with
        alpha = 0.3 per physics step (time constant ~28 ms). The physics runs at 100 Hz and the state
        is read at 50 Hz, so the smoothing also keeps single-step contact spikes from being picked up.
    Base frame so that the numbers do not depend on where the robot stands in the world (the UR5 base
    is not rotated here, but done properly).
    """
    hand: FingertipAdmittanceAction = env.action_manager.get_term(action_name)
    robot = env.scene[robot_cfg.name]
    f_w = hand.tip_forces_filtered  # (N, 5, 3), world frame, smoothed like in the admittance law
    q_root = robot.data.root_quat_w.unsqueeze(1).expand(-1, f_w.shape[1], -1)  # (N, 5, 4)
    f_b = quat_apply_inverse(q_root.reshape(-1, 4), f_w.reshape(-1, 3)).reshape(f_w.shape)
    return f_b.reshape(f_w.shape[0], -1)


def stiffness_log10(env: ManagerBasedRLEnv, action_name: str = "hand_action", k_max: float = 1.0e4) -> torch.Tensor:
    """The episode's fingertip stiffness as log10(K / (N/m)), (N, 1).

    Logarithmic because the effect of K is multiplicative (offset e = f / K): 150 -> 2.18, 500 -> 2.70.
    K = inf (admittance off) is clipped to k_max (log10 = 4.0), so the value stays finite for a network.
    Scaling to a fixed range is left to the training code (dataset statistics).
    """
    hand: FingertipAdmittanceAction = env.action_manager.get_term(action_name)
    k = hand.stiffness.clamp(max=k_max)  # (N, 1); inf -> k_max
    return torch.log10(k)
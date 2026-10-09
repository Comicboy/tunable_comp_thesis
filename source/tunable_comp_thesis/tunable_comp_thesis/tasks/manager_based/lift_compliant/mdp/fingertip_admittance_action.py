"""Fingertip admittance action for the Wuji hand.

A script sends finger joint references q_ref, exactly as with
Isaac Lab's JointPositionAction. Every physics step this term then

  1. reads the contact force f on each of the 5 fingertip links       (N, 5, 3)
  2. runs the admittance controller:  e = controller(f, K)            (N, 5, 3)
  3. maps each tip offset to its finger's 4 joints with the damped
     pseudo-inverse of the tip position Jacobian:  dq = J^+ e         (N, 20)
  4. sends q_cmd = q_ref + dq to the joint drives.

The stiffness K is fixed per environment for now (stiffness_values in the cfg).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

from tunable_comp_thesis.controllers import AdmittanceControllerCfg, StaticAdmittanceControllerCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import ContactSensor

__all__ = ["FingertipAdmittanceAction", "FingertipAdmittanceActionCfg"]

NUM_FINGERS = 5
JOINTS_PER_FINGER = 4


class FingertipAdmittanceAction(JointPositionAction):
    """Finger joint position control with a Cartesian admittance at each fingertip."""

    cfg: FingertipAdmittanceActionCfg

    # Reorganise the data since everything works differently:
    # The sensor lists its 5 tip bodies in its own order, the Jacobian has one row per body, 
    # its columns follow the articulation’s joint order, and the action has its own order for the 20 finger joints 
    # (breadth-first: all joint1s, then all joint2s, …)
    def __init__(self, cfg: FingertipAdmittanceActionCfg, env: ManagerBasedEnv):
        # resolves the finger joints, the scale and the default-pose offset, like a normal joint action
        super().__init__(cfg, env)
        side = cfg.side
        self._sensor: ContactSensor = env.scene.sensors[cfg.sensor_name]

        # --- index tables, one entry per finger (finger1 .. finger5) -------------------
        self._sensor_rows = []  # row of this fingertip in the contact sensor data
        self._jac_rows = []  # row of this fingertip in the Jacobian tensor
        self._jac_cols = []  # articulation ids of this finger's 4 joints (Jacobian columns)
        self._action_cols = []  # positions of these 4 joints in this action's joint list
        for f in range(1, NUM_FINGERS + 1):
            tip = f"{side}_finger{f}_tip_link"
            joints = [f"{side}_finger{f}_joint{j}" for j in range(1, JOINTS_PER_FINGER + 1)]

            self._sensor_rows.append(self._sensor.find_bodies(tip)[0][0])

            body_idx = self._asset.find_bodies(tip)[0][0]
            # a fixed-base Jacobian has no row for the root body, so we need to shift all rows by one 
            # (becauee PhysX leaves the fixed root body out of the Jacobian)
            self._jac_rows.append(body_idx - 1 if self._asset.is_fixed_base else body_idx)

            joint_ids, _ = self._asset.find_joints(joints, preserve_order=True)
            # a floating base would add 6 base columns in front; the UR5 is fixed
            self._jac_cols.append(joint_ids if self._asset.is_fixed_base else [i + 6 for i in joint_ids])
            self._action_cols.append([self._joint_names.index(name) for name in joints])

        # --- stiffness per environment: env i gets stiffness_values[i % len] ---------------
        values = torch.tensor(cfg.stiffness_values, dtype=torch.float, device=self.device)
        env_idx = torch.arange(self.num_envs, device=self.device)
        self._stiffness = values[env_idx % len(values)].unsqueeze(1)  # (N, 1)

        # --- the admittance controller (dynamic or static, whichever cfg was given) --------
        self._controller = cfg.controller.class_type(
            cfg.controller, self.num_envs, NUM_FINGERS, env.physics_dt, self.device
        )

        # --- buffers for logging ----------------------------------------------------------
        self._tip_forces = torch.zeros(self.num_envs, NUM_FINGERS, 3, device=self.device)
        self._joint_offsets = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._joint_cmd = torch.zeros(self.num_envs, self._num_joints, device=self.device)

    # ---------------------------------------------------------------------------------
    # read-only views for scripts and logging
    # ---------------------------------------------------------------------------------

    @property
    def joint_names(self) -> list[str]:
        """The finger joints this action drives, in action order."""
        return self._joint_names

    @property
    def stiffness(self) -> torch.Tensor:
        """K per environment [N/m], (N, 1)."""
        return self._stiffness

    @property
    def tip_forces(self) -> torch.Tensor:
        """Contact force on each fingertip, world frame [N], (N, 5, 3), finger1..finger5."""
        return self._tip_forces

    @property
    def tip_offsets(self) -> torch.Tensor:
        """Admittance offset e of each fingertip, world frame [m], (N, 5, 3)."""
        return self._controller.offset

    @property
    def joint_reference(self) -> torch.Tensor:
        """q_ref from the script or policy [rad], (N, 20)."""
        return self.processed_actions

    @property
    def joint_offsets(self) -> torch.Tensor:
        """dq = J^+ e [rad], (N, 20)."""
        return self._joint_offsets

    @property
    def joint_command(self) -> torch.Tensor:
        """q_cmd sent to the drives [rad], (N, 20)."""
        return self._joint_cmd

    # ---------------------------------------------------------------------------------
    # action term interface
    # ---------------------------------------------------------------------------------

    def apply_actions(self):
        """Called every physics step (q_ref itself only changes every env step)."""
        # 1. contact force acting ON each fingertip, world frame (the forces and J are both in world frame)
        f = self._sensor.data.net_forces_w[:, self._sensor_rows, :]  # (N, 5, 3)
        self._tip_forces[:] = f

        # 2. admittance: how far each fingertip should yield
        e = self._controller.compute(f, self._stiffness)  # (N, 5, 3)

        # 3. fingertip offset -> joint offset, finger by finger
        jacobians = self._asset.root_physx_view.get_jacobians()  # (N, bodies, 6, dofs), world frame
        dq = torch.zeros_like(self.processed_actions)
        for i in range(NUM_FINGERS):
            # The Jacobian has 6 rows (linear and angular velocity). The admittance acts only on the tip position, so only the first three are used.
            J = jacobians[:, self._jac_rows[i], :3, :][:, :, self._jac_cols[i]]  # (N, 3, 4) linear part
            dq[:, self._action_cols[i]] = self._damped_pinv_times(J, e[:, i])
        self._joint_offsets[:] = dq

        # 4. Calculate the joint commands, we need to clamp them to keep inside the joint limits
        q_cmd = self.processed_actions + dq
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]  # (N, 20, 2)
        q_cmd = torch.clamp(q_cmd, limits[..., 0], limits[..., 1])
        self._joint_cmd[:] = q_cmd
        self._asset.set_joint_position_target(q_cmd, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._controller.reset(env_ids)

    # ---------------------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------------------

    def _damped_pinv_times(self, J: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """dq = J^T (J J^T + lambda^2 I)^-1 e, batched. J: (N, 3, 4), e: (N, 3) -> (N, 4).

        Smallest joint motion that moves the tip by e; lambda keeps it bounded near
        singular poses (e.g. a straight finger cannot move along its own axis)."""
        lam2 = self.cfg.ik_damping**2
        A = J @ J.transpose(1, 2) + lam2 * torch.eye(3, device=J.device)  # (N, 3, 3)
        x = torch.linalg.solve(A, e.unsqueeze(-1))  # (N, 3, 1)
        return (J.transpose(1, 2) @ x).squeeze(-1)  # (N, 4)


@configclass
class FingertipAdmittanceActionCfg(JointPositionActionCfg):
    """Configuration for :class:`FingertipAdmittanceAction`.

    Inherits joint_names, scale, offset and use_default_offset from JointPositionActionCfg.
    """

    class_type: type = FingertipAdmittanceAction

    side: str = "right"
    """Which hand: selects the {side}_finger{1..5}_tip_link bodies and joints."""

    sensor_name: str = "fingertip_contacts"
    """Name of the contact sensor on the 5 fingertip links in the scene."""

    controller: AdmittanceControllerCfg | StaticAdmittanceControllerCfg = AdmittanceControllerCfg()
    """Which admittance law, and its parameters."""

    stiffness_values: list[float] = [float("inf")]
    """K [N/m]; env i gets stiffness_values[i % len]. inf = admittance off (plain joint control)."""

    ik_damping: float = 0.01
    """lambda of the damped pseudo-inverse [m]."""
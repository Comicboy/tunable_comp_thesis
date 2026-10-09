"""Discrete-time admittance controller (implemented in task space).

For P points (the fingertips) in each of N environments the script integrates with the semi-implicit Euler method:

    M e'' + D e' + K e = f,    D = 2 sqrt(K M)  (critical damping, fixed), where e = x_cmd - x_ref and x_cmd = x_ref + e (if we assume perfect tracking then x_cmd = x)

and returns e, how far each point should yield from its reference. At rest e = f / K so the stiffness is the only quantity that modifies the robot's behaviour.
(M and the critical damping only shape how fast and how smoothly it gets there) The job of the action term is to map e to joint commands

Basically what is going on:
  - In order to grip an object a gripper (or a the fingertips) have to go to a reference position which is inside the object itself
    because if x_ref would be exactly at the objects surface there wouldn't be any friction force, it would just slip out of the grip
  - When we command the robot (the hand and more specifrically the fingertips) to grab the cube the fingertips receive an x_ref value which is inside the cube
  - As the fingertip reaches the cube a contact force is generated. Normally, without any admittance controller the contact force (K_drive*(x_ref-x_object))
    would be determined only by the overlap and the stiffness of the hardware itself which in our case means the stiffness of the joint drives.
  - The admittance controller lets us influence the contact force through the tunable stiffness parameter and it is achieved by the commanded position 
    that is produced by the controller. With the controller we can picture this setup as if we were to put a spring to the and of the fingertip which compresses
    upon contact with the object and pushes the fingertip back. The ammount of compression in this case will be equal the ammount the finger would have pushed into the object
    which is eaxactly e = x - x_ref (in this convention the positive direction aligns with the contact force and not with the force exerted by the fingertip)
    (If we assume perfect position control i.e. x_cmd = x, if it is not the case then the two springs share the compression, and the tip stiffness becomes K_r = (1/K + 1/K_drive)^-1)
  - Therefore, the commanded poistion (x_cmd = x_ref + e) will be located (in steady state when there is no velocity and acceleration thus damping and inertia)
    exactly at the point where the imaginary spring would push back the fingertip from the reference position (i.e. where the virtual spring would balance out the contact force)
    with the amount of "pushback" being equal to e.
"""

from __future__ import annotations
from collections.abc import Sequence
from isaaclab.utils import configclass
import torch
import math

class AdmittanceController:
    """Batched admittance law for num_points (5 becasue of the 5 fingertips) 3-D points per environment."""

    def __init__(self, cfg: AdmittanceControllerCfg, num_envs: int, num_points: int, dt: float, device: str):
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_points = num_points
        self.dt = dt
        self.device = torch.device(device)
        shape = (num_envs, num_points, 3)
        self._e = torch.zeros(shape, device=self.device)       # offset [m]
        self._e_dot = torch.zeros(shape, device=self.device)   # offset velocity [m/s]
        self._f_filt = torch.zeros(shape, device=self.device)  # filtered force [N]

    @property
    def offset(self) -> torch.Tensor:
        """Current offset e [m], shape (N, P, 3)."""
        return self._e

    @property
    def max_stable_stiffness(self) -> float:
        """Stiffness above which the Euler update goes unstable [N/m].
        Critically damped semi-implicit Euler is stable while w*dt < 2(sqrt(2) - 1) = 0.83."""
        w_max = 2.0 * (math.sqrt(2.0) - 1.0) / self.dt
        return self.cfg.mass * w_max**2

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Clear the state of the given environments (all if None)."""
        idx = slice(None) if env_ids is None else env_ids
        self._e[idx] = 0.0
        self._e_dot[idx] = 0.0
        self._f_filt[idx] = 0.0

    def compute(self, f_ext: torch.Tensor, stiffness: torch.Tensor) -> torch.Tensor:
        """One step of the admittance dynamics.

        Args:
            f_ext: force the environment applies to each point [N], (N, P, 3).
            stiffness: [N/m], (N, 1) one per env or (N, P) one per point.
                inf = admittance off for that entry.

        Returns:
            offset e [m], (N, P, 3).
        """
        cfg = self.cfg
        stiffness = torch.as_tensor(stiffness, dtype=self._e.dtype, device=self.device)
        K = stiffness.reshape(self.num_envs, -1, 1).expand(self.num_envs, self.num_points, 3)

        # inf = rigid: compute with a dummy K, then force e to zero (inf * 0 would be NaN)
        rigid = ~torch.isfinite(K)
        K = torch.where(rigid, torch.ones_like(K), K)

        # low-pass the force
        a = cfg.force_filter_alpha
        self._f_filt = a * f_ext + (1.0 - a) * self._f_filt

        # semi-implicit Euler: velocity first, then position with the NEW velocity
        M = cfg.mass
        D = 2.0 * torch.sqrt(K * M)  # critical damping with zeta = 1, follows K automatically
        e_ddot = (self._f_filt - D * self._e_dot - K * self._e) / M
        self._e_dot = self._e_dot + self.dt * e_ddot
        self._e = self._e + self.dt * self._e_dot

        # clamp |e| per point, keeping its direction; stop the velocity at the clamp
        norm = self._e.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        scale = torch.clamp(cfg.max_offset / norm, max=1.0)
        self._e = self._e * scale
        self._e_dot = torch.where(scale < 1.0, torch.zeros_like(self._e_dot), self._e_dot)

        # rigid entries never yield
        self._e = torch.where(rigid, torch.zeros_like(self._e), self._e)
        self._e_dot = torch.where(rigid, torch.zeros_like(self._e_dot), self._e_dot)
        return self._e

@configclass
class AdmittanceControllerCfg:
    """Admittance parameters. The stiffness is not defined here rather it is passed to every
    compute() call, so it can differ per environment and change at runtime."""

    class_type: type = AdmittanceController
    """The controller class this cfg builds (lets the action term use either variant)."""

    mass: float = 0.3
    """Virtual mass M [kg]. Sets the speed of the response, w = sqrt(K/M)."""

    max_offset: float = 0.03
    """Clamp on |e| per point [m]."""

    force_filter_alpha: float = 0.3
    """Low-pass on the force: weight of the newest sample (1.0 = no filtering)."""
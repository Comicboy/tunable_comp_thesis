"""Static admittance controller: e = f / K, no virtual mass, no damper a.k.a. the Lorenzo version.

The offset jumps straight to the steady-state value of the dynamic law for the
current (low-passed) force.

Caveat: closed through the contact, the update is stable only while
K_drive / K < 2 / alpha - 1 (alpha = force_filter_alpha), i.e. only for stiffness
above min_stable_stiffness(K_drive). Without filtering (alpha = 1) that means
K > K_drive: it can then only render stiffness the drives already have.

Same interface as AdmittanceController, so the action term can use either one.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils import configclass


class StaticAdmittanceController:
    """Batched static admittance law for num_points 3-D points per environment."""

    def __init__(self, cfg: StaticAdmittanceControllerCfg, num_envs: int, num_points: int, dt: float, device: str):
        # dt is not used (no dynamics); it is accepted so both controllers are built the same way
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_points = num_points
        self.dt = dt
        self.device = torch.device(device)
        shape = (num_envs, num_points, 3)
        self._e = torch.zeros(shape, device=self.device)  # offset [m]
        self._f_filt = torch.zeros(shape, device=self.device)  # filtered force [N]

    @property
    def offset(self) -> torch.Tensor:
        """Current offset e [m], shape (N, P, 3)."""
        return self._e

    def min_stable_stiffness(self, drive_stiffness: float) -> float:
        """Lowest K for which the contact loop is stable, given the drives' stiffness at the tip [N/m].

        Quasi-static contact f = K_drive (overlap - e) and the filter give
        e_{k+1} = (1 - alpha (1 + K_drive / K)) e_k + const, stable while |...| < 1."""
        return drive_stiffness / (2.0 / self.cfg.force_filter_alpha - 1.0)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Clear the state of the given environments (all if None)."""
        idx = slice(None) if env_ids is None else env_ids
        self._e[idx] = 0.0
        self._f_filt[idx] = 0.0

    def compute(self, f_ext: torch.Tensor, stiffness: torch.Tensor) -> torch.Tensor:
        """One step: e = filtered force / K.

        Args:
            f_ext: force the environment applies to each point [N], (N, P, 3).
            stiffness: [N/m], (N, 1) one per env or (N, P) one per point.
                inf = admittance off for that entry.

        Returns:
            offset e [m], (N, P, 3).
        """
        cfg = self.cfg
        K = stiffness.reshape(self.num_envs, -1, 1).expand(self.num_envs, self.num_points, 3)

        # inf = rigid: divide by a dummy K, then force e to zero
        rigid = ~torch.isfinite(K)
        K = torch.where(rigid, torch.ones_like(K), K)

        # low-pass the force (this filter is also what keeps the contact loop stable)
        a = cfg.force_filter_alpha
        self._f_filt = a * f_ext + (1.0 - a) * self._f_filt

        # the law itself
        self._e = self._f_filt / K

        # clamp |e| per point, keeping its direction
        norm = self._e.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        self._e = self._e * torch.clamp(cfg.max_offset / norm, max=1.0)

        # rigid entries never yield
        self._e = torch.where(rigid, torch.zeros_like(self._e), self._e)
        return self._e


@configclass
class StaticAdmittanceControllerCfg:
    """Static admittance parameters. The stiffness is passed to every compute() call."""

    class_type: type = StaticAdmittanceController
    """The controller class this cfg builds (lets the action term use either variant)."""

    max_offset: float = 0.03
    """Clamp on |e| per point [m]."""

    force_filter_alpha: float = 0.3
    """Low-pass on the force: weight of the newest sample (1.0 = no filtering)."""
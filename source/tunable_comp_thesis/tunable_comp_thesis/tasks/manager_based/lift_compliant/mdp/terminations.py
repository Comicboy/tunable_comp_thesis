"""Termination terms for imitation learning that Isaac Lab does not provide.

* CubeHeldAboveHeight: the success condition of the cube lift, i.e. the cube is held at or above a
  target height for a while. Optionally shows the target height as a grid of thin green lines that
  turns darker green while the cube is above it (visual feedback for the teleoperator).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import TerminationTermCfg

__all__ = ["CubeHeldAboveHeight"]

PLANE_WAITING_COLOR = (0.55, 0.95, 0.55)
"""Light green: the cube is below the target height."""

PLANE_HOLDING_COLOR = (0.0, 0.28, 0.05)
"""Darker green: the cube is above the target height and the hold timer is running."""

PLANE_LINE_WIDTH = 0.0015
"""[m] Width of the grid lines (thin flat bars)."""

PLANE_LINE_THICKNESS = 0.001
"""[m] Vertical thickness of the grid lines."""


class CubeHeldAboveHeight(ManagerTermBase):
    """True when the cube has been at or above a target height for hold_s seconds without a break.

    The height is measured in the robot base frame, where the table top is at z = 0, at the cube's
    bottom face: z_bottom = z_centre - half_size (exact while the cube is level; with a tilted cube
    it is the bottom of a level cube with the same centre). Every env step:

        above   = z_bottom >= height - tolerance
        steps   = steps + 1   if above,   0 otherwise             (a dip restarts the count)
        success = steps >= round(hold_s / step_dt)                (3 s at 50 Hz = 150 steps)

    The hold is counted in whole env steps, not by adding up step_dt: in float32, 150 x 0.02 adds up
    to 2.999998 < 3, which made the first version end one step (20 ms) late.

    The tolerance keeps a small sag in the grasp from restarting the timer; there is no upper limit.
    The timer is cleared when the env resets (whatever the reason).

    Use it as a termination term with time_out=False: the episode then ends as "terminated", and
    env.termination_manager.get_term("success") tells a recording script that it ended successfully.

    The "plane" (show_plane=True) is drawn at the target height itself (not height - tolerance),
    centred at plane_center (x, y) in the base frame, as a square grid of plane_lines x plane_lines
    thin bars (edges included). A grid instead of a filled, semi-transparent square because the RTX
    real-time renderer drew the transparent square fully opaque (also with fractional cutout opacity
    on), hiding the cube below it; through a grid the cube stays visible. It is a visualization
    marker: no physics, no collision, not part of the state. Light green while waiting, darker green
    while the timer runs.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)  # env steps above, current run
        self._above = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._plane: VisualizationMarkers | None = None  # created at the first call (needs the params)
        self._line_pos_b: torch.Tensor | None = None  # (L, 3) line centres, base frame
        self._line_kind: torch.Tensor | None = None  # (L,) 0 = runs along x, 1 = runs along y

    @property
    def hold_time(self) -> torch.Tensor:
        """Time the cube has been above the target height so far [s], (N,). 0 while below."""
        return self._steps * self._env.step_dt

    @property
    def above(self) -> torch.Tensor:
        """Whether the cube is at or above the target height (minus the tolerance), (N,) bool."""
        return self._above

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._steps[env_ids] = 0
        self._above[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        height: float,
        half_size: float,
        hold_s: float,
        tolerance: float = 0.005,
        show_plane: bool = True,
        plane_center: tuple[float, float] = (0.0, 0.0),
        plane_size: tuple[float, float] = (0.30, 0.30),
        plane_lines: int = 5,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        robot, cube = env.scene[robot_cfg.name], env.scene[object_cfg.name]

        # cube centre in the base frame -> height of its bottom face above the table
        p_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, cube.data.root_pos_w)
        z_bottom = p_b[:, 2] - half_size

        self._above[:] = z_bottom >= height - tolerance
        self._steps[:] = torch.where(self._above, self._steps + 1, torch.zeros_like(self._steps))
        success = self._steps >= round(hold_s / env.step_dt)

        if show_plane:
            if self._plane is None:
                self._create_plane(plane_center, plane_size, height, plane_lines)
            self._draw_plane(robot)
        return success

    def _create_plane(self, center: tuple[float, float], size: tuple[float, float], height: float, lines: int):
        """Four bar shapes: along x / along y, each waiting (light) and holding (dark).

        Marker index = kind + 2 * above: 0 waiting along x, 1 waiting along y, 2 holding along x,
        3 holding along y.
        """
        sx, sy = size
        looks = {"waiting": PLANE_WAITING_COLOR, "holding": PLANE_HOLDING_COLOR}
        markers = {}
        for name, color in looks.items():
            material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
            markers[f"{name}_x"] = sim_utils.CuboidCfg(
                size=(sx + PLANE_LINE_WIDTH, PLANE_LINE_WIDTH, PLANE_LINE_THICKNESS), visual_material=material
            )
            markers[f"{name}_y"] = sim_utils.CuboidCfg(
                size=(PLANE_LINE_WIDTH, sy + PLANE_LINE_WIDTH, PLANE_LINE_THICKNESS), visual_material=material
            )
        self._plane = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/SuccessPlane", markers=markers))

        # line centres in the base frame: `lines` bars along x at evenly spaced y, and the same along y
        u = torch.linspace(-0.5, 0.5, lines, device=self.device)
        cx, cy = center
        along_x = torch.stack([torch.full_like(u, cx), cy + u * sy, torch.full_like(u, height)], dim=-1)
        along_y = torch.stack([cx + u * sx, torch.full_like(u, cy), torch.full_like(u, height)], dim=-1)
        self._line_pos_b = torch.cat([along_x, along_y])  # (2 * lines, 3)
        self._line_kind = torch.cat([torch.zeros(lines), torch.ones(lines)]).long().to(self.device)

    def _draw_plane(self, robot):
        """Place every env's grid (base frame -> world, every step) and pick each line's colour."""
        n, n_lines = self.num_envs, self._line_pos_b.shape[0]
        q = robot.data.root_quat_w.unsqueeze(1).expand(n, n_lines, 4).reshape(-1, 4)  # (N * L, 4)
        p_b = self._line_pos_b.unsqueeze(0).expand(n, n_lines, 3).reshape(-1, 3)
        pos_w = robot.data.root_pos_w.repeat_interleave(n_lines, dim=0) + quat_apply(q, p_b)
        idx = (self._line_kind.unsqueeze(0) + 2 * self._above.long().unsqueeze(1)).reshape(-1)  # (N * L,)
        self._plane.visualize(translations=pos_w, orientations=q, marker_indices=idx)
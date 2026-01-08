# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command generators for height-tracking tasks."""

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


@configclass
class UniformHeightCommandCfg(CommandTermCfg):
    """Configuration for uniform height command generator."""

    class_type: type = MISSING

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    sensor_name: str | None = None
    """Optional name of a ray-caster sensor used to estimate terrain height."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the height commands."""

        height: tuple[float, float] = MISSING
        """Range for the height command above terrain (in m)."""

    ranges: Ranges = MISSING
    """Ranges for the commands."""


class UniformHeightCommand(CommandTerm):
    """Command generator for generating height commands uniformly."""

    cfg: UniformHeightCommandCfg

    def __init__(self, cfg: UniformHeightCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.sensor: RayCaster | None = env.scene.sensors.get(cfg.sensor_name) if cfg.sensor_name else None

        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.target_height_w = torch.zeros(self.num_envs, device=self.device)
        self.metrics["height_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformHeightCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired height command. Shape is (num_envs, 1)."""
        return self.height_command

    def _get_ground_height(self) -> torch.Tensor:
        if self.sensor is None:
            return self.robot.data.default_root_state[:, 2]
        return torch.mean(self.sensor.data.ray_hits_w[..., 2], dim=1)

    def _update_target_height(self):
        ground_height = self._get_ground_height()
        self.target_height_w[:] = ground_height + self.height_command.squeeze(1)

    def _update_metrics(self):
        self._update_target_height()
        self.metrics["height_error"] = torch.abs(self.robot.data.root_pos_w[:, 2] - self.target_height_w)

    def _resample_command(self, env_ids):
        r = torch.empty(len(env_ids), device=self.device)
        self.height_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.height)

    def _update_command(self):
        self._update_target_height()


UniformHeightCommandCfg.class_type = UniformHeightCommand

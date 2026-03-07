# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command generators for height-tracking tasks."""

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class BoxCenterCommandCfg(CommandTermCfg):
    """Configuration for box-center position command generator."""

    class_type: type = MISSING

    asset_name: str = MISSING
    """Name of the box asset in the environment."""

    z_offset: float = 0.0
    """Optional z-offset added to the box center."""


class BoxCenterCommand(CommandTerm):
    """Command generator that tracks the box center position."""

    cfg: BoxCenterCommandCfg

    def __init__(self, cfg: BoxCenterCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.box = env.scene[cfg.asset_name]
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["pos_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pos_error_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pos_error_z"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "BoxCenterCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired box center position in world frame. Shape is (num_envs, 3)."""
        return self.target_pos

    def _update_metrics(self):
        robot = self._env.scene["robot"]
        pos_error = robot.data.root_pos_w - self.target_pos
        self.metrics["pos_error"] = torch.norm(pos_error, dim=1)
        self.metrics["pos_error_xy"] = torch.norm(pos_error[:, :2], dim=1)
        self.metrics["pos_error_z"] = torch.abs(pos_error[:, 2])

    def _resample_command(self, env_ids):
        if len(env_ids) == 0:
            return
        self.target_pos[env_ids] = self.box.data.root_pos_w[env_ids]
        self.target_pos[env_ids, 2] += self.cfg.z_offset

    def _update_command(self):
        self.target_pos[:] = self.box.data.root_pos_w
        self.target_pos[:, 2] += self.cfg.z_offset


BoxCenterCommandCfg.class_type = BoxCenterCommand


def command_target_pos_b(
    env: ManagerBasedEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return target position command expressed in the robot base frame."""
    asset = env.scene[asset_cfg.name]
    command_w = env.command_manager.get_command(command_name)
    rel_target_w = command_w - asset.data.root_pos_w
    return quat_apply_inverse(asset.data.root_quat_w, rel_target_w)


def command_time_to_go(env: ManagerBasedEnv) -> torch.Tensor:
    """Return remaining episode time as a single observation term."""
    remaining_s = (env.max_episode_length - env.episode_length_buf).float() * env.step_dt
    return remaining_s.unsqueeze(1)

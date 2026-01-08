# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def lin_vel_x_range(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    min_start: float,
    min_end: float,
    max_start: float,
    max_end: float,
    start_step: int,
    end_step: int,
) -> float:
    """Linearly ramp the commanded x-velocity range over training steps."""
    if end_step <= start_step:
        progress = 1.0
    else:
        progress = (env.common_step_counter - start_step) / float(end_step - start_step)
        progress = max(0.0, min(1.0, progress))

    min_vel = min_start + (min_end - min_start) * progress
    max_vel = max_start + (max_end - max_start) * progress

    cmd = env.command_manager.get_term(command_name)
    cmd.cfg.ranges.lin_vel_x = (min_vel, max_vel)
    return max_vel


def height_command_range(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    min_start: float,
    min_end: float,
    max_start: float,
    max_end: float,
    start_step: int,
    end_step: int,
) -> float:
    """Linearly ramp the commanded height range over training steps."""
    if end_step <= start_step:
        progress = 1.0
    else:
        progress = (env.common_step_counter - start_step) / float(end_step - start_step)
        progress = max(0.0, min(1.0, progress))

    min_height = min_start + (min_end - min_start) * progress
    max_height = max_start + (max_end - max_start) * progress

    cmd = env.command_manager.get_term(command_name)
    cmd.cfg.ranges.height = (min_height, max_height)
    return max_height

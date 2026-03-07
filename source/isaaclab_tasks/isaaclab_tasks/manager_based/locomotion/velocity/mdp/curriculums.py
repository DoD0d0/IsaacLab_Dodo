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


def box_height_steps(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg,
    local_xy: tuple[float, float],
    start_height: float,
    step_height: float,
    max_height: float,
    step_interval: int,
    start_step: int = 0,
) -> float:
    """Increase the box height in discrete steps based on the common step counter."""
    if step_interval <= 0:
        step_index = 0
    else:
        step_index = (env.common_step_counter - start_step) // step_interval
        step_index = max(0, step_index)
    target_height = min(start_height + step_height * float(step_index), max_height)

    box = env.scene[asset_cfg.name]
    env_origins = env.scene.env_origins[env_ids]
    target_pos = env_origins.clone()
    target_pos[:, 0] += local_xy[0]
    target_pos[:, 1] += local_xy[1]
    target_pos[:, 2] += target_height
    target_pose = torch.cat([target_pos, box.data.root_quat_w[env_ids]], dim=1)
    box.write_root_pose_to_sim(target_pose, env_ids=env_ids)
    return target_height


def box_height_on_reach(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | slice,
    asset_cfg: SceneEntityCfg,
    local_xy: tuple[float, float],
    start_height: float,
    step_height: float,
    max_height: float,
    reach_threshold: float,
    min_steps: int = 1,
    asset_target_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    z_offset: float = 0.0,
) -> float:
    """Increase box height when the target position is reached."""
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    if not hasattr(env, "_box_height_levels"):
        env._box_height_levels = torch.zeros(env.num_envs, device=env.device)
        env._box_reach_counts = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    levels = env._box_height_levels
    reach_counts = env._box_reach_counts

    max_level = int(max(0.0, (max_height - start_height) / step_height))
    levels.clamp_(0, max_level)

    box = env.scene[asset_cfg.name]
    env_origins = env.scene.env_origins[env_ids]
    target_height = start_height + levels[env_ids] * step_height
    target_height = torch.clamp(target_height, max=max_height)
    target_pos = env_origins.clone()
    target_pos[:, 0] += local_xy[0]
    target_pos[:, 1] += local_xy[1]
    target_pos[:, 2] += target_height
    target_pose = torch.cat([target_pos, box.data.root_quat_w[env_ids]], dim=1)
    box.write_root_pose_to_sim(target_pose, env_ids=env_ids)

    asset = env.scene[asset_target_cfg.name]
    goal_pos = target_pos.clone()
    goal_pos[:, 2] += z_offset
    dist = torch.norm(asset.data.root_pos_w[env_ids] - goal_pos, dim=1)
    in_target = dist < reach_threshold

    reach_counts[env_ids] = torch.where(
        in_target, reach_counts[env_ids] + 1, torch.zeros_like(reach_counts[env_ids])
    )
    advance = (reach_counts[env_ids] >= min_steps) & (levels[env_ids] < max_level)
    if torch.any(advance):
        levels[env_ids] = torch.where(advance, levels[env_ids] + 1, levels[env_ids])
        reach_counts[env_ids] = torch.where(advance, torch.zeros_like(reach_counts[env_ids]), reach_counts[env_ids])
        updated_height = start_height + levels[env_ids] * step_height
        updated_height = torch.clamp(updated_height, max=max_height)
        target_pos[:, 2] = env_origins[:, 2] + updated_height
        target_pose = torch.cat([target_pos, box.data.root_quat_w[env_ids]], dim=1)
        box.write_root_pose_to_sim(target_pose, env_ids=env_ids)

    return torch.mean(start_height + levels[env_ids] * step_height).item()


def box_top_height_on_reach(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | slice,
    asset_cfg: SceneEntityCfg,
    local_xy: tuple[float, float],
    start_height: float,
    step_height: float,
    max_height: float,
    box_half_height: float,
    reach_threshold: float | None = None,
    reach_threshold_xy: float | None = None,
    reach_threshold_z: float | None = None,
    min_steps: int = 1,
    asset_target_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    z_offset: float = 0.0,
    require_time_out: bool = False,
    forbid_term_names: Sequence[str] | None = None,
) -> float:
    """Increase box top height when the target position is reached.

    If both ``reach_threshold_xy`` and ``reach_threshold_z`` are provided, use split xy/z
    gates. Otherwise fall back to a single 3D distance threshold ``reach_threshold``.
    """
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    if not hasattr(env, "_box_top_height_levels"):
        env._box_top_height_levels = torch.zeros(env.num_envs, device=env.device)
        env._box_top_reach_counts = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    levels = env._box_top_height_levels
    reach_counts = env._box_top_reach_counts

    max_level = int(max(0.0, (max_height - start_height) / step_height))
    levels.clamp_(0, max_level)

    box = env.scene[asset_cfg.name]
    env_origins = env.scene.env_origins[env_ids]
    top_height = start_height + levels[env_ids] * step_height
    top_height = torch.clamp(top_height, max=max_height)
    center_height = top_height - box_half_height
    center_pos = env_origins.clone()
    center_pos[:, 0] += local_xy[0]
    center_pos[:, 1] += local_xy[1]
    center_pos[:, 2] += center_height
    target_pose = torch.cat([center_pos, box.data.root_quat_w[env_ids]], dim=1)
    box.write_root_pose_to_sim(target_pose, env_ids=env_ids)

    asset = env.scene[asset_target_cfg.name]
    goal_pos = env_origins.clone()
    goal_pos[:, 0] += local_xy[0]
    goal_pos[:, 1] += local_xy[1]
    goal_pos[:, 2] += top_height + z_offset
    pos_error = asset.data.root_pos_w[env_ids] - goal_pos
    if reach_threshold_xy is not None and reach_threshold_z is not None:
        dist_xy = torch.norm(pos_error[:, :2], dim=1)
        dist_z = torch.abs(pos_error[:, 2])
        in_target = (dist_xy < reach_threshold_xy) & (dist_z < reach_threshold_z)
    else:
        if reach_threshold is None:
            raise ValueError(
                "box_top_height_on_reach requires `reach_threshold`, or both "
                "`reach_threshold_xy` and `reach_threshold_z`."
            )
        dist = torch.norm(pos_error, dim=1)
        in_target = dist < reach_threshold

    # Filter advancement by termination outcomes, so failed episodes don't push curriculum upward.
    if require_time_out and "time_out" in env.termination_manager.active_terms:
        in_target = in_target & env.termination_manager.get_term("time_out")[env_ids]
    if forbid_term_names is not None:
        for term_name in forbid_term_names:
            if term_name in env.termination_manager.active_terms:
                in_target = in_target & (~env.termination_manager.get_term(term_name)[env_ids])

    reach_counts[env_ids] = torch.where(
        in_target, reach_counts[env_ids] + 1, torch.zeros_like(reach_counts[env_ids])
    )
    advance = (reach_counts[env_ids] >= min_steps) & (levels[env_ids] < max_level)
    if torch.any(advance):
        levels[env_ids] = torch.where(advance, levels[env_ids] + 1, levels[env_ids])
        reach_counts[env_ids] = torch.where(advance, torch.zeros_like(reach_counts[env_ids]), reach_counts[env_ids])
        updated_top = start_height + levels[env_ids] * step_height
        updated_top = torch.clamp(updated_top, max=max_height)
        updated_center = updated_top - box_half_height
        center_pos[:, 2] = env_origins[:, 2] + updated_center
        target_pose = torch.cat([center_pos, box.data.root_quat_w[env_ids]], dim=1)
        box.write_root_pose_to_sim(target_pose, env_ids=env_ids)

    return torch.mean(start_height + levels[env_ids] * step_height).item()


def foot_clearance_metrics(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | slice,
    asset_cfg: SceneEntityCfg,
    min_height: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Compute simple foot clearance metrics for logging."""
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    asset = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[env_ids][:, asset_cfg.body_ids, 2]
    env_origin_z = env.scene.env_origins[env_ids, 2].unsqueeze(1)
    heights = body_pos_w - env_origin_z
    min_height_val = torch.min(heights, dim=1)[0]
    frac_below = torch.mean((min_height_val < min_height).float())
    return {
        "min_height": torch.mean(min_height_val),
        "frac_below": frac_below,
    }

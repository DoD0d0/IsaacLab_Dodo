# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_base_height_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of commanded base height in world frame using exponential kernel."""
    asset = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    target_height = command_term.target_height_w
    height_error = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    return torch.exp(-height_error / std**2)


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def track_box_center_pos_exp(
    env,
    box_cfg: SceneEntityCfg,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    z_offset: float = 0.0,
) -> torch.Tensor:
    """Reward proximity to the box center (with optional z-offset) using exponential kernel."""
    asset = env.scene[asset_cfg.name]
    box = env.scene[box_cfg.name]
    target_pos = box.data.root_pos_w.clone()
    target_pos[:, 2] += z_offset
    pos_error = torch.sum(torch.square(asset.data.root_pos_w - target_pos), dim=1)
    return torch.exp(-pos_error / std**2)


def _get_asset_pos_w(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return root position or mean body position if body_ids are specified."""
    asset = env.scene[asset_cfg.name]
    if isinstance(asset_cfg.body_ids, slice):
        return asset.data.root_pos_w
    body_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    return torch.mean(body_pos, dim=1)


def track_command_pos_exp(
    env,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of commanded position in world frame using exponential kernel."""
    command = env.command_manager.get_command(command_name)
    pos_w = _get_asset_pos_w(env, asset_cfg)
    pos_error = torch.sum(torch.square(pos_w - command), dim=1)
    return torch.exp(-pos_error / std**2)


def track_command_pos_l2(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward negative distance to commanded position in world frame."""
    command = env.command_manager.get_command(command_name)
    pos_w = _get_asset_pos_w(env, asset_cfg)
    dist = torch.norm(pos_w - command, dim=1)
    return -dist


def track_command_pos_xy_l2(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward negative XY distance to commanded position in world frame."""
    command = env.command_manager.get_command(command_name)
    pos_w = _get_asset_pos_w(env, asset_cfg)
    dist = torch.norm(pos_w[:, :2] - command[:, :2], dim=1)
    return -dist


def track_command_height_exp(
    env,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of commanded height using exponential kernel."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    height_error = torch.square(asset.data.root_pos_w[:, 2] - command[:, 2])
    return torch.exp(-height_error / std**2)


def track_feet_below_command_height_penalty(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    min_excess: float = 0.0,
) -> torch.Tensor:
    """Penalty when feet are below commanded height (no penalty for overshoot)."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    target_z = command[:, 2].unsqueeze(1)
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    deficit = target_z + min_excess - feet_z
    return -torch.mean(torch.clamp(deficit, min=0.0), dim=1)


def feet_to_box_top_l2(
    env,
    box_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    box_half_height: float,
    z_offset: float = 0.0,
) -> torch.Tensor:
    """Return mean L2 XY distance of feet to the box top center in world frame."""
    box = env.scene[box_cfg.name]
    asset = env.scene[asset_cfg.name]
    target_pos = box.data.root_pos_w.clone()
    target_pos[:, 2] += box_half_height + z_offset
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    dist = torch.norm(feet_pos[..., :2] - target_pos[:, None, :2], dim=-1)
    return torch.mean(dist, dim=1)


def feet_to_box_top_height_exp(
    env,
    box_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    box_half_height: float,
    margin: float,
    std: float,
) -> torch.Tensor:
    """Reward feet height proximity to box top height using exponential kernel."""
    box = env.scene[box_cfg.name]
    asset = env.scene[asset_cfg.name]
    target_z = box.data.root_pos_w[:, 2] + box_half_height + margin
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    height_error = torch.mean(torch.square(feet_z - target_z.unsqueeze(1)), dim=1)
    return torch.exp(-height_error / std**2)


def feet_to_box_center_xy_exp(
    env,
    box_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Reward feet XY proximity to box center using exponential kernel."""
    box = env.scene[box_cfg.name]
    asset = env.scene[asset_cfg.name]
    target_xy = box.data.root_pos_w[:, :2]
    feet_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    xy_error = torch.mean(torch.square(feet_xy - target_xy.unsqueeze(1)), dim=(1, 2))
    return torch.exp(-xy_error / std**2)


def feet_to_box_top_height_exp_blend(
    env,
    box_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    box_half_height: float,
    margin: float,
    std: float,
    start_step: int,
    end_step: int,
) -> torch.Tensor:
    """Blend min->mean feet height error to box top height over training steps."""
    box = env.scene[box_cfg.name]
    asset = env.scene[asset_cfg.name]
    target_z = box.data.root_pos_w[:, 2] + box_half_height + margin
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    per_foot_error = torch.square(feet_z - target_z.unsqueeze(1))

    if end_step <= start_step:
        blend = 1.0
    else:
        blend = float(env.common_step_counter - start_step) / float(end_step - start_step)
    blend = float(max(0.0, min(1.0, blend)))

    min_error = torch.min(per_foot_error, dim=1)[0]
    mean_error = torch.mean(per_foot_error, dim=1)
    error = (1.0 - blend) * min_error + blend * mean_error
    return torch.exp(-error / std**2)


def feet_to_box_center_xy_exp_blend(
    env,
    box_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    std: float,
    start_step: int,
    end_step: int,
) -> torch.Tensor:
    """Blend min->mean feet XY error to box center over training steps."""
    box = env.scene[box_cfg.name]
    asset = env.scene[asset_cfg.name]
    target_xy = box.data.root_pos_w[:, :2]
    feet_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    per_foot_error = torch.sum(torch.square(feet_xy - target_xy.unsqueeze(1)), dim=-1)

    if end_step <= start_step:
        blend = 1.0
    else:
        blend = float(env.common_step_counter - start_step) / float(end_step - start_step)
    blend = float(max(0.0, min(1.0, blend)))

    min_error = torch.min(per_foot_error, dim=1)[0]
    mean_error = torch.mean(per_foot_error, dim=1)
    error = (1.0 - blend) * min_error + blend * mean_error
    return torch.exp(-error / std**2)


def base_pitch_back_penalty(
    env,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize backward pitch beyond a limit (negative pitch)."""
    asset = env.scene[asset_cfg.name]
    pitch = wrap_to_pi(euler_xyz_from_quat(asset.data.root_quat_w)[1])
    excess = torch.clamp((-pitch - limit_angle), min=0.0)
    return torch.square(excess)


def knees_to_box_center_height_exp(
    env,
    box_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
) -> torch.Tensor:
    """Reward knees being near box center XY at a target height above ground."""
    box = env.scene[box_cfg.name]
    asset = env.scene[asset_cfg.name]
    env_origin_z = env.scene.env_origins[:, 2]
    target_z = env_origin_z + target_height
    target_pos = box.data.root_pos_w.clone()
    target_pos[:, 2] = target_z
    knee_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    pos_error = torch.mean(torch.sum(torch.square(knee_pos - target_pos.unsqueeze(1)), dim=-1), dim=1)
    return torch.exp(-pos_error / std**2)

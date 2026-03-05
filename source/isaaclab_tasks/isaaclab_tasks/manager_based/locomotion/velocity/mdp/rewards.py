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
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

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


def track_command_pos_exp_curriculum_std(
    env,
    command_name: str,
    std_init: float,
    std_mid: float,
    std_final: float,
    height_mid: float,
    height_final: float,
    box_start_height: float,
    box_step_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position tracking reward with curriculum-conditioned std schedule.

    The std transitions in three stages based on the current box top height:
    - top_height < height_mid: std_init
    - height_mid <= top_height < height_final: std_mid
    - top_height >= height_final: std_final
    """
    command = env.command_manager.get_command(command_name)
    pos_w = _get_asset_pos_w(env, asset_cfg)
    pos_error = torch.sum(torch.square(pos_w - command), dim=1)

    if hasattr(env, "_box_top_height_levels"):
        top_height = box_start_height + env._box_top_height_levels * box_step_height
    else:
        top_height = torch.full((env.num_envs,), box_start_height, device=env.device)

    std = torch.full_like(top_height, std_final)
    std = torch.where(top_height < height_final, torch.full_like(std, std_mid), std)
    std = torch.where(top_height < height_mid, torch.full_like(std, std_init), std)

    return torch.exp(-pos_error / torch.square(std))


def track_command_pos_progress(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    clip: float = 0.25,
) -> torch.Tensor:
    """Reward one-step progress toward commanded position.

    Positive when the robot gets closer to the target position than in the previous step.
    """
    command = env.command_manager.get_command(command_name)
    pos_w = _get_asset_pos_w(env, asset_cfg)
    curr_dist = torch.norm(pos_w - command, dim=1)

    if not hasattr(env, "_prev_target_pos_dist"):
        env._prev_target_pos_dist = curr_dist.clone()

    progress = env._prev_target_pos_dist - curr_dist
    if hasattr(env, "episode_length_buf"):
        first_step = env.episode_length_buf <= 1
        progress = torch.where(first_step, torch.zeros_like(progress), progress)

    env._prev_target_pos_dist = curr_dist.clone()
    return torch.clamp(progress, min=-clip, max=clip)


def track_command_z_exp_curriculum(
    env,
    command_name: str,
    std: float,
    height_start: float,
    height_full: float,
    box_start_height: float,
    box_step_height: float,
    ramp_power: float = 2.0,
    max_scale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward commanded z-tracking with a curriculum-height ramp.

    The reward scale ramps from 0 to max_scale as box top height increases
    from height_start to height_full.
    """
    command = env.command_manager.get_command(command_name)
    pos_w = _get_asset_pos_w(env, asset_cfg)
    z_error_sq = torch.square(pos_w[:, 2] - command[:, 2])
    z_reward = torch.exp(-z_error_sq / std**2)

    if hasattr(env, "_box_top_height_levels"):
        top_height = box_start_height + env._box_top_height_levels * box_step_height
    else:
        top_height = torch.full((env.num_envs,), box_start_height, device=env.device)

    denom = max(height_full - height_start, 1.0e-6)
    ramp = torch.clamp((top_height - height_start) / denom, min=0.0, max=1.0)
    ramp = torch.pow(ramp, ramp_power) * max_scale
    return z_reward * ramp


def track_feet_to_box_top_exp_curriculum(
    env,
    asset_cfg: SceneEntityCfg,
    box_cfg: SceneEntityCfg,
    box_half_size_x: float,
    box_half_size_y: float,
    box_half_height: float,
    foot_z_offset: float,
    std: float,
    min_top_height: float,
    box_start_height: float,
    box_step_height: float,
) -> torch.Tensor:
    """Reward feet positions approaching the box top surface after curriculum activation."""
    robot = env.scene[asset_cfg.name]
    box = env.scene[box_cfg.name]
    foot_pos = robot.data.body_pos_w[:, asset_cfg.body_ids, :]

    box_center = box.data.root_pos_w
    box_top_z = box_center[:, 2] + box_half_height + foot_z_offset

    min_x = box_center[:, 0] - box_half_size_x
    max_x = box_center[:, 0] + box_half_size_x
    min_y = box_center[:, 1] - box_half_size_y
    max_y = box_center[:, 1] + box_half_size_y

    target_x = torch.clamp(foot_pos[:, :, 0], min=min_x.unsqueeze(1), max=max_x.unsqueeze(1))
    target_y = torch.clamp(foot_pos[:, :, 1], min=min_y.unsqueeze(1), max=max_y.unsqueeze(1))
    target_z = box_top_z.unsqueeze(1).expand_as(target_x)

    dx = foot_pos[:, :, 0] - target_x
    dy = foot_pos[:, :, 1] - target_y
    dz = foot_pos[:, :, 2] - target_z
    dist_sq = dx * dx + dy * dy + dz * dz
    foot_reward = torch.exp(-dist_sq / std**2)
    reward = torch.mean(foot_reward, dim=1)

    if hasattr(env, "_box_top_height_levels"):
        top_height = box_start_height + env._box_top_height_levels * box_step_height
    else:
        top_height = torch.full((env.num_envs,), box_start_height, device=env.device)
    active = top_height >= min_top_height

    return reward * active.float()


def nose_dive_penalty_curriculum(
    env,
    min_top_height: float,
    box_start_height: float,
    box_step_height: float,
    forward_z_tol: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize nose-down posture after curriculum passes a minimum box-top height.

    The penalty is active only when current box top height >= min_top_height.
    It uses the robot forward-axis z component in world frame:
    - forward_z >= -forward_z_tol: no penalty
    - forward_z < -forward_z_tol: quadratic penalty
    """
    if hasattr(env, "_box_top_height_levels"):
        top_height = box_start_height + env._box_top_height_levels * box_step_height
    else:
        top_height = torch.full((env.num_envs,), box_start_height, device=env.device)
    active = top_height >= min_top_height

    asset = env.scene[asset_cfg.name]
    forward_axis_b = torch.zeros((env.num_envs, 3), device=env.device)
    forward_axis_b[:, 0] = 1.0
    forward_axis_w = quat_apply(asset.data.root_quat_w, forward_axis_b)
    forward_z = forward_axis_w[:, 2]

    # Penalize only meaningful nose-down tilt, while allowing small pitch fluctuations.
    nose_down = torch.clamp(-(forward_z + forward_z_tol), min=0.0)
    penalty = torch.square(nose_down)
    return penalty * active.float()

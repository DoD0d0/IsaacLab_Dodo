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
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

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


def track_command_pos_task_window_curriculum(
    env,
    command_name: str,
    reward_window_s_init: float,
    reward_window_s_final: float,
    height_start: float,
    height_full: float,
    box_start_height: float,
    box_step_height: float,
    ramp_power: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """3D task reward with a curriculum-scheduled reward window.

    r_task = (1 / T_r) * 1 / (1 + ||x_b - x_b*||^2), active only for t > T - T_r.
    """
    command = env.command_manager.get_command(command_name)
    pos_w = _get_asset_pos_w(env, asset_cfg)
    pos_error_sq = torch.sum(torch.square(pos_w - command), dim=1)

    if hasattr(env, "_box_top_height_levels"):
        top_height = box_start_height + env._box_top_height_levels * box_step_height
    else:
        top_height = torch.full((env.num_envs,), box_start_height, device=env.device)

    denom = max(float(height_full - height_start), 1.0e-6)
    ramp = torch.clamp((top_height - height_start) / denom, min=0.0, max=1.0)
    ramp = torch.pow(ramp, ramp_power)
    T_r = reward_window_s_init + (reward_window_s_final - reward_window_s_init) * ramp
    T_r = torch.clamp(T_r, min=1.0e-6)

    t = env.episode_length_buf.float() * env.step_dt
    T = env.max_episode_length_s
    in_task_window = t > (T - T_r)

    r_task = (1.0 / T_r) * (1.0 / (1.0 + pos_error_sq))
    return torch.where(in_task_window, r_task, torch.zeros_like(r_task))


def feet_body_lin_acc_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize squared linear acceleration of selected robot bodies (e.g., feet).

    This matches the foot-acceleration penalty: sum_feet ||xddot_f||^2.
    """
    asset = env.scene[asset_cfg.name]
    lin_acc = asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :]
    return torch.sum(torch.sum(torch.square(lin_acc), dim=-1), dim=1)


def stall_penalty_far_from_target(
    env,
    command_name: str,
    speed_threshold: float = 0.1,
    distance_threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Stalling penalty: -1 when nearly stopped while far from target.

    r_stall = -1, if ||xdot_b|| < speed_threshold and ||x_b - x_b*|| > distance_threshold, else 0.
    """
    asset = env.scene[asset_cfg.name]
    speed_xy = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=1)

    command_xy = env.command_manager.get_command(command_name)[:, :2]
    pos_xy = _get_asset_pos_w(env, asset_cfg)[:, :2]
    dist_xy = torch.norm(pos_xy - command_xy, dim=1)

    stalled = speed_xy < speed_threshold
    far_from_target = dist_xy > distance_threshold
    should_penalize = stalled & far_from_target

    return torch.where(should_penalize, -torch.ones_like(speed_xy), torch.zeros_like(speed_xy))


def exploration_velocity_bias_xy(
    env,
    command_name: str,
    min_distance: float = 0.2,
    min_speed: float = 0.05,
    deactivate_top_height: float = 0.15,
    box_start_height: float = 0.0,
    box_step_height: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exploration reward using velocity alignment with target direction.

    r_bias = xdot_xy · (x*_xy - x_xy) / (||xdot_xy|| * ||x*_xy - x_xy||)

    The reward is active only when:
    - the robot is sufficiently far from target,
    - the robot is actually moving,
    - curriculum box top height is below `deactivate_top_height`.
    """
    asset = env.scene[asset_cfg.name]
    vel_xy = asset.data.root_lin_vel_w[:, :2]

    command_xy = env.command_manager.get_command(command_name)[:, :2]
    pos_xy = _get_asset_pos_w(env, asset_cfg)[:, :2]
    to_target_xy = command_xy - pos_xy

    speed = torch.norm(vel_xy, dim=1)
    distance = torch.norm(to_target_xy, dim=1)
    denom = speed * distance

    alignment = torch.where(
        denom > 1.0e-6,
        torch.sum(vel_xy * to_target_xy, dim=1) / denom,
        torch.zeros_like(speed),
    )
    alignment = torch.clamp(alignment, min=-1.0, max=1.0)

    active = (distance > min_distance) & (speed > min_speed)

    if hasattr(env, "_box_top_height_levels"):
        top_height = box_start_height + env._box_top_height_levels * box_step_height
    else:
        top_height = torch.full((env.num_envs,), box_start_height, device=env.device)
    active &= top_height < deactivate_top_height

    return torch.where(active, alignment, torch.zeros_like(alignment))

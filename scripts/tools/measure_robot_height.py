#!/usr/bin/env python3
"""Measure robot height from body link positions in a single env."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure robot height for a task.")
    parser.add_argument("--task", type=str, default="Isaac-Jump-Dodo-Play-v0", help="Task name.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    disable_fabric = getattr(args, "disable_fabric", False)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=not disable_fabric)
    env = gym.make(args.task, cfg=env_cfg)
    env.reset()

    # Step once to ensure data buffers are updated
    action_dim = env.unwrapped.action_manager.total_action_dim
    action = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)
    env.step(action)

    robot = env.unwrapped.scene["robot"]
    body_pos = robot.data.body_pos_w[0]
    min_z = body_pos[:, 2].min().item()
    max_z = body_pos[:, 2].max().item()
    root_z = robot.data.root_pos_w[0, 2].item()
    height = max_z - min_z

    print(f"root_z: {root_z:.4f} m")
    print(f"min_body_z: {min_z:.4f} m")
    print(f"max_body_z: {max_z:.4f} m")
    print(f"approx_height: {height:.4f} m")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

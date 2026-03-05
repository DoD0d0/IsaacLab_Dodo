# Jump Task State

This document summarizes the current implementation of the Dodo jump task in this repository.
It includes only behaviors and parameters that are present in code right now.

## Task Registration

- Task IDs:
  - `Isaac-Jump-Dodo-v0`
  - `Isaac-Jump-Dodo-Play-v0`
- Registry file:
  - `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/dodo/__init__.py`

## Files Used by the Jump Task

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/dodo/jump_env_cfg.py`
  - Backward-compatible export layer.
  - Re-exports the actual configs from `jump_task.py` and `jump_terms.py`.

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/dodo/jump_task.py`
  - Assembles the jump environment from scene/command/obs/reward/curriculum configs.
  - Applies runtime environment settings (simulation, reset pose, action scale, terminations, play env count).

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/dodo/jump_terms.py`
  - Defines jump scene objects, command term, observation terms, reward terms, curriculum term.
  - Contains most task-level constants (box geometry, target margins, curriculum thresholds).

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/commands.py`
  - Implements `BoxCenterCommand` used by the jump task.
  - Publishes `Metrics/target_pos/pos_error`.

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/curriculums.py`
  - Implements `box_top_height_on_reach` used for jump box-height curriculum progression.

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py`
  - Implements custom jump reward functions currently referenced by `jump_terms.py`.

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/dodo/agents/rsl_rl_ppo_cfg.py`
  - Defines `DodoJumpPPORunnerCfg` (PPO hyperparameters used for jump training/play checkpoints).

- `source/isaaclab_assets/isaaclab_assets/robots/dodo.py`
  - Robot articulation/actuator configuration used when the jump env resolves Dodo robot cfg.

## Current Environment Assembly (`jump_task.py`)

- Base class: `LocomotionVelocityRoughEnvCfg`
- Scene:
  - `num_envs=4096` (train), `num_envs=1` (play)
  - Terrain forced to plane (`terrain_type="plane"`, generator disabled)
  - Height scanner attached to `Robot/body_link`, meshes: ground + box
- Episode/action/reset:
  - `episode_length_s=8.0`
  - `actions.joint_pos.scale=0.45`
  - Base reset pose fixed at `x=-1.0, y=0.0, z=0.16, yaw=0.0`
  - Base reset velocities all zeros
- Randomization:
  - Push event disabled
  - Extra base mass randomization disabled
  - Joint reset range fixed to `(1.0, 1.0)`
- Terminations:
  - `base_contact` checks `body_link`
  - `root_height_below_minimum.minimum_height=0.12`
  - `bad_pitch` enabled with pitch limit `120 deg`

## Scene and Command (`jump_terms.py`, `commands.py`)

- Box object:
  - Size: `1.5 x 1.5 x 1.0` m
  - Kinematic rigid body, gravity disabled
  - Initial center at `(0,0,0)`

- Command:
  - Type: `BoxCenterCommand`
  - `target_pos = box.root_pos_w + [0, 0, z_offset]`
  - Jump z-offset:
    - `BOX_HALF_HEIGHT=0.5`
    - `COMMAND_TARGET_TOP_MARGIN=0.6`
    - `TARGET_POS_Z_OFFSET=1.1`

- Command metric:
  - `Metrics/target_pos/pos_error = ||robot.root_pos_w - target_pos||_2` (3D distance)

## Observations (`jump_terms.py`)

Policy observation terms:

- `projected_gravity`
- `target_pos` (generated command)
- `joint_pos` (relative)
- `joint_vel` (relative)
- `actions` (last action)
- `height_scan`
  - Sensor: `height_scanner`
  - Offset: `19.5`
  - Clip: `(-8.0, 8.0)`

Observation corruption:

- Noise is configured on individual terms.
- Group-level corruption is disabled in `jump_task.py` (`enable_corruption=False` override).

## Curriculum (`jump_terms.py`, `curriculums.py`)

Active curriculum term: `box_height` via `box_top_height_on_reach`.

Configured parameters:

- `start_height=0.0`
- `step_height=0.1`
- `max_height=1.0`
- `box_half_height=0.5`
- `reach_threshold=0.38`
- `min_steps=2`
- `z_offset=0.2` (`CURRICULUM_REACH_TOP_MARGIN`)

Implementation behavior:

- Keeps per-env level counters (`_box_top_height_levels`, `_box_top_reach_counts`).
- Places box center from current top height each step.
- Computes goal as box top + `z_offset`.
- Increments reach count while distance-to-goal `< reach_threshold`.
- Advances level when count `>= min_steps`.

## Reward Terms (Current Active Set)

All are defined in `jump_terms.py` and implemented in `mdp/rewards.py` unless noted.

1. `position_tracking` (weight `4.0`)
- Function: `track_command_pos_exp_curriculum_std`
- Meaning: 3D command tracking reward with curriculum-dependent std.
- Formula:
  - `r = exp(-||p - c||^2 / std(h)^2)`
  - `std(h)` schedule:
    - `1.0` if top height `< 0.1`
    - `0.9` if `0.1 <= h < 0.2`
    - `0.8` if `h >= 0.2`

2. `position_progress` (weight `3.0`)
- Function: `track_command_pos_progress`
- Meaning: one-step progress toward 3D target.
- Formula:
  - `r = clamp(d_prev - d_now, -0.25, 0.25)`
  - First step of each episode is forced to 0.

3. `z_tracking_curriculum` (weight `2.0`)
- Function: `track_command_z_exp_curriculum`
- Meaning: z-only command tracking, ramped by curriculum height.
- Formula:
  - `z_r = exp(-(z - z_cmd)^2 / 0.45^2)`
  - `ramp = clamp((h - 0.03)/(0.4 - 0.03), 0, 1)^2`
  - `r = z_r * ramp`

4. `feet_to_box_top` (weight `1.5`)
- Function: `track_feet_to_box_top_exp_curriculum`
- Feet used: `left_link_4`, `right_link_4`
- Meaning: rewards feet being near the box top surface region (XY clamped to box top bounds, target Z at top surface + offset).
- Parameters:
  - Box half sizes: `0.75, 0.75`
  - Box half height: `0.5`
  - Foot z offset above top: `0.02`
  - Std: `0.35`
  - Activation: only when top height `>= 0.03`

5. `is_termination` (weight `-100.0`)
- Function: `mdp.is_terminated_term` (from base env mdp)
- Penalized term keys:
  - `base_contact`
  - `root_height_below_minimum`

6. `dive_penalty` (weight `-0.05`)
- Function: `nose_dive_penalty_curriculum`
- Meaning: penalizes nose-down orientation after curriculum activation.
- Activation: top height `>= 0.05`
- Uses forward-axis world `z` component with tolerance `0.05`.

## PPO Config Used by Jump (`DodoJumpPPORunnerCfg`)

- `num_steps_per_env=96`
- `max_iterations=1500`
- `experiment_name="dodo_jump"`
- `empirical_normalization=True`
- Policy:
  - `init_noise_std=0.3`
  - hidden dims `[512, 256, 128]` (actor/critic)
- Algorithm:
  - `entropy_coef=0.001`
  - `learning_rate=3e-4`
  - `clip_param=0.2`
  - `gamma=0.99`, `lam=0.95`
  - `desired_kl=0.01`

## Train vs Play Difference

- Same jump MDP structure is used for both.
- Play config only reduces environment count to `1` (in `DodoJumpEnvCfg_PLAY`).

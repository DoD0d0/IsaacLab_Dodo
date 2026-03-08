# IsaacLab_Dodo

Isaac Lab workspace for training the Dodo bipedal robot. Contains velocity tracking and jump tasks using RSL-RL (PPO).

---

## Prerequisites

- Ubuntu 22.04 or Windows 11 (x64)
- 32GB RAM, 16GB VRAM
- Python 3.10 (Isaac Sim 4.X) or 3.11 (Isaac Sim 5.X)
- NVIDIA GPU with driver
- Conda recommended
- **Isaac Sim must be installed**

[Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

---

## Setup

```bash
git clone https://github.com/DoD0d0/IsaacLab_Dodo.git
cd IsaacLab_Dodo
conda create -n isaaclab python=3.10
conda activate isaaclab
pip install --upgrade pip
pip install "isaacsim[all,extscache]==4.5.0.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
sudo apt install cmake build-essential
./isaaclab.sh --install rsl_rl
pip install wandb
```

---

## Repository Structure

```
IsaacLab_Dodo/
├── assets/robots/dodo/
│   ├── dodobot_v3/urdf/      # URDF sources
│   └── dodo.usd              # Converted USD
├── scripts/
│   ├── reinforcement_learning/rsl_rl/
│   │   ├── train.py          # Training entry point
│   │   └── play.py           # Evaluation/visualization
│   └── tools/convert_urdf.py # URDF→USD converter
├── source/
│   ├── isaaclab_assets/robots/dodo.py    # Robot asset definition
│   └── isaaclab_tasks/.../velocity/
│       ├── config/dodo/
│       │   ├── jump_task.py              # Jump environment assembly
│       │   ├── jump_terms.py             # MDP terms (obs/rewards/curriculum)
│       │   ├── flat_env_cfg.py           # Flat terrain config
│       │   ├── rough_env_cfg.py          # Rough terrain config
│       │   └── agents/rsl_rl_ppo_cfg.py  # PPO hyperparameters
│       └── mdp/
│           ├── commands.py               # Command generators
│           ├── rewards.py                # Custom reward functions
│           └── curriculums.py            # Curriculum logic
└── README.md
```

### How to Customize Tasks

**Add new reward term:**
1. Implement function in `mdp/rewards.py`:
   ```python
   def my_reward(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
       # Return per-env reward tensor
       return torch.zeros(env.num_envs, device=env.device)
   ```
2. Add to config in `*_env_cfg.py` or `*_terms.py`:
   ```python
   my_reward = RewTerm(func=mdp.my_reward, weight=1.0, params={...})
   ```

**Add new observation:**
1. Implement in `mdp/observations.py` (or use existing functions)
2. Add to observation config:
   ```python
   my_obs = ObsTerm(func=mdp.my_observation_func)
   ```

**Add new command type:**
1. Implement command class in `mdp/commands.py`
2. Add to commands config:
   ```python
   my_command = mdp.MyCommandCfg(class_type=mdp.MyCommand, ...)
   ```

**Modify robot:**
1. Edit URDF in `assets/robots/dodo/dodobot_v3/urdf/`
2. Reconvert to USD: `./isaaclab.sh -p scripts/tools/convert_urdf.py ...`
3. Update robot config in `source/isaaclab_assets/robots/dodo.py` if needed

**Create new task:**
1. Copy existing task config (e.g., `jump_task.py`)
2. Modify scene, observations, rewards, commands
3. Register task in `__init__.py`
4. Train: `--task=Isaac-MyTask-Dodo-v0`

---

## Robot Specification

- **Type:** Bipedal
- **Control:** Joint position targets (implicit PD: stiffness=40, damping=2)
- **Initial state:** Base at (0, 0, 0.45), joints at zero
- **Asset format:** USD converted from URDF

**Convert URDF to USD:**
```bash
./isaaclab.sh -p scripts/tools/convert_urdf.py \
  assets/robots/dodo/dodobot_v3/urdf/dodo.urdf \
  assets/robots/dodo/dodo.usd
```

### Control Abstraction

**Why position control instead of torque?**

Position control provides a higher-level abstraction that simplifies learning:
- **Stability:** Implicit PD controller handles low-level stabilization. Policy doesn't need to learn feedback control from scratch.
- **Smoother learning:** Position targets are easier to optimize than raw torques. Less prone to instability during exploration.
- **Sim-to-real transfer:** Position control abstracts away motor dynamics. Real robot controllers often use position interfaces, reducing sim-to-real gap.
- **Computational efficiency:** Fewer dimensions to explore in action space compared to full torque control.

**PD parameters (stiffness=40, damping=2):**
- Tuned for responsive yet stable tracking
- Higher stiffness → faster response but more oscillation
- Higher damping → smoother but slower
- Current values provide good balance for dynamic locomotion

**Action scaling (0.45 for jump task):**
- Raw network output ∈ [-1, 1]
- Scaled to joint position offsets from default pose
- Scale factor controls movement range:
  - Too high → unstable, aggressive motions
  - Too low → limited expressiveness
  - 0.45 allows large motions while maintaining stability

---

# Velocity Locomotion Task

## Overview

Trains Dodo to track commanded base velocities (forward/lateral velocity + yaw rate).

**Environments:**
- **Flat terrain:** Infinite plane, no obstacles, velocity curriculum
- **Rough terrain:** Procedural terrain, height scanning, terrain curriculum

**RL:** RSL-RL (PPO)

### Task Formulation

**MDP components:**
- State: Robot observations (velocities, joint states, terrain, commands)
- Action: Joint position targets $\in [-1, 1]^N$ (scaled by 0.5)
- Reward: Weighted sum of tracking rewards and penalties
- Episode: 20s max, early termination on falls

**Manager-based architecture:**
- ObservationManager, RewardManager, CommandManager, CurriculumManager, TerminationManager
- Modular design allows config-based experimentation

## Observations

Concatenated vector, ~200-300 dims (depends on terrain).

| Term | Dim | Description |
|------|-----|-------------|
| `base_lin_vel` | 3 | Base linear velocity (noisy) |
| `base_ang_vel` | 3 | Base angular velocity (noisy) |
| `projected_gravity` | 3 | Gravity in robot frame |
| `velocity_commands` | 3 | Target velocities (vx, vy, ωz) |
| `joint_pos` | N | Joint positions (noisy) |
| `joint_vel` | N | Joint velocities (noisy) |
| `actions` | N | Previous action |
| `height_scan` | ~160 | RayCaster output (rough terrain only) |

Noise applied during training, disabled during eval.

### Observation Design Rationale

**Proprioceptive observations:**
- **Base velocities:** Direct feedback for velocity tracking. Noise simulates IMU characteristics.
- **Projected gravity:** Rotation-invariant orientation. Avoids gimbal lock issues of Euler angles.
- **Joint positions/velocities:** Essential for state understanding. Velocities enable momentum estimation.
- **Previous actions:** Temporal context for action smoothness.

**Exteroceptive observations:**
- **Height scan:** Rough terrain only. Local terrain geometry for foot placement.

**Command observations:**
- **Velocity commands / Target position:** Goal specification in robot frame.
- **Time to go (jump):** Enables time-aware planning for task window reward.

## Rewards (Flat Terrain)

### Reward Design Philosophy

**Balance:**
- Task rewards (tracking, reaching) dominate signal
- Regularization (smoothness, energy) prevents degenerate solutions

**Sim-to-real:**
- Penalize accelerations/torques → smooth, efficient motions
- Penalize sliding → predictable contact
- Encourage natural behaviors within physical limits

### Reward Terms

Based on `flat_env_cfg.py`:

| Term | Weight | Formula | Purpose |
|------|--------|---------|---------|
| `track_lin_vel_xy_exp` | 4.5 | $\exp(-\frac{\|v_{xy} - v_{xy}^{\text{cmd}}\|^2}{0.4^2})$ | Track XY velocity |
| `track_ang_vel_z_exp` | 1.6 | $\exp(-\frac{(\omega_z - \omega_z^{\text{cmd}})^2}{0.5^2})$ | Track yaw rate |
| `termination_penalty` | -100.0 | -100 if terminated | Penalize falls |
| `flat_orientation_l2` | -0.8 | $-\|$pitch, roll$\|^2$ | Keep upright |
| `lin_vel_z_l2` | -0.5 | $-v_z^2$ | Minimize vertical motion |
| `ang_vel_xy_l2` | -0.05 | $-\|\omega_{xy}\|^2$ | Minimize roll/pitch rate |
| `dof_acc_l2` | $-5 \times 10^{-8}$ | $-\sum_j \ddot{q}_j^2$ | Smooth joint motion |
| `dof_torques_l2` | $-1 \times 10^{-6}$ | $-\sum_j \tau_j^2$ | Minimize torque |
| `action_rate_l2` | -0.002 | $-\sum_i (a_t - a_{t-1})^2$ | Smooth actions |
| `feet_air_time` | 0.75 | $\sum t_{\text{air}}$ (if $v_x > 0.1$) | Encourage stepping |
| `feet_slide` | -0.25 | $-\sum \|v_{\text{foot}}\|$ (if contact) | Penalize sliding |
| `joint_deviation_hip` | -0.02 | $-\|q_{\text{hip}} - q_{\text{default}}\|$ | Hip near default |
| `joint_deviation_l1` | -0.05 | $-\sum_j \|q_j - q_{\text{default}}\|$ | Joints near default |

## Commands (Flat Terrain)

From `flat_env_cfg.py:82-89`:

```python
lin_vel_x: (0.2, 0.8)      # Forward speed range (m/s)
lin_vel_y: (0.0, 0.0)      # Lateral speed (disabled)
ang_vel_z: (-0.5, 0.5)     # Yaw rate range (rad/s)
heading: (0.0, 0.0)        # Heading (disabled)
resampling_time_range: (2.0, 4.0)  # Command duration
```

Commands resampled every 2-4 seconds.

## Curriculum (Flat)

Linear velocity X range ramps from (0.2, 0.8) → (0.4, 1.2) over 1.5M steps.

## Rough Terrain Configuration

Rough terrain variant uses procedural terrain generation and height scanning for robust locomotion.

**Key differences from flat:**
- Terrain generator creates varied surfaces (inherited from parent config)
- Height scanner provides ~160-dim local terrain map
- Terrain curriculum progressively increases difficulty
- Adjusted reward weights for terrain contact patterns

**From `rough_env_cfg.py`:**
- Command ranges: `lin_vel_x=(0.0, 1.0)`, `ang_vel_z=(-1.0, 1.0)`
- Terrain curriculum automatically adjusts difficulty based on performance
- Height scanner attached to body_link, 1.6m × 1.0m grid at 0.1m resolution

## PPO Config (Flat)

| Parameter | Value |
|-----------|-------|
| `actor_hidden_dims` | [256, 128, 128] |
| `critic_hidden_dims` | [256, 128, 128] |
| `learning_rate` | 1e-3 |
| `entropy_coef` | 0.008 |
| `max_iterations` | 1500 |

## Training

**Flat:**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Velocity-Flat-Dodo-v0 \
  --num_envs=8192 \
  --headless
```

**Rough:**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Velocity-Rough-Dodo-v0 \
  --num_envs=8192 \
  --headless
```

**Play:**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Isaac-Velocity-Flat-Dodo-Play-v0 \
  --checkpoint=logs/rsl_rl/dodo_flat/<run>/model_<iter>.pt \
  --num_envs=50
```

### Training Tips

**Monitoring:**
- Episode reward should trend upward
- High base_contact rate indicates falling issues
- Check velocity/position errors decreasing

**Common issues:**
- Divergence → lower LR, reduce entropy
- Too conservative → increase exploration noise
- Falling → increase termination penalty
- Slow → verify task reward weights dominate

---

# Jump Task

## Overview

Trains Dodo to jump onto a box and reach a 3D target above it. Box height increases progressively through curriculum.

**Status:**
- ✅ Successfully tested at **0.49m** box height
- Checkpoint: `logs/rsl_rl/dodo_jump/2026-03-08_12-26-51/model_3798.pt`
- Curriculum step: 3cm increments
- Episode length: 8 seconds

**Task setup:**
- Flat plane with kinematic box (1.5m × 1.5m × 1.0m)
- Target: box top + 0.6m z-offset
- Action: Joint position targets (scale=0.45)
- Observation: ~330-dim vector

### Task Formulation

**MDP components:**
- State: Proprioception + target position (base frame) + time remaining + height scan
- Action: Joint position targets $\in [-1, 1]^N$ (scaled by 0.45)
- Reward: Task window (w=200) + exploration bias (w=1) + stall penalty (w=3) + smoothness penalties
- Episode: 8s max, early termination on falls

**Curriculum:**
- Box height: $h_{\text{start}} + \text{level} \times 0.03$ m
- Target: box top + 0.6m margin
- Advances after 30 consecutive clean successes

---

## Scene

### Box

Static kinematic object (`jump_terms.py`):

```python
box = RigidObjectCfg(
    spawn=sim_utils.MeshCuboidCfg(
        size=(1.5, 1.5, 1.0),           # Width, depth, full height
        kinematic_enabled=True,          # Static
        disable_gravity=True,
    ),
)
```

- Full height: 1.0m, half-height: 0.5m
- Position: dynamically adjusted by curriculum
  - Box center Z = `(box_top_height - 0.5m)`
  - Top height ranges 0.0m → 1.0m

### Robot Start

1m behind box (`jump_task.py`):
```python
pose_range: {"x": (-1.0, -1.0), "y": (0.0, 0.0), "z": (0.16, 0.16), "yaw": (0.0, 0.0)}
```

Provides approach space before jumping.

### Height Scanner (RayCaster)

Measures terrain/obstacle heights using downward-pointing ray grid.

**Configuration:**
```python
height_scanner = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/body_link",           # Attached to body
    offset=RayCasterCfg.OffsetCfg(pos=(0.25, 0.0, 20.0)), # Forward-shifted, 20m up
    attach_yaw_only=True,                                  # Rotates with yaw only
    pattern_cfg=patterns.GridPatternCfg(
        resolution=0.1,      # 10cm spacing
        size=[1.6, 1.0],     # 1.6m forward × 1.0m lateral
    ),
    mesh_prim_paths=["/World/ground", "/World/envs/env_.*/Box"],
)
```

**How it works:**

```
Side view:

    Robot body
        │
        │ ← Rays from 20m above
        ▼
 ┌──────────────┐
 │  Ray grid    │ 1.6m × 1.0m @ 0.1m res
 │ (+0.25m fwd) │ ~160 rays
 └──────────────┘
        │
────────┼──────── Ground
        │
   ┌────┴────┐
   │         │
   │   Box   │ 0.49m top (current)
   │         │
   └─────────┘
```

**Processing:**
1. Rays hit ground/box, measure height differences
2. Offset by -19.5m (rays start at +20m)
3. Clip to [-1, 1], scale by 2 → range [-2, 2]m
4. Add noise: Uniform[-0.1, 0.1]
5. Output: ~160 values

**Why +0.25m forward bias?**
Box edge appears earlier in scan, giving more prep time.

---

## Observations

~330-dim concatenated vector.

| Term | Dim | Function | Description |
|------|-----|----------|-------------|
| `projected_gravity` | 3 | `mdp.projected_gravity` | Gravity in robot frame |
| `target_pos_b` | 3 | `mdp.command_target_pos_b` | Target in robot base frame |
| `time_to_go` | 1 | `mdp.command_time_to_go` | Remaining time (seconds) |
| `joint_pos` | N | `mdp.joint_pos_rel` | Joint positions (noisy) |
| `joint_vel` | N | `mdp.joint_vel_rel` | Joint velocities (noisy) |
| `actions` | N | `mdp.last_action` | Previous action |
| `height_scan` | ~160 | `mdp.height_scan` | RayCaster output |

Noise applied during training for robustness, disabled during eval.

### Key Implementations

**Target in base frame** (`commands.py:80-89`):
```python
def command_target_pos_b(env, command_name, asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    command_w = env.command_manager.get_command(command_name)
    rel_target_w = command_w - asset.data.root_pos_w
    return quat_apply_inverse(asset.data.root_quat_w, rel_target_w)
```

Makes target orientation-independent.

**Time to go** (`commands.py:92-95`):
```python
def command_time_to_go(env):
    remaining_s = (env.max_episode_length - env.episode_length_buf).float() * env.step_dt
    return remaining_s.unsqueeze(1)
```

Enables time-aware behavior.

### Why Joint Velocity Matters

For dynamic tasks like jumping:
- **With velocity:** Policy knows momentum, can time explosive extensions
- **Without velocity:** Must estimate from position history (POMDP), slower learning, worse timing

For quasi-static tasks, velocity can be omitted. For jumping, it's critical.

---

## Commands

Target position computed by `BoxCenterCommand` (`commands.py:37-76`):

```python
self.target_pos[:, :2] = box.data.root_pos_w[:, :2]          # XY: box center
self.target_pos[:, 2] = box.data.root_pos_w[:, 2] + z_offset  # Z: top + offset
```

**Z-offset:**
```python
TARGET_POS_Z_OFFSET = BOX_HALF_HEIGHT + COMMAND_TARGET_TOP_MARGIN
                    = 0.5m            + 0.6m
                    = 1.1m
```

**Why 0.6m margin?**

Dodo's standing height is ~0.6m (base to head). Setting target 0.6m above box top ensures:
1. Robot lands feet-first on box, not head-first
2. Avoids "head dive" local optimum
3. Provides clearance for whole-body landing

**Metrics tracked:**
- `pos_error`: 3D Euclidean distance
- `pos_error_xy`: Horizontal distance
- `pos_error_z`: Vertical distance

---

## Rewards

Total: $r_t = \sum_i w_i \cdot r_i(t)$

**Constants:**
```python
BOX_HALF_HEIGHT = 0.5
BOX_START_TOP_HEIGHT = 0.0
BOX_STEP_HEIGHT = 0.03
COMMAND_TARGET_TOP_MARGIN = 0.6
CURRICULUM_REACH_THRESHOLD_XY = 0.30
CURRICULUM_REACH_THRESHOLD_Z = 0.15
```

### Task Rewards

#### 1. Task Window (w=200.0)

`track_command_pos_task_window_curriculum` (`rewards.py:118-154`)

$$
r_{\text{task}} = \begin{cases}
\frac{1}{T_r} \cdot \frac{1}{1 + \|x_b - x_b^*\|^2}, & t > T - T_r \\
0, & \text{otherwise}
\end{cases}
$$

- $x_b$: robot base position
- $x_b^*$: target position
- $T$: 8s episode length
- $T_r$: reward window (curriculum-scheduled, currently 40s = always active)
- $t$: current time

Activates only near episode end, forcing policy to reach target before timeout. Inverse squared error: smooth gradients near target, heavy penalty far away.

#### 2. Exploration Bias (w=1.0)

`exploration_velocity_bias_xy` (`rewards.py:195-240`)

$$
r_{\text{bias}} = \frac{v_{xy} \cdot (x^*_{xy} - x_{xy})}{\|v_{xy}\| \cdot \|x^*_{xy} - x_{xy}\|}
$$

Cosine similarity between velocity and direction-to-target.

**Active when:**
- Distance > 0.2m
- Speed > 0.05 m/s
- Box height < 0.4m

Encourages directional movement early in training, deactivates at higher heights.

#### 3. Stall Penalty (w=3)

`stall_penalty_far_from_target` (`rewards.py:170-192`)

$$
r_{\text{stall}} = \begin{cases}
-1, & \|v_{xy}\| < 0.1 \text{ and } \|x_{xy} - x^*_{xy}\| > 0.4 \\
0, & \text{otherwise}
\end{cases}
$$

Prevents crouching near target without jumping (observed local optimum).

### Smoothness Penalties

L2 squared kernel for smooth gradients.

| Term | Weight | Formula | File |
|------|--------|---------|------|
| Joint acceleration | $-4.5 \times 10^{-8}$ | $\sum_j \ddot{q}_j^2$ | `isaaclab/.../rewards.py:168` |
| Joint torques | $-1.1 \times 10^{-7}$ | $\sum_j \tau_j^2$ | `isaaclab/.../rewards.py:141` |
| Action rate | $-0.002$ | $\sum_i (a_{t,i} - a_{t-1,i})^2$ | base |
| Feet acceleration | $-5.0 \times 10^{-7}$ | $\sum_{\text{foot}} \|\ddot{x}_{\text{foot}}\|^2$ | `mdp/rewards.py:157` |
| Undesired contacts | $-1.0$ | $-1$ if body hits ground | base |

**Feet acceleration** (`rewards.py:157-167`):
```python
def feet_body_lin_acc_l2(env, asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    lin_acc = asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :]
    return torch.sum(torch.sum(torch.square(lin_acc), dim=-1), dim=1)
```

Encourages smooth, safe motions. Reduces sim-to-real gap.

---

## Curriculum

### Curriculum Learning Motivation

Jumping to 1m is too hard from scratch (sparse reward, exploration challenge).

**How curriculum helps:**
- Bootstrapping: Learn easy heights first, transfer to harder
- Denser rewards: Incremental progress provides learning signal
- Reduced variance: Easier tasks stabilize gradients

**Design:**
- 3cm steps prevent training collapse
- 30 consecutive successes filter luck
- Split XY/Z thresholds match controllability
- Clean episodes only (no failures allowed)

### Curriculum Implementation

`box_top_height_on_reach` (`curriculums.py:202-293`)

**Mechanism:**
1. Each env tracks integer level
2. Box top height: $h = h_{\text{start}} + \text{level} \times h_{\text{step}}$
3. Box center Z: $z = h - 0.5$
4. On reset, check previous episode success
5. If success for 30 consecutive resets → level++

**Success criteria (all required):**

$$
\begin{aligned}
\|x_{xy} - x^*_{xy}\| &< 0.30 \text{ m} \\
|x_z - x^*_z| &< 0.15 \text{ m} \\
\text{time\_out} &= \text{True} \\
\text{base\_contact} &= \text{False} \\
\text{root\_height\_below\_minimum} &= \text{False} \\
\text{bad\_pitch} &= \text{False}
\end{aligned}
$$

**Parameters:**
- Start: 0.0m
- Step: 0.03m (3cm)
- Max: 1.0m
- Min successes: 30
- XY threshold: 0.30m
- Z threshold: 0.15m

**Rationale:**
- **Split XY/Z:** Jumping has asymmetric error (XY easier to control)
- **Forbid failures:** Only clean episodes count
- **Consecutive successes:** Ensures robustness before advancing

**Terminations** (`jump_task.py:67-72`):
- `time_out`: Episode reaches 8s (valid ✓)
- `base_contact`: Body hits ground (fail)
- `root_height_below_minimum`: Base < 0.12m (fail)
- `bad_pitch`: Pitch > ±120° (fail)

---

## PPO Config

`rsl_rl_ppo_cfg.py:49-74` (DodoJumpPPORunnerCfg)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_steps_per_env` | 96 | Rollout length |
| `max_iterations` | 1500 | Training iterations |
| `empirical_normalization` | True | Normalize obs |
| `init_noise_std` | 0.35 | Exploration noise |
| `actor_hidden_dims` | [512, 256, 128] | Policy net |
| `critic_hidden_dims` | [512, 256, 128] | Value net |
| `activation` | elu | |
| `value_loss_coef` | 1.0 | |
| `clip_param` | 0.2 | PPO clip |
| `entropy_coef` | 0.003 | Entropy reg |
| `num_learning_epochs` | 5 | |
| `num_mini_batches` | 4 | |
| `learning_rate` | 3e-4 | Lower than velocity task |
| `schedule` | adaptive | KL-based |
| `gamma` | 0.99 | |
| `lam` | 0.95 | GAE |
| `desired_kl` | 0.01 | |
| `max_grad_norm` | 1.0 | |

**vs. velocity task:** Lower LR (3e-4 vs 1e-3), lower noise (0.35 vs 0.7), empirical norm enabled.

### PPO Algorithm Overview

Uses Proximal Policy Optimization (PPO) via RSL-RL:
- Actor-critic architecture (separate policy and value networks)
- Collects rollouts, computes advantages (GAE), updates policy with clipped objective
- Adaptive learning rate based on KL divergence
- Empirical normalization for stable gradients

---

## Training

### From Scratch

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Jump-Dodo-v0 \
  --num_envs=8192 \
  --log_project_name=DodoJump \
  --headless
```

### Resume

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Jump-Dodo-v0 \
  --num_envs=8192 \
  --headless \
  --resume \
  --load_run=2026-03-08_12-26-51 \
  --checkpoint=model_3798.pt
```

### Resume with Curriculum Override

When resuming from checkpoint trained at height $h$:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Jump-Dodo-v0 \
  --num_envs=8192 \
  --headless \
  --resume \
  --load_run=2026-03-08_12-26-51 \
  --checkpoint=model_3798.pt \
  env.curriculum.box_height.params.start_height=0.49 \
  env.rewards.task_pos_end_window.params.box_start_height=0.49 \
  env.rewards.exploration_bias_xy.params.box_start_height=0.49
```

**Critical:** Align all `box_start_height` params. Misalignment confuses reward scaling.

### Play at Fixed Height

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Isaac-Jump-Dodo-Play-v0 \
  --checkpoint=logs/rsl_rl/dodo_jump/2026-03-08_12-26-51/model_3798.pt \
  --num_envs=1 \
  --box_top_height=0.49
```

`--box_top_height` overrides curriculum.

---

## Troubleshooting

### Training Collapse

**Symptoms:**
- Reward/episode length drop together
- `base_contact` > 10%
- `pos_error` > 0.5m
- Curriculum advances despite poor quality

**Fixes:**
1. Rollback to stable checkpoint
2. Reduce `BOX_STEP_HEIGHT` (0.03 → 0.02)
3. Increase `min_steps` (30 → 50)
4. Lower LR
5. Check `box_start_height` alignment

### Stalling

**Symptoms:**
- Low velocity, moderate distance
- Episode times out

**Fixes:**
1. Increase `stall_penalty` weight (current: 3)
2. Increase `task_pos_end_window` weight (current: 200)
3. Check `exploration_bias_xy` active

### Head Dive

**Symptoms:**
- Excessive pitch
- High `bad_pitch` rate

**Fixes:**
1. Verify `COMMAND_TARGET_TOP_MARGIN = 0.6` (≥ robot height)
2. Increase if robot taller

### Hyperparameter Tuning

| Problem | Parameter | Direction |
|---------|-----------|-----------|
| Too conservative | `init_noise_std` | ↑ |
| Too noisy | `entropy_coef` | ↓ |
| Unstable | `learning_rate` | ↓ |
| Curriculum too fast | `step_height` | ↓ |
| Curriculum too slow | `min_steps` | ↓ |
| Task reward weak | `task_pos_end_window` | ↑ |
| Stalling | `stall_penalty` | ↑ |
| Too aggressive | Smoothness penalties | ↑ |

### Monitoring

**Warning signs:**
- `Episode_Termination/base_contact` ↑
- `Metrics/target_pos/pos_error_xy` > 0.5
- `Metrics/target_pos/pos_error_z` > 0.3
- `Mean episode length` ↓

**Healthy:**
- Gradual reward ↑
- Stable episode length
- Contact < 5%
- Steady curriculum

---

## Implementation Notes

### Design Decisions

1. **Task window reward:** Deferred signal forces reaching target before timeout
2. **Exploration bias (early):** Bootstrap movement at low heights, deactivate for precision
3. **Stall penalty:** Address observed crouching optimum
4. **Split XY/Z gates:** Asymmetric criteria match physics
5. **Forbid failures:** Only clean episodes advance curriculum
6. **Forward-biased scanner:** +0.25m offset gives advance warning
7. **Target above box:** 0.6m matches robot height, prevents head-dive

### What Worked

- Task window reward: clear optimization signal
- 3cm curriculum steps: smooth progression, avoided collapse
- Split thresholds (0.30m XY, 0.15m Z): matched task characteristics
- Stall penalty: mitigated crouching
- Feet acceleration penalty: improved landing quality

### File Locations

**Config:**
- `source/isaaclab_tasks/.../velocity/config/dodo/jump_task.py`
- `source/isaaclab_tasks/.../velocity/config/dodo/jump_terms.py`
- `source/isaaclab_tasks/.../velocity/config/dodo/agents/rsl_rl_ppo_cfg.py`

**MDP:**
- `source/isaaclab_tasks/.../velocity/mdp/rewards.py`
- `source/isaaclab_tasks/.../velocity/mdp/curriculums.py`
- `source/isaaclab_tasks/.../velocity/mdp/commands.py`
- `source/isaaclab/isaaclab/envs/mdp/rewards.py` (generic penalties)

---

## Logging

Checkpoints/metrics → `logs/rsl_rl/<experiment>/`

**Metrics:**
- Episode reward
- Position error
- Termination stats
- Curriculum level

**Weights & Biases (optional):**
```bash
export WANDB_PROJECT=DodoJump
export WANDB_ENTITY=<your_entity>
./isaaclab.sh -p ... --logger wandb ...
```

---

## Common Issues

**CUDA out of memory:** Reduce `--num_envs`

**Training diverges:** Verify `box_start_height` alignment when resuming, lower LR

**Curriculum not advancing:** Check success criteria met, monitor termination reasons

**Robot frozen in play mode:** Verify checkpoint path, network architecture matches training config

## Notes

- Training runs headless on workstation/server
- USD variants testable via `DODO_ROBOT_USD` env var
- For code customization, see `source/isaaclab_tasks/`
- All training logs saved to `logs/rsl_rl/<experiment>/`
- Checkpoints saved every 50 iterations by default
- For questions/issues, check Isaac Lab documentation or GitHub issues

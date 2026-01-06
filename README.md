# IsaacLab_Dodo

A lightweight Isaac Lab workspace customized specifically for Dodo robot training.

This repository contains only the essential components required to train and evaluate reinforcement learning policies for the Dodo robot using Isaac Lab.

---

## 🧩 Prerequisites

- Ubuntu 22.04 (Linux x64) or Windows 11 (x64)
- RAM: 32GB
- GPU VRAM: 16GB
- Python: 3.11 for Isaac Sim 5.X // 3.10 for Isaac Sim 4.X
- NVIDIA GPU with compatible driver
- Conda (recommended)
- Isaac Sim compatible GPU setup

This repository assumes that **Isaac Sim is already installed and functional**.
If not, please follow the official Isaac Sim installation guide first.

[Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

---

## ⚙️ Environment Setup

Clone this repository and set up the environment:

```bash
git clone https://github.com/DoD0d0/IsaacLab_Dodo.git
cd IsaacLab_Dodo
conda create -n isaaclab python=3.10
conda activate isaalab
pip install --upgrade pip
pip install "isaacsim[all,extscache]==4.5.0.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
sudo apt install cmake build-essential
./isaaclab.sh --install rsl_rl
pip install wandb
```

All commands below assume execution from the `IsaacLab_Dodo` root directory.

---

## 🧠 Task Overview: Velocity Locomotion (Dodo)

This project trains a **bipedal Dodo robot** to track commanded base velocities using reinforcement learning.

- Task type: Velocity tracking
- Environment variants:
  - **Flat terrain** (`DodoFlatEnvCfg`)
  - **Rough terrain** (`DodoRoughEnvCfg`)
- RL framework: **RSL-RL (PPO)**
- Control mode: **Joint position control (implicit actuators)**

The task is implemented as a **Manager-Based RL Environment** in Isaac Lab.

---

## 🤖 Robot Specification (Dodo)

- Robot type: Biped
- Asset format: USD (converted from URDF)
- Default USD path: `assets/robots/dodo/dodo.usd`
- Root prim path: `{ENV_REGEX_NS}/Robot`

### Actuation
- Control type: Joint position targets
- Actuator model: Implicit actuators
- Controlled joints:
  - `left_joint_*`
  - `right_joint_*`
- Actuator parameters:
  - Stiffness: 40.0
  - Damping: 2.0

### Initial State
- Base position: `(0.0, 0.0, 0.45)`
- Joint positions: all zeros
- Joint velocities: all zeros

---

## 🎮 Action Space

- **Type**: Joint position targets
- **Dimension**: Number of robot joints
- **Definition**:
  - Action represents **relative joint position offsets**
  - Scaled by `0.5`
  - Default joint offsets are applied

This action space is defined via:
- `JointPositionActionCfg`

---

## 🎛️ Control Abstraction Rationale

The Dodo robot is controlled using **joint position targets with implicit actuators**.

This choice reflects a trade-off between:
- Learning stability
- Sim-to-real relevance
- Policy expressiveness

### Why Joint Position Control?
- Implicit PD dynamics provide inherent stabilization
- Reduces sensitivity to high-frequency policy noise
- Simplifies reward shaping compared to torque control

While torque control offers greater expressiveness, joint position control
was chosen to prioritize stable locomotion learning in early-stage experiments.

---

## 👀 Observation Space

The policy receives a **concatenated observation vector** composed of the following terms:

### Proprioceptive Observations
- Base linear velocity (with noise)
- Base angular velocity (with noise)
- Projected gravity vector
- Joint positions (relative, noisy)
- Joint velocities (relative, noisy)
- Previous action

### Command Observations
- Target base velocity command:
  - Linear velocity (x, y)
  - Angular velocity (z)
  - Heading (optional)

### Exteroceptive Observations (Rough Terrain only)
- Height scan from ray-cast grid sensor

Noise is applied to most observations to improve robustness.
Observation corruption is **disabled during evaluation/play**.

---

## 🧠 Observation Design Considerations

The observation space is designed to balance:
- Proprioceptive feedback
- Command awareness
- Minimal exteroceptive sensing

Key design principles:
- No global position information (promotes generalization)
- Velocity-based state representation
- Height scanning enabled only in rough terrain

This encourages the policy to learn **reactive locomotion**
rather than memorizing terrain layouts.

---

## 📐 Observation & Action Dimensions

### Action Space
| Component | Dimension | Description |
|---|---:|---|
| Joint position targets | N<sub>joints</sub> | Relative joint position offsets for all leg joints |

> The exact action dimension equals the number of actuated joints defined in the Dodo USD.

---

### Observation Space (Policy)

| Observation Term | Dimension | Notes |
|---|---:|---|
| Base linear velocity | 3 | Noisy |
| Base angular velocity | 3 | Noisy |
| Projected gravity | 3 | In body frame |
| Velocity command | 3 | (v<sub>x</sub>, v<sub>y</sub>, ω<sub>z</sub>) |
| Joint positions (relative) | N<sub>joints</sub> | Noisy |
| Joint velocities (relative) | N<sub>joints</sub> | Noisy |
| Previous action | N<sub>joints</sub> | Action history |
| Height scan (rough only) | N<sub>rays</sub> | Ray-cast grid, clipped |

All observation terms are concatenated into a single flat vector before being passed to the policy.

---

## 🏞️ Environment Variants

### Flat Terrain (`DodoFlatEnvCfg`)
- Terrain type: Infinite plane
- Height scanning: Disabled
- Terrain curriculum: Disabled
- Used for:
  - Faster training
  - Debugging
  - Baseline performance

### Rough Terrain (`DodoRoughEnvCfg`)
- Terrain type: Procedurally generated rough terrain
- Height scanning: Enabled
- Terrain curriculum: Enabled
- Used for:
  - Robust locomotion learning
  - Disturbance resilience

---

## 🎯 Reward Function

The reward is a weighted sum of multiple terms.

### Task Rewards
- Linear velocity tracking (XY)
- Angular velocity tracking (Z)
- Foot air-time (encourages stepping)

### Penalties
- Vertical linear velocity
- Angular velocity in roll/pitch
- Joint torques
- Joint accelerations
- Action rate (smoothness)
- Foot sliding
- Joint deviation from default pose

### Termination Penalty
- Large negative reward when an episode terminates early

Reward weights differ slightly between flat and rough terrain variants.

---

## 📉 Why Exponential Tracking Rewards?

Velocity tracking rewards are implemented using exponential functions:

$$
r = \exp\left(-\frac{\| e \|^2}{\sigma^2}\right)
$$

This formulation provides:
- Smooth gradients near zero error
- Strong penalties for large deviations
- Bounded reward values for numerical stability

Compared to linear or L2 penalties, exponential rewards
improve early learning stability and reduce reward hacking.

---

## 🧪 Reward Ablation Notes

Several reward terms are selectively enabled or re-weighted depending on the environment variant.

- **Flat terrain**
  - Emphasizes velocity tracking and stepping regularity
  - Reduced penalties on vertical velocity and joint accelerations
  - Used for fast convergence and baseline policy learning

- **Rough terrain**
  - Stronger penalties for instability and undesired contacts
  - Height scanning and terrain curriculum enabled
  - Designed for robustness and disturbance resilience

This staged reward design allows the policy to first learn stable locomotion
and then progressively adapt to more challenging terrains.

---

## 📐 Reward Formulation (Mathematical Description)

The total reward at time step $t$ is defined as a weighted sum of individual reward and penalty terms:

$$
r_t = \sum_i w_i \, r_i(t)
$$

where $ r_i(t) $ denotes each reward component and $ w_i $ its corresponding weight.

---

### Velocity Tracking Rewards

**Linear velocity tracking (XY)**  
Encourages the robot to match the commanded planar velocity:

$$
r_{\text{lin}} = \exp\left(-\frac{\| \mathbf{v}_{xy} - \mathbf{v}_{xy}^{\text{cmd}} \|^2}{\sigma_{\text{lin}}^2}\right)
$$

**Angular velocity tracking (Z)**  
Encourages accurate yaw rate tracking:

$$
r_{\text{ang}} = \exp\left(-\frac{(\omega_z - \omega_z^{\text{cmd}})^2}{\sigma_{\text{ang}}^2}\right)
$$

---

### Gait & Contact Rewards

**Foot air-time reward**  
Encourages alternating stepping patterns:

$$
r_{\text{air}} =
\begin{cases}
\alpha \cdot t_{\text{air}}, & \text{if } v_x^{\text{cmd}} > v_{\text{th}} \\
0, & \text{otherwise}
\end{cases}
$$

This term promotes dynamic walking while suppressing unnecessary hopping when the commanded velocity is low.

---

### Stability & Smoothness Penalties

**Vertical velocity penalty**
$$
r_{v_z} = -\| v_z \|^2
$$

**Angular velocity penalty (roll/pitch)**
$$
r_{\omega_{xy}} = -\| \boldsymbol{\omega}_{xy} \|^2
$$

**Joint torque penalty**
$$
r_{\tau} = -\| \boldsymbol{\tau} \|^2
$$

**Joint acceleration penalty**
$$
r_{\ddot{q}} = -\| \ddot{\mathbf{q}} \|^2
$$

**Action rate penalty**
$$
r_{\Delta a} = -\| \mathbf{a}_t - \mathbf{a}_{t-1} \|^2
$$

---

### Termination Penalty

A large negative reward is applied when an episode terminates early:

$$
r_{\text{term}} =
\begin{cases}
-200, & \text{if episode terminates} \\
0, & \text{otherwise}
\end{cases}
$$

This discourages unstable behaviors that lead to falls or illegal contacts.


---

## ⛔ Termination Conditions

An episode terminates when any of the following occurs:

- Episode time limit reached
- Illegal contact with the robot base/body
- Robot base height falls below a minimum threshold

Additional orientation-based terminations are defined but disabled by default.

---

## 🔁 Curriculum & Randomization

### Curriculum
- Progressive terrain difficulty (rough terrain only)

### Randomization (Training only)
- Base mass variation
- Center-of-mass shifts
- External disturbances (disabled for flat terrain)

All randomization is disabled during evaluation and play.

### 🪜 Curriculum Learning Motivation

Training starts on simpler terrains and progressively increases difficulty.

Benefits:
- Faster initial convergence
- Reduced catastrophic failures early in training
- Improved robustness to terrain variation

This mirrors human motor learning, where stable gaits
are acquired before adapting to uneven surfaces.


---

## 🧪 Training Configuration (PPO)

Training is performed using **Proximal Policy Optimization (PPO)**.

### Network Architecture
- Actor: MLP with layers `[256, 128, 128]` (flat terrain)
- Critic: MLP with layers `[256, 128, 128]`
- Activation: ELU

### PPO Parameters
- Discount factor (gamma): 0.99
- GAE lambda: 0.95
- Learning rate: 1e-3 (adaptive schedule)
- Entropy coefficient: 0.008
- Clip range: 0.2

### Training Settings
- Steps per environment: 24
- Max iterations:
  - Flat: 1500
  - Rough: 3000

---

## 📦 USD Override Mechanism

The robot USD can be overridden at runtime using an environment variable:

```bash
export DODO_ROBOT_USD=dodo_1.usd
```

This allows rapid testing of:
- Different robot morphologies
- Modified mass or geometry
- Ablation studies

---

## 🤖 URDF → USD Conversion (Dodo Robot)

The Dodo robot is authored in URDF and must be converted to USD before simulation or training.

### Convert URDF to USD

example:

```bash
./isaaclab.sh -p scripts/tools/convert_urdf.py \
  assets/robots/dodo/dodobot_v3/urdf/dodo.urdf \
  assets/robots/dodo/dodo.usd
```

- Input: Dodo URDF model
- Output: USD model used by Isaac Lab
- The generated USD will be placed under assets/robots/dodo/

You can generate multiple USD variants (e.g., dodo_1.usd, dodo_3.usd) for experimentation.

---

## 🚀 Training the Dodo Robot (RL)

Training is performed using **RSL-RL (PPO)**.
Headless execution is strongly recommended.

### Run Training

example (Flat Terrain Training):

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Velocity-Flat-Dodo-v0 \
  --num_envs=8192 \
  --log_project_name=DodoFlat \
  --headless \
  --robot-usd=dodo.usd \
  --max_iterations=1500
```

#### Key Arguments

- `--task` : Registered Dodo task name
- `--num_envs` : Number of parallel simulation environments
- `--headless` : Run without GUI (recommended for training)
- `--robot-usd` : Specify which Dodo USD model to use
(default: dodo.usd)
- `--max_iterations` : Training iterations
(default is 1500 but can be changed in `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/dodo/agents/rsl_rl_ppo_cfg.py` as `max_iterations`; can be reduced here for faster experiments)
Training logs and checkpoints are saved under: `logs/rsl_rl/`

---

## 📊 Evaluating & Playing Trained Policies

After training, the policy can be evaluated or visualized using the `play.py` script.

### Play a Trained Policy (Single Environment)

example:

```bash
DODO_ROBOT_USD=dodo.usd ./isaaclab.sh \
  -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Isaac-Velocity-Flat-Dodo-v0 \
  --checkpoint=logs/rsl_rl/dodo_flat/2025-X-X_X-X-X/model_14999.pt \
  --num_envs=1
```
#### Notes

- `DODO_ROBOT_USD` overrides the default robot USD at runtime
- This mode launches Isaac Sim with rendering enabled

---

## 📈 Logging & Experiment Tracking

Training metrics and checkpoints are logged automatically during training.

### Default Logging
- Checkpoints: `logs/rsl_rl/<experiment_name>/`
- Metrics:
  - Episode reward
  - Velocity tracking error
  - Termination statistics

### Weights & Biases (Optional)

If enabled, experiments can be tracked using **Weights & Biases (W&B)**.

Example:
```bash
export WANDB_PROJECT=DodoFlat
export WANDB_ENTITY=<your_entity>
```
Then add `--logger wandb` when you run training.

---

## 📝 Notes

- Training is typically executed in headless mode on a workstation or server
- Multiple USD variants can be used to test morphology or parameter changes

For advanced customization (reward shaping, observations, domain randomization), refer to the Dodo task implementation under the project source directory.


## 📁 Repository Structure

The repository is organized to include only Dodo-related assets and tasks.

```text
IsaacLab_Dodo/
├── assets/
│   └── robots/
│       └── dodo/
│           ├── dodobot_v3/
│           │   ├── meshes/           # STL meshes
│           │   ├── urdf/             # URDF sources
│           │   └── config/
│           └── dodo.usd              # Converted USD asset
│
├── scripts/
│   └── reinforcement_learning/
│       └── rsl_rl/
│           ├── train.py              # Training entry point
│           └── play.py               # Evaluation / visualization
│
├── source/
│   ├── isaaclab_assets/
│   │   └── robots/
│   │       └── dodo.py               # Dodo articulation & actuator config
│   │
│   └── isaaclab_tasks/
│       └── manager_based/
│           └── locomotion/
│               └── velocity/
│                   ├── velocity_env_cfg.py   # Base locomotion env
│                   ├── rough_env_cfg.py      # Rough terrain config
│                   ├── flat_env_cfg.py       # Flat terrain config
│                   └── rsl_rl_ppo_cfg.py     # PPO hyperparameters
│
├── environment.yml
├── isaaclab.sh
├── README.md
└── VERSION

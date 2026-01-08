# Configuration for Dodo robot in jump-height environment. - YOU-RI

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from .rough_env_cfg import _resolve_robot_cfg


@configclass
class DodoJumpCommandsCfg:
    """Command specifications for the jump-height task."""

    base_height = mdp.UniformHeightCommandCfg(
        asset_name="robot",
        sensor_name="height_scanner",
        resampling_time_range=(2.0, 2.0),
        ranges=mdp.UniformHeightCommandCfg.Ranges(height=(0.05, 0.15)),
    )


@configclass
class DodoJumpObservationsCfg:
    """Observation specifications for the jump-height task."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        height_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_height"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class DodoJumpRewardsCfg:
    """Reward terms for the jump-height task."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_height_exp = RewTerm(
        func=mdp.track_base_height_exp,
        weight=6.0,
        params={"command_name": "base_height", "std": 0.15},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.6)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)


@configclass
class DodoJumpCurriculumCfg:
    """Curriculum terms for the jump-height task."""

    terrain_levels: CurrTerm | None = None
    height_range = CurrTerm(
        func=mdp.height_command_range,
        params={
            "command_name": "base_height",
            "min_start": 0.02,
            "min_end": 0.08,
            "max_start": 0.15,
            "max_end": 0.35,
            "start_step": 0,
            "end_step": 2_000_000,
        },
    )


@configclass
class DodoJumpEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for the Dodo jump-height environment."""

    observations: DodoJumpObservationsCfg = DodoJumpObservationsCfg()
    commands: DodoJumpCommandsCfg = DodoJumpCommandsCfg()
    rewards: DodoJumpRewardsCfg = DodoJumpRewardsCfg()
    curriculum: DodoJumpCurriculumCfg = DodoJumpCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        self.sim.use_fabric = False
        self.episode_length_s = 8.0

        self.scene.robot = _resolve_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/body_link"

        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["body_link"]
        self.events.base_com.params["asset_cfg"].body_names = ["body_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        self.terminations.base_contact.params["sensor_cfg"].body_names = ["body_link"]


@configclass
class DodoJumpEnvCfg_PLAY(DodoJumpEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

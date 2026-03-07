# Configuration terms for Dodo jump task.

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import MySceneCfg

BOX_HALF_HEIGHT = 0.5
BOX_START_TOP_HEIGHT = 0.0
BOX_STEP_HEIGHT = 0.05
# Keep command target high enough to avoid "head dive" local optimum.
COMMAND_TARGET_TOP_MARGIN = 0.6
# Use the same top-margin as the command target for curriculum reach checks.
CURRICULUM_REACH_TOP_MARGIN = COMMAND_TARGET_TOP_MARGIN
TARGET_POS_Z_OFFSET = BOX_HALF_HEIGHT + COMMAND_TARGET_TOP_MARGIN
CURRICULUM_REACH_THRESHOLD_XY = 0.30
CURRICULUM_REACH_THRESHOLD_Z = 0.15


@configclass
class DodoJumpSceneCfg(MySceneCfg):
    """Scene with a static box for the jump task."""

    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.MeshCuboidCfg(
            size=(1.5, 1.5, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )


@configclass
class DodoJumpCommandsCfg:
    """Command specs for the jump task."""

    target_pos = mdp.BoxCenterCommandCfg(
        class_type=mdp.BoxCenterCommand,
        asset_name="box",
        z_offset=TARGET_POS_Z_OFFSET,
        resampling_time_range=(1.0, 1.0),
    )


@configclass
class DodoJumpObservationsCfg:
    """Observation specs for the jump task."""

    @configclass
    class PolicyCfg(ObsGroup):
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        target_pos_b = ObsTerm(func=mdp.command_target_pos_b, params={"command_name": "target_pos"})
        time_to_go = ObsTerm(func=mdp.command_time_to_go)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 19.5},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            scale=2.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class DodoJumpRewardsCfg:
    """Reward terms for the jump task."""

    task_pos_end_window = RewTerm(
        func=mdp.track_command_pos_task_window_curriculum,
        weight=150.0,
        params={
            "command_name": "target_pos",
            # Keep the task window open in the current training horizon so the task reward is not stuck at zero.
            "reward_window_s_init": 40.0,
            "reward_window_s_final": 40.0,
            "height_start": 0.0,
            "height_full": 0.5,
            "box_start_height": BOX_START_TOP_HEIGHT,
            "box_step_height": BOX_STEP_HEIGHT,
            "ramp_power": 1.0,
        },
    )
    exploration_bias_xy = RewTerm(
        func=mdp.exploration_velocity_bias_xy,
        weight=2.0,
        params={
            "command_name": "target_pos",
            "min_distance": 0.2,
            "min_speed": 0.05,
            "deactivate_top_height": 0.2,
            "box_start_height": BOX_START_TOP_HEIGHT,
            "box_step_height": BOX_STEP_HEIGHT,
        },
    )

    # Penalties to encourage smooth and safe motions, and to mitigate sim-to-real gap by discouraging aggressive behaviors
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-4.5e-8,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_joint_.*", "right_joint_.*"])},
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.1e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_joint_.*", "right_joint_.*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body_link"]), "threshold": 1.0},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
    feet_acc_l2 = RewTerm(
        func=mdp.feet_body_lin_acc_l2,
        weight=-5.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_link_4", "right_link_4"])},
    )

    # Penalty to mitigate "stalling" local optimum where the agent stays in a crouched position close to the target without actually jumping onto the box
    stall_penalty = RewTerm(
        func=mdp.stall_penalty_far_from_target,
        weight=3,
        params={"command_name": "target_pos", "speed_threshold": 0.1, "distance_threshold": 0.4},
    )


@configclass
class DodoJumpCurriculumCfg:
    """Curriculum terms for the jump task."""

    terrain_levels: CurrTerm | None = None
    box_height = CurrTerm(
        func=mdp.box_top_height_on_reach,
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "local_xy": (0.0, 0.0),
            "start_height": BOX_START_TOP_HEIGHT,
            "step_height": BOX_STEP_HEIGHT,
            "max_height": 1.0,
            "box_half_height": BOX_HALF_HEIGHT,
            "reach_threshold_xy": CURRICULUM_REACH_THRESHOLD_XY,
            "reach_threshold_z": CURRICULUM_REACH_THRESHOLD_Z,
            "min_steps": 10,
            "z_offset": CURRICULUM_REACH_TOP_MARGIN,
            "require_time_out": True,
            "forbid_term_names": ("base_contact", "root_height_below_minimum", "bad_pitch"),
        },
    )

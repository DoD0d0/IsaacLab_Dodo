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
BOX_HALF_SIZE_X = 0.75
BOX_HALF_SIZE_Y = 0.75
BOX_START_TOP_HEIGHT = 0.0
BOX_STEP_HEIGHT = 0.1
# Keep command target high enough to avoid "head dive" local optimum.
COMMAND_TARGET_TOP_MARGIN = 0.6
# Use a slightly easier reach criterion for curriculum progression.
CURRICULUM_REACH_TOP_MARGIN = 0.2
TARGET_POS_Z_OFFSET = BOX_HALF_HEIGHT + COMMAND_TARGET_TOP_MARGIN


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
        target_pos = ObsTerm(func=mdp.generated_commands, params={"command_name": "target_pos"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 19.5},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-8.0, 8.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class DodoJumpRewardsCfg:
    """Reward terms for the jump task."""

    position_tracking = RewTerm(
        func=mdp.track_command_pos_exp_curriculum_std,
        weight=4.0,
        params={
            "command_name": "target_pos",
            "std_init": 1.0,
            "std_mid": 0.9,
            "std_final": 0.8,
            "height_mid": 0.1,
            "height_final": 0.2,
            "box_start_height": BOX_START_TOP_HEIGHT,
            "box_step_height": BOX_STEP_HEIGHT,
        },
    )
    position_progress = RewTerm(
        func=mdp.track_command_pos_progress,
        weight=3.0,
        params={"command_name": "target_pos", "clip": 0.25},
    )
    z_tracking_curriculum = RewTerm(
        func=mdp.track_command_z_exp_curriculum,
        weight=2.0,
        params={
            "command_name": "target_pos",
            "std": 0.45,
            "height_start": 0.03,
            "height_full": 0.4,
            "box_start_height": BOX_START_TOP_HEIGHT,
            "box_step_height": BOX_STEP_HEIGHT,
            "ramp_power": 2.0,
            "max_scale": 1.0,
        },
    )
    feet_to_box_top = RewTerm(
        func=mdp.track_feet_to_box_top_exp_curriculum,
        weight=1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_link_4", "right_link_4"]),
            "box_cfg": SceneEntityCfg("box"),
            "box_half_size_x": BOX_HALF_SIZE_X,
            "box_half_size_y": BOX_HALF_SIZE_Y,
            "box_half_height": BOX_HALF_HEIGHT,
            "foot_z_offset": 0.02,
            "std": 0.35,
            "min_top_height": 0.03,
            "box_start_height": BOX_START_TOP_HEIGHT,
            "box_step_height": BOX_STEP_HEIGHT,
        },
    )
    is_termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=-100.0,
        params={"term_keys": ["base_contact", "root_height_below_minimum"]},
    )
    dive_penalty = RewTerm(
        func=mdp.nose_dive_penalty_curriculum,
        weight=-0.05,
        params={
            "min_top_height": 0.05,
            "box_start_height": BOX_START_TOP_HEIGHT,
            "box_step_height": BOX_STEP_HEIGHT,
            "forward_z_tol": 0.05,
        },
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
            "reach_threshold": 0.38,
            "min_steps": 2,
            "z_offset": CURRICULUM_REACH_TOP_MARGIN,
        },
    )

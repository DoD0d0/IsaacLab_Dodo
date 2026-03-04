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
TARGET_TOP_MARGIN = 0.6
TARGET_POS_Z_OFFSET = BOX_HALF_HEIGHT + TARGET_TOP_MARGIN


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
    """Reward terms for the jump task."""

    position_tracking = RewTerm(
        func=mdp.track_command_pos_exp,
        weight=4.0,
        params={"command_name": "target_pos", "std": 1.2},
    )
    is_termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=-100.0,
        params={"term_keys": ["base_contact", "root_height_below_minimum"]},
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
            "start_height": 0.0,
            "step_height": 0.1,
            "max_height": 1.0,
            "box_half_height": BOX_HALF_HEIGHT,
            "reach_threshold": 0.3,
            "min_steps": 3,
            "z_offset": TARGET_TOP_MARGIN,
        },
    )

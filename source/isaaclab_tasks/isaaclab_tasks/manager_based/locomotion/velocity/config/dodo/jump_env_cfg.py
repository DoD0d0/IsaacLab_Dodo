# Configuration for Dodo robot in jump-height environment. - YOU-RI

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, MySceneCfg

from .rough_env_cfg import _resolve_robot_cfg


@configclass
class DodoJumpSceneCfg(MySceneCfg):
    """Scene with a static box for the jump-height task."""

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
    """Command specifications for the jump-height task."""

    target_pos = mdp.BoxCenterCommandCfg(
        class_type=mdp.BoxCenterCommand,
        asset_name="box",
        z_offset=1.1,
        resampling_time_range=(1.0, 1.0),
    )


@configclass
class DodoJumpObservationsCfg:
    """Observation specifications for the jump-height task."""

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
    """Reward terms for the jump-height task."""

    termination_base_contact = RewTerm(
        func=mdp.is_terminated_term,
        weight=-500.0,
        params={"term_keys": ["base_contact"]},
    )
    termination_root_height = RewTerm(
        func=mdp.is_terminated_term,
        weight=-400.0,
        params={"term_keys": ["root_height_below_minimum"]},
    )
    # termination_bad_pitch = RewTerm(
    #     func=mdp.is_terminated_term,
    #     weight=-400.0,
    #     params={"term_keys": ["bad_pitch"]},
    # )
    track_box_center_exp = RewTerm(
        func=mdp.track_command_pos_exp,
        weight=100.0,
        params={"command_name": "target_pos", "std": 0.6},
    )
    track_box_center_l2 = RewTerm(
        func=mdp.track_command_pos_l2,
        weight=1.0,
        params={"command_name": "target_pos"},
    )
    feet_height_to_box_top_exp = RewTerm(
        func=mdp.feet_to_box_top_height_exp_blend,
        weight=3.0,
        params={
            "box_cfg": SceneEntityCfg("box"),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_link_4", "right_link_4"]),
            "box_half_height": 0.5,
            "margin": 0.05,
            "std": 0.2,
            "start_step": 0,
            "end_step": 100000,
        },
    )
    feet_xy_to_box_center_exp = RewTerm(
        func=mdp.feet_to_box_center_xy_exp_blend,
        weight=2.0,
        params={
            "box_cfg": SceneEntityCfg("box"),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_link_4", "right_link_4"]),
            "std": 0.7,
            "start_step": 0,
            "end_step": 100000,
        },
    )
    knees_to_box_center_height_exp = RewTerm(
        func=mdp.knees_to_box_center_height_exp,
        weight=1.0,
        params={
            "box_cfg": SceneEntityCfg("box"),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_link_3", "right_link_3"]),
            "target_height": 0.4,
            "std": 0.2,
        },
    )
    base_pitch_back_penalty = RewTerm(
        func=mdp.base_pitch_back_penalty,
        weight=-3.0,
        params={"limit_angle": math.radians(30.0)},
    )
    body_contact_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-100.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body_link"]), "threshold": 1.0},
    )


@configclass
class DodoJumpCurriculumCfg:
    """Curriculum terms for the jump-height task."""

    terrain_levels: CurrTerm | None = None
    box_height = CurrTerm(
        func=mdp.box_top_height_on_reach,
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "local_xy": (0.0, 0.0),
            "start_height": 0.0,
            "step_height": 0.1,
            "max_height": 1.0,
            "box_half_height": 0.5,
            "reach_threshold": 0.5,
            "min_steps": 20,
            "z_offset": 0.5,
        },
    )


@configclass
class DodoJumpEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for the Dodo jump-height environment."""

    scene: DodoJumpSceneCfg = DodoJumpSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: DodoJumpObservationsCfg = DodoJumpObservationsCfg()
    commands: DodoJumpCommandsCfg = DodoJumpCommandsCfg()
    rewards: DodoJumpRewardsCfg = DodoJumpRewardsCfg()
    curriculum: DodoJumpCurriculumCfg = DodoJumpCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        self.sim.use_fabric = False
        self.episode_length_s = 8.0
        self.actions.joint_pos.scale = 1.0

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None

        self.scene.robot = _resolve_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/body_link"
        self.scene.height_scanner.mesh_prim_paths = ["/World/ground", "/World/envs/env_.*/Box"]

        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["body_link"]
        self.events.base_com.params["asset_cfg"].body_names = ["body_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-1.1, -1.1), "y": (0.0, 0.0), "z": (0.16, 0.16), "yaw": (0.0, 0.0)},
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
        self.terminations.root_height_below_minimum.params["minimum_height"] = 0.2
        self.terminations.bad_pitch = DoneTerm(
            func=mdp.bad_orientation,
            params={"axis": 1, "limit_angle": math.radians(120.0)},
        )


@configclass
class DodoJumpEnvCfg_PLAY(DodoJumpEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

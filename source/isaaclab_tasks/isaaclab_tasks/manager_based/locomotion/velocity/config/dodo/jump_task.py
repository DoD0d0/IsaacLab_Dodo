# Environment assembly for Dodo jump task.

import math

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from .jump_terms import (
    DodoJumpCommandsCfg,
    DodoJumpCurriculumCfg,
    DodoJumpObservationsCfg,
    DodoJumpRewardsCfg,
    DodoJumpSceneCfg,
)
from .rough_env_cfg import _resolve_robot_cfg


@configclass
class DodoJumpEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for the Dodo jump environment."""

    scene: DodoJumpSceneCfg = DodoJumpSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: DodoJumpObservationsCfg = DodoJumpObservationsCfg()
    commands: DodoJumpCommandsCfg = DodoJumpCommandsCfg()
    rewards: DodoJumpRewardsCfg = DodoJumpRewardsCfg()
    curriculum: DodoJumpCurriculumCfg = DodoJumpCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        self.sim.use_fabric = False
        self.episode_length_s = 8.0
        self.actions.joint_pos.scale = 0.45
        self.observations.policy.enable_corruption = False

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
            # Box half-length is 0.75 m; start farther from the front edge to allow approach before stepping up.
            "pose_range": {"x": (-1.0, -1.0), "y": (0.0, 0.0), "z": (0.16, 0.16), "yaw": (0.0, 0.0)},
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
        self.terminations.root_height_below_minimum.params["minimum_height"] = 0.12
        self.terminations.bad_pitch = DoneTerm(
            func=mdp.bad_orientation,
            params={"axis": 1, "limit_angle": math.radians(120.0)},
        )


@configclass
class DodoJumpEnvCfg_PLAY(DodoJumpEnvCfg):
    """Play configuration for jump task with same MDP settings as train."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

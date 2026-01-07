# Configuration for Dodo robot in flat environment for velocity locomotion task. - YOU-RI

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import DodoRoughEnvCfg


@configclass
class DodoFlatEnvCfg(DodoRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        # velocity command curriculum (ramp lin_vel_x range)
        self.curriculum.lin_vel_x = CurrTerm(
            func=mdp.lin_vel_x_range,
            params={
                "command_name": "base_velocity",
                "min_start": 0.2,
                "min_end": 0.4,
                "max_start": 0.8,
                "max_end": 1.2,
                "start_step": 0,
                "end_step": 1_500_000,
            },
        )

        # ===========================================================================

        # # Rewards (original rough terrain settings)
        # self.rewards.track_ang_vel_z_exp.weight = 1.0
        # self.rewards.lin_vel_z_l2.weight = -0.2
        # self.rewards.action_rate_l2.weight = -0.005
        # self.rewards.dof_acc_l2.weight = -1.0e-7
        # self.rewards.feet_air_time.weight = 0.75
        # self.rewards.feet_slide.weight = -0.2 # penalty for sliding
        # self.rewards.feet_air_time.params["threshold"] = 0.4
        # self.rewards.dof_torques_l2.weight = -2.0e-6
        # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=["left_joint_.*", "right_joint_.*"]
        # )
        # # Commands
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 5.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        # self.commands.base_velocity.ranges.ang_vel_z = (0, 0)

        # ===========================================================================

        # Rewards (velocity tracking)
        self.rewards.track_lin_vel_xy_exp.weight = 4.5  # linear velocity tracking
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.4  # tracking tolerance
        self.rewards.track_ang_vel_z_exp.weight = 1.6  # yaw rate tracking
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.5  # yaw tracking tolerance
        self.rewards.termination_penalty.weight = -100.0  # termination penalty
        self.rewards.flat_orientation_l2.weight = -0.8  # flat orientation penalty
        self.rewards.lin_vel_z_l2.weight = -0.5  # vertical velocity penalty
        self.rewards.action_rate_l2.weight = -0.002  # action rate penalty
        self.rewards.dof_acc_l2.weight = -5.0e-8  # joint acceleration penalty
        self.rewards.dof_torques_l2.weight = -1.0e-6  # joint torque penalty
        self.rewards.feet_air_time.weight = 0.75  # foot air time reward
        self.rewards.feet_slide.weight = -0.25  # foot sliding penalty
        self.rewards.joint_deviation_hip.weight = -0.02  # hip deviation penalty
        self.rewards.joint_deviation_l1.weight = -0.05  # joint deviation penalty
        self.rewards.ang_vel_xy_l2.weight = -0.05  # roll/pitch rate penalty
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["left_joint_.*", "right_joint_.*"]
        )
        # Commands (velocity targets)
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.8)  # forward speed range
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)  # lateral speed range
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)  # yaw rate range
        self.commands.base_velocity.rel_standing_envs = 0.0  # standstill probability
        self.commands.base_velocity.rel_heading_envs = 1.0  # heading env probability
        self.commands.base_velocity.heading_command = True  # heading control flag
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)  # heading range
        self.commands.base_velocity.resampling_time_range = (2.0, 4.0)  # command duration range

        # Actions (policy output)
        self.actions.joint_pos.scale = 0.5  # action scale


class DodoFlatEnvCfg_PLAY(DodoFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

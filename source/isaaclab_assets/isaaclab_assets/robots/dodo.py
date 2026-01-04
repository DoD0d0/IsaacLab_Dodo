"""Configuration helpers for the Dodo robot."""

from __future__ import annotations

import os
from functools import lru_cache

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Default relative path to the Dodo USD asset.
_DEFAULT_USD_REL_PATH = "assets/robots/dodo/dodo.usd"


def _resolve_usd_path(path: str) -> str:
    """Return an absolute USD path; relative paths are assumed workspace-relative."""
    if os.path.isabs(path):
        return path
    # allow users to pass just the filename (e.g. dodo_1.usd)
    candidate = path
    if not candidate.startswith("assets/"):
        candidate = os.path.join("assets/robots/dodo", candidate)
    return os.path.join(os.getcwd(), candidate)


@lru_cache(maxsize=None)
def get_dodo_cfg(usd_path: str | None = None) -> ArticulationCfg:
    """Return an articulation configuration for the Dodo robot.

    Args:
        usd_path: Optional path to a USD describing the robot. Relative paths are interpreted
            with respect to the repository root. When omitted, ``default usd`` is used.
    """
    usd_path = _DEFAULT_USD_REL_PATH if usd_path is None else usd_path
    usd_path = _resolve_usd_path(usd_path)

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,    
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.45),
            joint_pos={
                "left_joint_.*": 0.0,
                "right_joint_.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=["left_joint_.*", "right_joint_.*"],
                stiffness=40.0,
                damping=2.0,
            )
        },
    )


# Backwards-compatible constant for modules that import the original symbol.
DODO_ORIGIN_CFG = get_dodo_cfg()

"""Termination terms for jumping task."""

import torch

from mjlab.entity.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def root_height_below_minimum(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    minimum_height: float = 0.5,
) -> torch.Tensor:
    """Terminate if robot's root (torso) height falls below minimum threshold.
    
    This catches cases where the robot collapses.
    
    Args:
        env: The environment instance.
        asset_cfg: The asset configuration.
        minimum_height: Minimum allowed root height in meters (default: 0.5m).
    
    Returns:
        Tensor of shape (num_envs,) with True if root is too low, False otherwise.
    """
    asset: Entity = env.scene[asset_cfg.name]
    
    # Get root position
    root_pose = asset.data.root_link_pose_w
    root_height = root_pose[:, 2]  # Z-coordinate
    
    # Check if root height is below minimum
    too_low = root_height < minimum_height
    
    return too_low.bool()


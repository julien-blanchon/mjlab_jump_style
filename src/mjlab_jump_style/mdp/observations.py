"""Observation terms for jumping task."""

import torch

from mjlab.entity.entity import Entity
from mjlab.envs import ManagerBasedEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def base_height(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Base height above ground.
    
    Args:
        env: The environment instance.
        asset_cfg: The asset configuration.
    
    Returns:
        Tensor of shape (num_envs, 1) with base height in meters.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2:3]


def base_vertical_velocity(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Vertical (Z) component of base velocity.
    
    Args:
        env: The environment instance.
        asset_cfg: The asset configuration.
    
    Returns:
        Tensor of shape (num_envs, 1) with vertical velocity.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_w[:, 2:3]


"""Utility for converting G1 robot poses to SMPL/HumanML3D 22-joint format.

This module provides functions to extract body positions from the G1 humanoid
and map them to the 22-joint SMPL skeleton used by the MLD VAE model.
"""

from typing import Optional

import torch

from mjlab.entity.entity import Entity

# Mapping from SMPL 22-joint indices to G1 body link names
# Based on HumanML3D format described in standalone_demo README
SMPL_TO_G1_BODY_MAPPING = {
    0: "pelvis",                     # root (pelvis/root)
    1: "right_hip_roll_link",        # RH (right hip)
    2: "left_hip_roll_link",         # LH (left hip)
    3: "waist_yaw_link",             # BP (lower spine/spine1)
    4: "right_knee_link",            # RK (right knee)
    5: "left_knee_link",             # LK (left knee)
    6: "torso_link",                 # BT (upper spine/spine3)
    7: "right_ankle_roll_link",      # RMrot (right ankle/heel)
    8: "left_ankle_roll_link",       # LMrot (left ankle/heel)
    9: "torso_link",                 # BLN (lower neck) - approximate with torso
    10: "right_ankle_roll_link",     # RF (right foot tip) - use ankle
    11: "left_ankle_roll_link",      # LF (left foot tip) - use ankle
    12: "torso_link",                # BMN (mid/top neck) - approximate with torso
    13: "right_shoulder_roll_link",  # RSI (right collar)
    14: "left_shoulder_roll_link",   # LSI (left collar)
    15: "torso_link",                # BUN (head) - approximate with torso top
    16: "right_shoulder_pitch_link", # RS (right shoulder)
    17: "left_shoulder_pitch_link",  # LS (left shoulder)
    18: "right_elbow_link",          # RE (right elbow)
    19: "left_elbow_link",           # LE (left elbow)
    20: "right_wrist_yaw_link",      # RW (right wrist/hand)
    21: "left_wrist_yaw_link",       # LW (left wrist/hand)
}


def extract_smpl_joints(
    asset: Entity,
    env_ids: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Extract 22 SMPL joints from G1 body positions.
    
    Maps G1 body link positions to the 22-joint SMPL/HumanML3D format
    expected by the MLD VAE model. Some joints are approximated when
    G1 doesn't have an exact equivalent (e.g., neck subdivisions).
    
    Args:
        asset: The G1 robot entity
        env_ids: Optional tensor of environment indices. If None, extracts
                 joints for all environments.
    
    Returns:
        torch.Tensor: Joint positions in world frame.
                     Shape: (num_envs, 22, 3) if env_ids is None
                     Shape: (len(env_ids), 22, 3) if env_ids provided
    """
    # Get body positions in world frame: (num_envs, num_bodies, 3)
    body_positions = asset.data.body_link_pos_w
    
    # Find body indices for each SMPL joint
    smpl_joints = []
    for smpl_idx in range(22):
        body_name = SMPL_TO_G1_BODY_MAPPING[smpl_idx]
        body_ids, _ = asset.find_bodies([body_name])
        
        if len(body_ids) == 0:
            raise ValueError(f"Body '{body_name}' not found in G1 robot")
        
        # Get position of this body for all environments
        body_pos = body_positions[:, body_ids[0], :]  # (num_envs, 3)
        smpl_joints.append(body_pos)
    
    # Stack to create (num_envs, 22, 3)
    joints_22 = torch.stack(smpl_joints, dim=1)
    
    # Filter by environment IDs if specified
    if env_ids is not None:
        joints_22 = joints_22[env_ids]
    
    return joints_22


def joints_to_humanml3d_features(
    joints: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Convert 22-joint positions to normalized HumanML3D features.
    
    This is a placeholder that will be replaced by the actual MLD preprocessing
    pipeline. The actual conversion requires computing velocities, rotations,
    and foot contacts which is handled by standalone_demo.
    
    Args:
        joints: Joint positions (batch, frames, 22, 3)
        mean: Normalization mean (263,)
        std: Normalization std (263,)
    
    Returns:
        torch.Tensor: Normalized features (batch, frames, 263)
    """
    raise NotImplementedError(
        "This function is a placeholder. Feature extraction is handled by "
        "the MotionVAEEncoder class using standalone_demo transforms."
    )


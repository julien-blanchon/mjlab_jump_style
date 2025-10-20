"""Convert robot joint positions to HumanML3D 263-dimensional motion features.

This module implements a simplified version of the HumanML3D feature extraction
pipeline for use in real-time RL environments.
"""

import torch
import torch.nn.functional as F


def compute_motion_features(
  joints: torch.Tensor,
  njoints: int = 22,
) -> torch.Tensor:
  """Convert joint positions to HumanML3D-style motion features.

  This is a simplified feature extraction that computes the essential
  components without full quaternion/rotation representations.

  Feature breakdown (263 dimensions for HumanML3D):
  - Root rotation velocity: 1
  - Root linear velocity (XZ): 2
  - Root height (Y): 1
  - Local joint positions: 63 (21 joints × 3)
  - Joint rotations (6D): 126 (21 joints × 6)
  - Joint velocities: 66 (22 joints × 3)
  - Foot contacts: 4

  Args:
      joints: Joint positions (batch, frames, 22, 3) or (frames, 22, 3)

  Returns:
      features: Motion features (batch, frames-1, 263) or (frames-1, 263)
               Note: frames-1 because we compute velocities
  """
  # Handle batched or single sequence
  if joints.ndim == 3:
    joints = joints.unsqueeze(0)  # Add batch dimension
    was_unbatched = True
  else:
    was_unbatched = False

  batch_size, nframes, njoints_in, _ = joints.shape
  assert njoints_in == njoints, f"Expected {njoints} joints, got {njoints_in}"

  # We'll compute features for frames [0, nframes-2] using positions at [t] and [t+1]
  # This gives us (nframes-1) feature frames

  # Extract root (joint 0) and other joints
  root_pos = joints[:, :, 0, :]  # (batch, frames, 3)
  joint_pos = joints[:, :, 1:, :]  # (batch, frames, 21, 3)

  # ========== 1. Root features (4 dims) ==========

  # Root rotation velocity (1 dim) - simplified: compute yaw rotation from XY velocity
  # MuJoCo convention: X=forward/back, Y=left/right, Z=up
  root_vel_xy = (
    root_pos[:, 1:, [0, 1]] - root_pos[:, :-1, [0, 1]]
  )  # (batch, frames-1, 2)
  root_rot_vel = torch.atan2(
    root_vel_xy[:, :, 1], root_vel_xy[:, :, 0]
  )  # (batch, frames-1)
  root_rot_vel = root_rot_vel.unsqueeze(-1)  # (batch, frames-1, 1)

  # Root linear velocity in XY plane (2 dims) - horizontal movement
  # MuJoCo: Z is up, so XY is the ground plane
  root_lin_vel = root_vel_xy  # (batch, frames-1, 2)

  # Root height (1 dim) - Z coordinate (MuJoCo convention: Z is up!)
  root_height = root_pos[:, :-1, 2:3]  # (batch, frames-1, 1)

  # ========== 2. Local joint positions (63 dims) ==========

  # Convert to root-relative coordinates (rotation-invariant)
  # MuJoCo: Z is up, so we subtract root position in XY plane (ground)
  local_joints = joint_pos[:, :-1].clone()  # (batch, frames-1, 21, 3)
  local_joints[:, :, :, 0] -= root_pos[:, :-1, 0:1]  # Subtract root X
  local_joints[:, :, :, 1] -= root_pos[:, :-1, 1:2]  # Subtract root Y
  # Keep Z as is (height is important in absolute terms)

  local_joints_flat = local_joints.reshape(
    batch_size, nframes - 1, -1
  )  # (batch, frames-1, 63)

  # ========== 3. Joint rotations (126 dims) ==========

  # Simplified: Use 6D rotation representation from consecutive joint pairs
  # For each joint, compute a pseudo-rotation from bone directions
  # This is simplified - proper version would use quaternions

  # Compute bone directions (approximation of rotations)
  # Use differences between parent-child joints
  bone_directions = torch.zeros(batch_size, nframes - 1, 21, 6, device=joints.device)

  # Simplified approach: use local joint positions as rotation proxy
  # Normalize to unit vectors and expand to 6D
  local_joints_normalized = F.normalize(
    local_joints, p=2, dim=-1
  )  # (batch, frames-1, 21, 3)

  # Create 6D representation (simplified: repeat normalized vector twice)
  bone_directions[:, :, :, :3] = local_joints_normalized
  bone_directions[:, :, :, 3:] = local_joints_normalized

  bone_directions_flat = bone_directions.reshape(
    batch_size, nframes - 1, -1
  )  # (batch, frames-1, 126)

  # ========== 4. Joint velocities (66 dims) ==========

  # Compute velocities for all joints (including root)
  joint_velocities = joints[:, 1:] - joints[:, :-1]  # (batch, frames-1, 22, 3)
  joint_velocities_flat = joint_velocities.reshape(
    batch_size, nframes - 1, -1
  )  # (batch, frames-1, 66)

  # ========== 5. Foot contacts (4 dims) ==========

  # Simplified foot contact detection based on:
  # - Low foot height (Z coordinate in MuJoCo)
  # - Low foot velocity

  left_foot = joints[:, :, 11, :]  # LF (joint 11)
  right_foot = joints[:, :, 10, :]  # RF (joint 10)

  # Compute foot heights and velocities
  # MuJoCo: Z is up (index 2)
  left_foot_height = left_foot[:, :-1, 2]  # (batch, frames-1)
  right_foot_height = right_foot[:, :-1, 2]

  left_foot_vel = torch.norm(
    left_foot[:, 1:] - left_foot[:, :-1], dim=-1
  )  # (batch, frames-1)
  right_foot_vel = torch.norm(right_foot[:, 1:] - right_foot[:, :-1], dim=-1)

  # Contact if height < 0.05m AND velocity < 0.02 m/step
  left_contact = ((left_foot_height < 0.05) & (left_foot_vel < 0.02)).float()
  right_contact = ((right_foot_height < 0.05) & (right_foot_vel < 0.02)).float()

  # Two contacts per foot (toe and heel) - duplicate for simplicity
  foot_contacts = torch.stack(
    [left_contact, left_contact, right_contact, right_contact], dim=-1
  )  # (batch, frames-1, 4)

  # ========== Concatenate all features ==========

  features = torch.cat(
    [
      root_rot_vel,  # 1
      root_lin_vel,  # 2
      root_height,  # 1
      local_joints_flat,  # 63
      bone_directions_flat,  # 126
      joint_velocities_flat,  # 66
      foot_contacts,  # 4
    ],
    dim=-1,
  )  # (batch, frames-1, 263)

  # Remove batch dimension if input was unbatched
  if was_unbatched:
    features = features.squeeze(0)

  return features


def normalize_features(
  features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
  """Normalize features using dataset statistics.

  Args:
      features: Raw features (batch, frames, 263)
      mean: Mean normalization values (263,)
      std: Std normalization values (263,)

  Returns:
      Normalized features (batch, frames, 263)
  """
  return (features - mean) / (std + 1e-8)

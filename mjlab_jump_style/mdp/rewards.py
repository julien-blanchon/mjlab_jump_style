"""Reward terms for jumping task."""

import torch

from mjlab.entity.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.motion_encoding import extract_smpl_joints
from mjlab_jump_style.utils.t2m_motion_encoder import T2MMotionEncoder

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

from mjlab_jump_style.utils.motion_vae_encoder import MotionVAEEncoder


def vertical_velocity_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  threshold: float = 1.5,
) -> torch.Tensor:
  """Reward for upward velocity during takeoff phase.

  Encourages the robot to generate upward momentum for jumping.

  Args:
      env: The environment instance.
      asset_cfg: The asset configuration.
      threshold: Target vertical velocity (m/s).

  Returns:
      Tensor of shape (num_envs,) with reward values.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # Get vertical velocity
  vertical_vel = asset.data.root_link_lin_vel_w[:, 2]

  # Reward positive vertical velocity, saturating at threshold
  reward = torch.clamp(vertical_vel / threshold, min=0.0, max=1.0)

  return reward


def apex_height_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  target_height: float = 0.3,
) -> torch.Tensor:
  """Reward for reaching target jump height above initial position.

  Args:
      env: The environment instance.
      asset_cfg: The asset configuration.
      target_height: Target height above standing (meters).

  Returns:
      Tensor of shape (num_envs,) with reward values.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # Get current height
  current_height = asset.data.root_link_pos_w[:, 2]

  # Estimate standing height (approximate from default state)
  standing_height = asset.data.default_root_state[:, 2]

  # Height above standing
  height_gain = current_height - standing_height

  # Reward based on proximity to target
  reward = torch.exp(-torch.abs(height_gain - target_height) / 0.2)

  # Only give reward if actually elevated
  reward = torch.where(height_gain > 0.05, reward, torch.zeros_like(reward))

  return reward


def landing_stability(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for smooth landing with low impact.

  Penalizes high downward velocity near ground and rewards
  being stationary on ground.

  Args:
      env: The environment instance.
      asset_cfg: The asset configuration.

  Returns:
      Tensor of shape (num_envs,) with reward values.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # Get height and vertical velocity
  height = asset.data.root_link_pos_w[:, 2]
  vertical_vel = asset.data.root_link_lin_vel_w[:, 2]

  # Check if near ground (within 0.1m of standing height)
  standing_height = asset.data.default_root_state[:, 2]
  near_ground = height < (standing_height + 0.1)

  # Reward low velocity when near ground
  landing_reward = torch.exp(-torch.abs(vertical_vel) / 0.5)

  # Only apply when near ground
  reward = torch.where(near_ground, landing_reward, torch.zeros_like(landing_reward))

  return reward


def upright_posture_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for maintaining upright orientation during jump.

  Args:
      env: The environment instance.
      asset_cfg: The asset configuration.

  Returns:
      Tensor of shape (num_envs,) with reward values.
  """
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity = asset.data.projected_gravity_b

  # Ideal projected gravity is [0, 0, -1] when upright
  upright_error = torch.abs(projected_gravity[:, 2] + 1.0)

  return torch.exp(-upright_error / 0.25)


def feet_clearance_during_jump(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for having both feet off ground during jump.

  Requires foot contact sensors to be configured.

  Args:
      env: The environment instance.
      asset_cfg: The asset configuration.

  Returns:
      Tensor of shape (num_envs,) with reward values.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # Check if contact sensors exist
  if "left_foot_ground_contact" not in asset.data.sensor_data:
    # No sensors configured, return zero reward
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  # Get contact sensor data for both feet
  left_foot_contact = asset.data.sensor_data["left_foot_ground_contact"][:, 0] > 0
  right_foot_contact = asset.data.sensor_data["right_foot_ground_contact"][:, 0] > 0

  # Both feet off ground
  both_off = ~left_foot_contact & ~right_foot_contact

  # Get height to ensure we're actually jumping, not just lifting feet while standing
  height = asset.data.root_link_pos_w[:, 2]
  standing_height = asset.data.default_root_state[:, 2]
  elevated = height > (standing_height + 0.05)

  # Reward only if both conditions met
  reward = (both_off & elevated).float()

  return reward


class vae_motion_similarity:
  """Reward based on VAE latent similarity to reference jump motion.

  This class:
  1. Maintains a MotionVAEEncoder instance
  2. Updates buffer each step with current robot pose
  3. Periodically computes latent encoding and distance
  4. Returns reward based on similarity (lower distance = higher reward)
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    """Initialize the VAE motion similarity reward.

    Args:
        cfg: Reward term configuration
        env: Environment instance
    """
    # Get parameters
    self.asset_cfg = cfg.params["asset_cfg"]
    self.weight = cfg.weight
    reference_latent_path = cfg.params.get("reference_latent_path")
    buffer_length = cfg.params.get("buffer_length", 300)
    target_frames = cfg.params.get("target_frames", 120)
    vae_update_interval = cfg.params.get("vae_update_interval", 10)

    # Only load VAE if weight > 0 (skip during evaluation to save time/memory)
    if self.weight > 0:
      # Create VAE encoder
      self.vae_encoder = MotionVAEEncoder(
        num_envs=env.num_envs,
        buffer_length=buffer_length,
        target_frames=target_frames,
        device=env.device,
        vae_update_interval=vae_update_interval,
      )

      print("[vae_motion_similarity] Initialized with:")
      print(f"  - buffer_length: {buffer_length}")
      print(f"  - target_frames: {target_frames}")
      print(f"  - vae_update_interval: {vae_update_interval}")
      print(f"  - reference_latent_path: {reference_latent_path}")
    else:
      self.vae_encoder = None
      print("[vae_motion_similarity] Disabled (weight=0, skipping VAE loading)")

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Compute VAE similarity reward.

    Args:
        env: Environment instance
        **kwargs: Additional parameters (unused)

    Returns:
        Tensor of shape (num_envs,) with reward values
    """
    # If VAE is disabled (weight=0), return zeros
    if self.vae_encoder is None:
      return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    # Extract SMPL joints from G1
    asset: Entity = env.scene[self.asset_cfg.name]
    joints_22 = extract_smpl_joints(asset)

    # Update buffer
    self.vae_encoder.update(joints_22)

    # Compute distances (returns cached if not update step)
    distances = self.vae_encoder.compute_latent_distances()

    # Convert distance to reward
    # Based on notebook: jump ~12.6, walk ~42.3, mean variation ~25.3
    # Good jump should be < 20, poor jump > 30
    # Using exponential falloff with scale of 15.0
    reward = torch.exp(-distances / 15.0)

    return reward

  def reset(self, env_ids: torch.Tensor, env: ManagerBasedRlEnv) -> None:
    """Reset buffer for specific environments.

    Args:
        env_ids: Environment indices to reset
        env: Environment instance
    """
    if self.vae_encoder is not None:
      self.vae_encoder.reset_buffer(env_ids)


class t2m_motion_similarity:
  """Reward based on T2M embedding similarity to reference jump motion.

  This uses T2M (Text-to-Motion) embeddings instead of VAE latents for
  MUCH BETTER discrimination (9.24x better separation between motion classes!).

  Key advantages:
  - 9.24x better separation than VAE latents
  - More consistent and robust
  - Specifically trained for motion understanding
  - Used in MLD evaluation metrics

  This class:
  1. Maintains a T2MMotionEncoder instance
  2. Updates buffer each step with current robot pose
  3. Periodically computes T2M embedding and distance
  4. Returns reward based on similarity (lower distance = higher reward)
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    """Initialize the T2M motion similarity reward.

    Args:
        cfg: Reward term configuration
        env: Environment instance
    """
    # Get parameters
    self.asset_cfg = cfg.params["asset_cfg"]
    self.weight = cfg.weight
    buffer_length = cfg.params.get("buffer_length", 300)
    target_frames = cfg.params.get("target_frames", 120)
    update_interval = cfg.params.get("update_interval", 10)

    # Only load T2M if weight > 0 (skip during evaluation to save time/memory)
    if self.weight > 0:
      # Create T2M encoder
      self.t2m_encoder = T2MMotionEncoder(
        num_envs=env.num_envs,
        buffer_length=buffer_length,
        target_frames=target_frames,
        device=env.device,
        update_interval=update_interval,
      )

      print("[t2m_motion_similarity] Initialized with:")
      print(f"  - buffer_length: {buffer_length}")
      print(f"  - target_frames: {target_frames}")
      print(f"  - update_interval: {update_interval}")
      print("  - Using T2M embeddings (9.24x better than VAE!)")
    else:
      self.t2m_encoder = None
      print("[t2m_motion_similarity] Disabled (weight=0, skipping T2M loading)")

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Compute T2M similarity reward.

    Args:
        env: Environment instance
        **kwargs: Additional parameters (unused)

    Returns:
        Tensor of shape (num_envs,) with reward values
    """
    # If T2M is disabled (weight=0), return zeros
    if self.t2m_encoder is None:
      return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    # Extract SMPL joints from G1
    asset: Entity = env.scene[self.asset_cfg.name]
    joints_22 = extract_smpl_joints(asset)

    # Update buffer
    self.t2m_encoder.update(joints_22)

    # Compute distances (returns cached if not update step)
    distances = self.t2m_encoder.compute_embedding_distances()

    # Convert distance to reward
    # Based on comparison results with T2M embeddings:
    # - Test jump: ~5.5
    # - Test walk: ~11.8
    # - Jump variations: 0.3-0.9 (mean 0.6)
    # Good jump should be < 6, poor jump > 10
    # Using exponential falloff with scale of 5.0
    # Update to a better scale: 3.0 more aggressive falloff
    reward = torch.exp(-distances / 3.0)

    return reward

  def reset(self, env_ids: torch.Tensor, env: ManagerBasedRlEnv) -> None:
    """Reset buffer for specific environments.

    Args:
        env_ids: Environment indices to reset
        env: Environment instance
    """
    if self.t2m_encoder is not None:
      self.t2m_encoder.reset_buffer(env_ids)

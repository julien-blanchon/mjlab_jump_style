"""VAE-based motion encoding for computing latent similarity rewards.

This module provides a buffer-based encoder that:
1. Maintains a rolling buffer of robot poses over time
2. Downsamples from mjlab frequency (50Hz) to MLD frequency (20fps)
3. Converts poses to HumanML3D features using standalone_demo
4. Encodes features to latent space using the MLD VAE
5. Computes L2 distance to a reference latent
"""

from typing import Optional

import numpy as np
import torch
from motion_latent_diffusion_standalone.models import MotionVAE
from motion_latent_diffusion_standalone.transforms import MotionTransform

# Import our feature extraction
from mjlab.utils.motion_features import compute_motion_features, normalize_features


class MotionVAEEncoder:
  """Encodes robot motion sequences to VAE latents and computes similarity.

  This class maintains a rolling buffer of robot poses, downsamples them to
  the MLD expected frame rate, converts to motion features, and encodes to
  latent space for similarity comparison.
  """

  def __init__(
    self,
    num_envs: int,
    buffer_length: int = 300,
    target_frames: int = 120,
    device: str = "cuda",
    reference_latent_path: Optional[str] = None,
    vae_update_interval: int = 10,
  ):
    """Initialize the motion VAE encoder.

    Args:
        num_envs: Number of parallel environments
        buffer_length: Length of pose buffer in steps (default: 300 for 6s at 50Hz)
        target_frames: Target number of frames for VAE (default: 120 for 6s at 20fps)
        device: Device for computation ("cuda" or "cpu")
        reference_latent_path: Path to reference latent .pt file
        vae_update_interval: Update VAE encoding every N steps (default: 10)
        vae_checkpoint: Optional path to VAE checkpoint (uses default if None)
        vae_resources_dir: Optional path to resources directory (auto-downloads if None)
    """
    self.num_envs = num_envs
    self.buffer_length = buffer_length
    self.target_frames = target_frames
    self.device = device
    self.vae_update_interval = vae_update_interval

    # Initialize step counter for update interval
    self.step_counter = 0

    # Initialize pose buffer: (num_envs, buffer_length, 22, 3)
    self.pose_buffer = torch.zeros(
      (num_envs, buffer_length, 22, 3), dtype=torch.float32, device=device
    )
    self.buffer_index = 0  # Current position in circular buffer
    self.buffer_full = False

    # Cache for last computed distances
    self.cached_distances = torch.zeros(num_envs, dtype=torch.float32, device=device)

    # Load VAE model
    self._load_vae()

    # Load reference latent
    self._load_reference_latent(reference_latent_path)

    # Compute downsampling indices: 50Hz -> 20fps
    # 50Hz to 20fps = 2.5x downsampling
    # For 120 frames at 20fps, we need 300 steps at 50Hz
    self.downsample_indices = self._compute_downsample_indices()

  def _load_vae(self):
    """Load the MLD VAE model."""
    print("[MotionVAEEncoder] Loading MLD VAE model...")

    self.vae = MotionVAE.from_pretrained(
      "blanchon/motion-latent-diffusion-standalone-vae"
    )

    self.vae.to(self.device)

    # Load motion transform (for denormalization/normalization)
    mean = self.vae.mean
    std = self.vae.std
    self.motion_transform = MotionTransform(mean, std, njoints=22)

    # Store mean/std as tensors for feature normalization
    self.mean = torch.from_numpy(mean).float().to(self.device)
    self.std = torch.from_numpy(std).float().to(self.device)

    print(f"[MotionVAEEncoder] VAE loaded successfully on {self.device}")

  def _load_reference_latent(self, latent_path: Optional[str]):
    """Load reference latent for similarity comparison."""
    if latent_path is None:
      print("[MotionVAEEncoder] Warning: No reference latent provided")
      self.reference_latent = None
      return

    print(f"[MotionVAEEncoder] Loading reference latent from {latent_path}")
    self.reference_latent = torch.load(latent_path, map_location=self.device)

    # Ensure it's the right shape: (1, 1, 256)
    if self.reference_latent.shape != (1, 1, 256):
      print(
        f"[MotionVAEEncoder] Warning: Expected shape (1, 1, 256), got {self.reference_latent.shape}"
      )

    print(f"[MotionVAEEncoder] Reference latent loaded: {self.reference_latent.shape}")

  def _compute_downsample_indices(self) -> list:
    """Compute indices for downsampling from 50Hz to 20fps.

    50Hz to 20fps = 2.5x downsampling
    For target_frames at 20fps, we need buffer_length steps at 50Hz

    Returns:
        List of indices to sample from buffer
    """
    # Generate evenly spaced indices
    indices = np.linspace(0, self.buffer_length - 1, self.target_frames, dtype=int)
    return indices.tolist()

  def update(self, joints_22: torch.Tensor) -> None:
    """Update buffer with new joint positions.

    Args:
        joints_22: Joint positions (num_envs, 22, 3)
    """
    # Add to circular buffer
    self.pose_buffer[:, self.buffer_index, :, :] = joints_22

    # Increment buffer index
    self.buffer_index = (self.buffer_index + 1) % self.buffer_length

    # Mark buffer as full once we've wrapped around
    if self.buffer_index == 0:
      self.buffer_full = True

    # Increment step counter
    self.step_counter += 1

  @torch.no_grad()
  def compute_latent_distances(
    self, env_ids: Optional[torch.Tensor] = None
  ) -> torch.Tensor:
    """Compute L2 distance from current motion to reference latent.

    Only computes if:
    1. Buffer is full
    2. Reference latent is loaded
    3. Update interval has been reached (or forced)

    Otherwise returns cached distances.

    Args:
        env_ids: Optional environment IDs to compute for (not used, computes for all)

    Returns:
        torch.Tensor: L2 distances for all environments (num_envs,)
    """
    # Return cached if not time to update
    if self.step_counter % self.vae_update_interval != 0:
      return self.cached_distances

    # Return zeros if buffer not full or no reference
    if not self.buffer_full or self.reference_latent is None:
      return self.cached_distances

    # Downsample buffer to target frame rate
    # Buffer is circular, so we need to read from current position backwards
    if self.buffer_index == 0:
      # Buffer is exactly full, read in order
      downsampled = self.pose_buffer[:, self.downsample_indices, :, :]
    else:
      # Need to reorder circular buffer
      # Reorder so oldest frame is first
      reordered = torch.cat(
        [
          self.pose_buffer[:, self.buffer_index :, :, :],
          self.pose_buffer[:, : self.buffer_index, :, :],
        ],
        dim=1,
      )
      downsampled = reordered[:, self.downsample_indices, :, :]

    # downsampled shape: (num_envs, target_frames, 22, 3)

    # Convert joints to HumanML3D features
    # compute_motion_features returns (batch, frames-1, 263) since it computes velocities
    # So we get 119 feature frames from 120 joint frames
    features = compute_motion_features(downsampled, njoints=22)
    # features shape: (num_envs, target_frames-1, 263)

    # Normalize features using dataset statistics
    features = normalize_features(features, self.mean, self.std)

    # Encode to latent using VAE
    # VAE expects (batch, frames, nfeats) and lengths
    # Note: features has target_frames-1 since we compute velocities
    actual_length = self.target_frames - 1
    lengths = [actual_length] * self.num_envs
    latents, dist = self.vae.encode(features, lengths)  # Returns (latent, distribution)

    # IMPORTANT: Use mu (mean) instead of sampled latent for deterministic rewards
    # Sampling introduces noise which is bad for RL training stability
    # dist.mean gives us the deterministic expected latent representation
    mu = dist.mean  # Shape: (latent_size, num_envs, latent_dim)
    mu = mu.permute(1, 0, 2)  # (num_envs, latent_size, latent_dim) = (num_envs, 1, 256)

    # Compute L2 distance to reference using mu (not sampled latent)
    # reference_latent: (1, 1, 256)
    # mu: (num_envs, 1, 256)
    distances = torch.norm(mu - self.reference_latent, p=2, dim=(1, 2))

    # Cache and return
    self.cached_distances = distances
    return distances

  def reset_buffer(self, env_ids: Optional[torch.Tensor] = None):
    """Reset buffer for specific environments.

    Args:
        env_ids: Environment IDs to reset. If None, resets all.
    """
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)

    self.pose_buffer[env_ids] = 0.0

    # Note: We don't reset buffer_index or buffer_full as these are global
    # This means after reset, we'll have a mix of old and new data
    # For proper reset, consider resetting per-env buffer state

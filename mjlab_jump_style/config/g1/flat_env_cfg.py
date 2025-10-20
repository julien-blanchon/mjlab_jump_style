"""G1-specific configuration for jumping task."""

from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE, G1_ROBOT_CFG
from mjlab.tasks.jumping.jumping_env_cfg import JumpingEnvCfg
from mjlab.utils.spec_config import ContactSensorCfg


@dataclass
class G1JumpingEnvCfg(JumpingEnvCfg):
  def __post_init__(self):
    # Configure contact sensors for feet
    foot_contact_sensors = [
      ContactSensorCfg(
        name=f"{side}_foot_ground_contact",
        body1=f"{side}_ankle_roll_link",
        body2="terrain",
        num=1,
        data=("found",),
        reduce="netforce",
      )
      for side in ["left", "right"]
    ]
    
    g1_cfg = replace(G1_ROBOT_CFG, sensors=tuple(foot_contact_sensors))
    
    self.scene.entities = {"robot": g1_cfg}
    self.actions.joint_pos.scale = G1_ACTION_SCALE
    
    # Set T2M reference embedding path (9.24x better discrimination than VAE!)
    self.rewards.t2m_similarity.params["reference_embedding_path"] = (
      "/workspace/ai-toolkit/mjlab/outputs/jump_t2m/jump_mean.t2m_emb.pt"
    )

    self.viewer.body_name = "torso_link"


@dataclass
class G1JumpingEnvCfg_PLAY(G1JumpingEnvCfg):
  def __post_init__(self):
    super().__post_init__()
    
    # Disable observation corruption for evaluation
    self.observations.policy.enable_corruption = False
    
    # Disable T2M reward during evaluation (not needed for policy execution)
    # The policy is already trained - we don't need to compute motion similarity
    self.rewards.t2m_similarity.weight = 0.0
    
    # Effectively infinite episode length for evaluation
    self.episode_length_s = int(1e9)


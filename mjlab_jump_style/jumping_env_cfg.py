"""Jumping task configuration.

This module defines the base configuration for jumping tasks.
Robot-specific configurations are located in the config/ directory.
"""

import math
from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import EventTermCfg as EventTerm
from mjlab.managers.manager_term_config import ObservationGroupCfg as ObsGroup
from mjlab.managers.manager_term_config import ObservationTermCfg as ObsTerm
from mjlab.managers.manager_term_config import RewardTermCfg as RewardTerm
from mjlab.managers.manager_term_config import TerminationTermCfg as DoneTerm
from mjlab.managers.manager_term_config import term
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.jumping import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

##
# Scene.
##

SCENE_CFG = SceneCfg(
  terrain=TerrainImporterCfg(terrain_type="plane"),
  num_envs=1,
  extent=2.0,
)

VIEWER_CONFIG = ViewerConfig(
  origin_type=ViewerConfig.OriginType.ASSET_BODY,
  asset_name="robot",
  body_name="",  # Override in robot cfg.
  distance=3.0,
  elevation=-10.0,
  azimuth=90.0,
)

##
# MDP.
##


@dataclass
class ActionCfg:
  joint_pos: mdp.JointPositionActionCfg = term(
    mdp.JointPositionActionCfg,
    asset_name="robot",
    actuator_names=[".*"],
    scale=0.5,
    use_default_offset=True,
  )


@dataclass
class ObservationCfg:
  @dataclass
  class PolicyCfg(ObsGroup):
    base_lin_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.base_lin_vel,
      noise=Unoise(n_min=-0.1, n_max=0.1),
    )
    base_ang_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.base_ang_vel,
      noise=Unoise(n_min=-0.2, n_max=0.2),
    )
    projected_gravity: ObsTerm = term(
      ObsTerm,
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    joint_pos: ObsTerm = term(
      ObsTerm,
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    joint_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    )
    actions: ObsTerm = term(ObsTerm, func=mdp.last_action)

    def __post_init__(self):
      self.enable_corruption = True

  @dataclass
  class PrivilegedCfg(PolicyCfg):
    def __post_init__(self):
      super().__post_init__()
      self.enable_corruption = False

  policy: PolicyCfg = field(default_factory=PolicyCfg)
  critic: PrivilegedCfg = field(default_factory=PrivilegedCfg)


@dataclass
class EventCfg:
  reset_base: EventTerm = term(
    EventTerm,
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {"yaw": (-0.2, 0.2)},
      "velocity_range": {},
    },
  )
  reset_robot_joints: EventTerm = term(
    EventTerm,
    func=mdp.reset_joints_by_scale,
    mode="reset",
    params={
      "position_range": (0.8, 1.2),  # Allow some variation in starting pose
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
    },
  )


@dataclass
class RewardCfg:
  # T2M-based motion similarity (9.24x better discrimination than VAE!)
  t2m_similarity: RewardTerm = term(
    RewardTerm,
    func=mdp.t2m_motion_similarity,
    # weight=5.0,  # Main motion quality objective
    weight=4,  # Increased from 5.0 to better match T2M embeddings
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "reference_embedding_path": "",  # Set in G1 config
      "t2m_checkpoint_path": None,  # Auto-finds checkpoint
      "buffer_length": 300,  # 6s at 50Hz
      "target_frames": 120,  # 20fps for 6s
      "update_interval": 10,  # Compute every 10 steps
    },
  )
  
  # Basic jumping rewards (rebalanced for use with T2M)
  vertical_velocity: RewardTerm = term(
    RewardTerm,
    func=mdp.vertical_velocity_reward,
    weight=0.5,  # Reduced from 3.0 now that T2M is providing motion guidance
    params={"threshold": 1.5},
  )
  
  apex_height: RewardTerm = term(
    RewardTerm,
    func=mdp.apex_height_reward,
    weight=0.5,  # Reduced from 8.0 - T2M handles motion quality
    params={"target_height": 0.2},
  )
  
  landing_stability: RewardTerm = term(
    RewardTerm,
    func=mdp.landing_stability,
    weight=0.1,  # Increased from 0.2 - important for complete jumps
  )
  
  upright_posture: RewardTerm = term(
    RewardTerm,
    func=mdp.upright_posture_reward,
    weight=0.1,
  )
  
  feet_clearance: RewardTerm = term(
    RewardTerm,
    func=mdp.feet_clearance_during_jump,
    weight=0.5,  # Reduced from 2.0
  )
  
  # Penalties
  action_rate_l2: RewardTerm = term(
    RewardTerm,
    func=mdp.action_rate_l2,
    weight=-0.10,
  )
  
  dof_pos_limits: RewardTerm = term(
    RewardTerm,
    func=mdp.joint_pos_limits,
    weight=-0.5,
  )


@dataclass
class TerminationCfg:
  # Time out after episode length
  time_out: DoneTerm = term(DoneTerm, func=mdp.time_out, time_out=True)
  
  # Terminate if robot tilts too much
  fell_over: DoneTerm = term(
    DoneTerm,
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )


##
# Environment.
##

SIM_CFG = SimulationCfg(
  nconmax=140_000,
  njmax=300,
  mujoco=MujocoCfg(
    timestep=0.005,
    iterations=10,
    ls_iterations=20,
  ),
)


@dataclass
class JumpingEnvCfg(ManagerBasedRlEnvCfg):
  scene: SceneCfg = field(default_factory=lambda: SCENE_CFG)
  observations: ObservationCfg = field(default_factory=ObservationCfg)
  actions: ActionCfg = field(default_factory=ActionCfg)
  rewards: RewardCfg = field(default_factory=RewardCfg)
  events: EventCfg = field(default_factory=EventCfg)
  terminations: TerminationCfg = field(default_factory=TerminationCfg)
  commands: None = None  # No commands for jumping task
  curriculum: None = None  # No curriculum for jumping task
  sim: SimulationCfg = field(default_factory=lambda: SIM_CFG)
  viewer: ViewerConfig = field(default_factory=lambda: VIEWER_CONFIG)
  decimation: int = 4  # 50 Hz control frequency
  episode_length_s: float = 5.0  # Short episodes for jumping


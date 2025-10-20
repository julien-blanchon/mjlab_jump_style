"""Gym environment registration for G1 jumping task."""

import gymnasium as gym

gym.register(
  id="Mjlab-Jumping-Flat-Unitree-G1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1JumpingEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:G1JumpingPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Jumping-Flat-Unitree-G1-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1JumpingEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:G1JumpingPPORunnerCfg",
  },
)


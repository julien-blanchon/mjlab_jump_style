"""Event terms for jumping task."""

import torch

from mjlab.envs import ManagerBasedEnv


def randomize_initial_crouch(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
) -> None:
    """Randomize initial knee bend to vary starting pose for jumping.
    
    This gives the robot different starting configurations, from nearly
    straight legs to deep crouch, to learn jumping from various poses.
    
    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
    """
    # Get robot asset
    robot = env.scene["robot"]
    
    # Find knee and hip joints
    knee_ids, _ = robot.find_joints([".*_knee_joint"])
    hip_pitch_ids, _ = robot.find_joints([".*_hip_pitch_joint"])
    ankle_pitch_ids, _ = robot.find_joints([".*_ankle_pitch_joint"])
    
    # Get default positions
    default_pos = robot.data.default_joint_pos[env_ids]
    
    # Randomize knee flexion: 0.2 to 0.8 radians (slight bend to deep crouch)
    knee_flex = torch.rand(len(env_ids), len(knee_ids), device=env.device) * 0.6 + 0.2
    
    # Set knee positions
    for i, knee_id in enumerate(knee_ids):
        default_pos[:, knee_id] = knee_flex[:, i]
    
    # Adjust hip pitch to compensate (negative to lean forward)
    hip_compensation = -knee_flex[:, 0] * 0.5
    for hip_id in hip_pitch_ids:
        default_pos[:, hip_id] = hip_compensation
    
    # Adjust ankle pitch to compensate
    ankle_compensation = -knee_flex[:, 0] * 0.4
    for ankle_id in ankle_pitch_ids:
        default_pos[:, ankle_id] = ankle_compensation
    
    # Write joint positions
    robot.data.joint_pos[env_ids] = default_pos
    robot.data.joint_vel[env_ids] = 0.0


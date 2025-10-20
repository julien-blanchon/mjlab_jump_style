# G1 Jumping Task with VAE Motion Similarity

A reinforcement learning task for training the Unitree G1 humanoid robot to perform jumping motions with realistic motion patterns using VAE-based similarity rewards.

## Overview

This task combines traditional RL rewards (vertical velocity, height, landing stability) with a sophisticated VAE-based reward that compares the robot's motion to reference jump motions in latent space.

### Key Features

- **VAE Motion Similarity**: Uses the Motion Latent Diffusion (MLD) VAE to encode robot motion sequences and compute similarity to reference jump patterns
- **Multi-objective Rewards**: Balances jump height, takeoff velocity, landing stability, and motion naturalness
- **Rolling Buffer**: Maintains 6 seconds of motion history for VAE encoding
- **Configurable Update Frequency**: VAE encoding can be computed at adjustable intervals (default: every 10 steps)

## Architecture

### Reward Components

1. **VAE Similarity (weight: 5.0)** - Main objective

   - Encodes robot motion using MLD VAE
   - Computes L2 distance to reference "jump" latent
   - Reference: mean of 20 jump motion variations
   - Good jump: distance < 20, Bad: distance > 30

2. **Vertical Velocity (weight: 1.0)**

   - Rewards upward velocity during takeoff
   - Threshold: 1.5 m/s

3. **Apex Height (weight: 2.0)**

   - Rewards reaching target jump height
   - Target: 0.3m above standing

4. **Landing Stability (weight: 1.0)**

   - Rewards smooth landing with low impact
   - Penalizes high downward velocity near ground

5. **Upright Posture (weight: 0.5)**

   - Maintains upright orientation during jump

6. **Feet Clearance (weight: 0.5)**

   - Rewards both feet off ground during jump

7. **Penalties**
   - Action rate L2: -0.01
   - Joint position limits: -1.0

### Motion Processing Pipeline

1. **Extract SMPL Joints**: G1 body positions → 22-joint SMPL format
2. **Buffer Management**: Maintain 300 steps (6s at 50Hz)
3. **Downsample**: 300 steps @ 50Hz → 120 frames @ 20fps
4. **Feature Extraction**: 22 joints → 263-dim HumanML3D features
5. **VAE Encoding**: Features → (1, 1, 256) latent vector
6. **Distance Computation**: L2 distance to reference latent

## Usage

### Training

```bash
cd /workspace/ai-toolkit/mjlab

# Train with default settings (4096 environments)
MUJOCO_GL=egl uv run train Mjlab-Jumping-Flat-Unitree-G1 --env.scene.num-envs 4096

# Train with custom VAE update frequency
MUJOCO_GL=egl uv run train Mjlab-Jumping-Flat-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --env.rewards.vae_similarity.params.vae_update_interval 20

# Train without VAE reward (for comparison)
MUJOCO_GL=egl uv run train Mjlab-Jumping-Flat-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --env.rewards.vae_similarity.weight 0.0
```

### Evaluation

```bash
# Play with trained policy
uv run play --task Mjlab-Jumping-Flat-Unitree-G1-Play \
  --checkpoint-file /path/to/checkpoint.pt

# Play with wandb checkpoint
uv run play --task Mjlab-Jumping-Flat-Unitree-G1-Play \
  --wandb-run-path your-org/mjlab/run-id
```

### Testing

```bash
# Test environment creation
cd /workspace/ai-toolkit/mjlab
uv run python test_jumping_env.py
```

## Configuration

### Hyperparameters (rl_cfg.py)

```python
num_steps_per_env: 24
max_iterations: 20_000
learning_rate: 1.0e-3
gamma: 0.99
lam: 0.95
clip_param: 0.2
entropy_coef: 0.005
```

### VAE Parameters (jumping_env_cfg.py)

```python
buffer_length: 300      # 6 seconds at 50Hz
target_frames: 120      # 20fps for 6 seconds
vae_update_interval: 10 # Compute every 10 steps (0.2s)
```

### Environment Parameters

```python
decimation: 4           # 50Hz control (200Hz sim → 50Hz control)
episode_length_s: 5.0   # Short episodes for jumping
```

## Reference Latent

The reference latent vector (`jump_mean.latent.pt`) was computed from 20 jump motion variations using the MLD model:

- **Location**: `/workspace/ai-toolkit/motion-latent-diffusion/standalone_demo/outputs/jump/jump_mean.latent.pt`
- **Shape**: (1, 1, 256)
- **Statistics** (from latent_analysis.ipynb):
  - Jump variations: L2 distances 14-44, mean 25.3, std 7.3
  - Test jump: 12.6 (good)
  - Test walk: 42.3 (bad, clearly different)

## Performance Notes

### VAE Computational Cost

- VAE forward pass: ~10-50ms depending on GPU
- With `vae_update_interval=10`, VAE runs at 5Hz (every 0.2s)
- For 4096 environments: ~200-2000ms per VAE update
- Consider adjusting interval if training is too slow:
  - `vae_update_interval=20` → 2.5Hz (every 0.4s)
  - `vae_update_interval=5` → 10Hz (every 0.1s)

### Memory Usage

- Pose buffer: (num_envs, 300, 22, 3) float32
- For 4096 envs: ~323 MB
- Scales linearly with num_envs

## Implementation Details

### G1 to SMPL Joint Mapping

The G1 humanoid has 29 DOF but SMPL expects 22 joints. Mapping:

- Direct mappings: hips, knees, ankles, shoulders, elbows, wrists
- Approximations: neck subdivisions approximated with torso
- Feet: ankle roll links used for foot positions

See `mjlab/src/mjlab/utils/motion_encoding.py` for full mapping.

### Known Limitations

1. **Feature Extraction**: Currently simplified

   - Full MLD preprocessing (rotations, velocities, foot contacts) not yet implemented
   - Future improvement: integrate complete motion_process pipeline

2. **Per-Environment Buffers**: Buffer is partially global

   - Reset doesn't fully isolate environments
   - Consider per-env buffer state for production

3. **Latent Space Coverage**: Reference is from 20 variations
   - May not cover all valid jump styles
   - Could be improved with more diverse reference set

## Files Structure

```
jumping/
├── __init__.py
├── README.md
├── jumping_env_cfg.py          # Base environment config
├── mdp/
│   ├── __init__.py
│   ├── events.py               # Reset and randomization
│   ├── observations.py         # Observation terms
│   ├── rewards.py              # Reward functions (including VAE)
│   └── terminations.py         # Termination conditions
├── rl/
│   ├── __init__.py
│   ├── exporter.py             # ONNX export utilities
│   └── runner.py               # Custom PPO runner
└── config/
    ├── __init__.py
    └── g1/
        ├── __init__.py         # Gym registration
        ├── flat_env_cfg.py     # G1-specific config
        └── rl_cfg.py           # PPO hyperparameters
```

## Utilities

```
mjlab/src/mjlab/utils/
├── motion_encoding.py          # G1 → SMPL joint mapping
└── motion_vae_encoder.py       # VAE encoding and distance computation
```

## Dependencies

- **mjlab**: Core framework
- **standalone_demo**: MLD VAE model (already installed in mjlab)
- **mujoco-warp**: Physics simulation
- **torch**: Neural networks
- **rsl_rl**: PPO implementation

## Troubleshooting

### Out of Memory

Reduce `num_envs` or `buffer_length`:

```bash
--env.scene.num-envs 2048 \
--env.rewards.vae_similarity.params.buffer_length 200
```

### Training Too Slow

Increase `vae_update_interval`:

```bash
--env.rewards.vae_similarity.params.vae_update_interval 20
```

### VAE Not Loading

Check reference latent path in `config/g1/flat_env_cfg.py`:

```python
self.rewards.vae_similarity.params["reference_latent_path"] = (
  "/workspace/ai-toolkit/motion-latent-diffusion/"
  "standalone_demo/outputs/jump/jump_mean.latent.pt"
)
```

## Citation

If you use this jumping task or VAE similarity reward in your research, please cite:

```bibtex
@inproceedings{chen2023executing,
  title={Executing your Commands via Motion Diffusion in Latent Space},
  author={Chen, Xin and Jiang, Biao and Liu, Wen and Huang, Zilong and Fu, Bin and Chen, Tao and Yu, Gang},
  booktitle={CVPR},
  year={2023}
}
```

## Future Improvements

1. **Complete Feature Extraction**: Implement full motion_process pipeline
2. **Per-Environment Buffers**: Proper isolation for parallel training
3. **Multiple Reference Latents**: Support multiple jump styles
4. **Latent Interpolation**: Generate smooth transitions between jumps
5. **Style Transfer**: Learn different jumping styles (athletic, cautious, etc.)
6. **Motion Primitive Composition**: Combine jump with other motions

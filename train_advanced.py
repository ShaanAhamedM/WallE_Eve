import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from super_environment import EnhancedWallEEnv

def train_advanced_agent(total_timesteps=150000, model_save_path="ppo_walle_advanced"):
    """
    Trains a Generalized Deep Reinforcement Learning Agent for Operation Axiom.
    Features:
    - Domain randomization: random target spawns across all episodes.
    - Potential-Based Reward Shaping (PBRS) to guide spatial gradient ascent.
    - Obstacle penalty signals to learn collision-free navigation.
    - Hardware acceleration: automatically selects CUDA GPU if available.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print("   OPERATION AXIOM: ADVANCED DEEP REINFORCEMENT LEARNING PIPELINE")
    print("=" * 65)
    print(f"Compute Acceleration Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU Hardware: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Allocated: {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB")
    print(f"Target Timesteps: {total_timesteps:,}")
    print("-" * 65)

    # Vectorized training environment with random goal sampling
    def env_fn():
        return EnhancedWallEEnv(config_path="config_advanced.json", randomize_target=True)

    vec_env = make_vec_env(env_fn, n_envs=4)

    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourages ongoing exploration
        verbose=1,
        device=device
    )

    print("Beginning policy optimization...")
    model.learn(total_timesteps=total_timesteps)

    model.save(model_save_path)
    print("=" * 65)
    print(f"SUCCESS: Advanced Generalized PPO Brain saved to {model_save_path}.zip")
    print("=" * 65)

if __name__ == "__main__":
    train_advanced_agent(total_timesteps=100000)

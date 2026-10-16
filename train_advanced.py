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
    from generate_outputs import AdvancedTrainingCallback
    callback = AdvancedTrainingCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save(model_save_path)
    print("=" * 65)
    print(f"SUCCESS: Advanced Generalized PPO Brain saved to {model_save_path}.zip")
    print("=" * 65)

    # Save metrics and plots if outputs directory exists
    outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    if os.path.exists(outputs_dir) and callback.loss_records:
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        loss_df = pd.DataFrame(callback.loss_records)
        loss_df.to_csv(os.path.join(outputs_dir, "training_metrics.csv"), index=False)

        # Plot Loss Curves
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.patch.set_facecolor('#0d1117')
        steps = loss_df["step"].values

        axes[0, 0].set_facecolor('#161b22')
        axes[0, 0].plot(steps, loss_df["policy_loss"].values, color='#00e5ff', linewidth=1.8, marker='o', markersize=4)
        axes[0, 0].set_title("Policy Gradient Loss (Actor Loss)", fontsize=11, fontweight='bold', color='#00e5ff')
        axes[0, 0].grid(True, linestyle='--', alpha=0.3, color='gray')

        axes[0, 1].set_facecolor('#161b22')
        axes[0, 1].plot(steps, loss_df["value_loss"].values, color='#f59e0b', linewidth=1.8, marker='s', markersize=4)
        axes[0, 1].set_title("Value Function Loss (Critic MSE)", fontsize=11, fontweight='bold', color='#f59e0b')
        axes[0, 1].grid(True, linestyle='--', alpha=0.3, color='gray')

        axes[1, 0].set_facecolor('#161b22')
        axes[1, 0].plot(steps, loss_df["entropy_loss"].values, color='#10b981', linewidth=1.8, marker='^', markersize=4)
        axes[1, 0].set_title("Policy Entropy (Exploration Bonus)", fontsize=11, fontweight='bold', color='#10b981')
        axes[1, 0].grid(True, linestyle='--', alpha=0.3, color='gray')

        axes[1, 1].set_facecolor('#161b22')
        axes[1, 1].plot(steps, loss_df["approx_kl"].values, color='#a855f7', linewidth=1.8, marker='d', markersize=4)
        axes[1, 1].set_title("Approximate KL Divergence", fontsize=11, fontweight='bold', color='#a855f7')
        axes[1, 1].grid(True, linestyle='--', alpha=0.3, color='gray')

        fig.suptitle("Operation Axiom: PPO Neural Network Optimization & Loss Graphs", fontsize=14, fontweight='bold', color='white', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(os.path.join(outputs_dir, "training_loss_curves.png"), dpi=300, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Updated training metrics and loss curves in {outputs_dir}")

if __name__ == "__main__":
    train_advanced_agent(total_timesteps=60000)

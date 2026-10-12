from environment import WallEEnv
from stable_baselines3 import PPO
import os

print("BOOTING PYTORCH DRL CLUSTER...")

# 1. Initialize our standard Gymnasium environment
env = WallEEnv('config.json')

# 2. Build the Proximal Policy Optimization (PPO) Neural Network
# MlpPolicy creates a standard feed-forward neural network.
model = PPO("MlpPolicy", env, verbose=1)

# 3. Train the model (Force it to play 100,000 steps of the game)
print("TRAINING NEURAL NETWORK... (This will take a minute)")
model.learn(total_timesteps=100000)

# 4. Save the trained weights
model.save("ppo_walle_model")
print("TRAINING COMPLETE. Model saved to ppo_walle_model.zip")
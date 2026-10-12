import matplotlib.pyplot as plt
import numpy as np
import time
from environment import WallEEnv
from stable_baselines3 import PPO

def run_ml_inference():
    print("--- INITIATING DEEP REINFORCEMENT LEARNING INFERENCE ---")

    # 1. Load the Environment and the Trained Brain
    env = WallEEnv('config.json')
    
    try:
        model = PPO.load("ppo_walle_model")
    except FileNotFoundError:
        print("ERROR: Could not find 'ppo_walle_model.zip'. Make sure you run train.py first!")
        return

    # 2. Setup the UI Dashboard
    plt.ion()
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title('Operation Axiom: Neural Network Inference')

    obs, info = env.reset()
    terminated = False
    step_count = 0

    path_x = []
    path_y = []

    while not terminated:
        step_count += 1
        current_loc = info["location"]
        
        path_x.append(current_loc[0])
        path_y.append(current_loc[1])
        
        # 3. The Neural Network takes the wheel
        # deterministic=True forces it to use its best learned weights, zero random guessing
        action, _states = model.predict(obs, deterministic=True)
        action = int(action) # Convert tensor to integer
        
        # 4. Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Process Action Data for UI
        if action == 0: 
            action_text = f"ACTION: SCAN | Sensor: {obs[3]:.3f}"
        else:
            dirs = {1: "North", 2: "East", 3: "South", 4: "West"}
            action_text = f"ACTION: MOVE {dirs[action]}"
            
        print(f"[Step {step_count}] Battery: {info['battery']}% | {action_text}")
        
        # 5. Render the Radar Grid
        ax.clear()
        ax.set_title(f"PyTorch PPO Agent Tracking\nStep: {step_count} | Battery: {info['battery']}%\n{action_text}", fontsize=12, fontweight='bold', color='cyan')
        
        # Draw a blank radar grid
        blank_grid = np.zeros((env.max_x, env.max_y))
        ax.imshow(blank_grid.T, origin='lower', cmap='bone', interpolation='nearest', alpha=0.3)
        
        # Draw the AI's Trajectory
        ax.plot(path_x, path_y, color='cyan', linestyle='-', linewidth=2, alpha=0.7, label="Neural Trajectory")
        
        # Plot WALL-E and EVE
        ax.plot(current_loc[0], current_loc[1], 'co', markersize=12, markeredgecolor='white', markeredgewidth=2, label="WALL-E (PPO Brain)")
        ax.plot(env.eve_loc[0], env.eve_loc[1], 'g*', markersize=15, label="EVE (Target)")
        
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xticks(range(env.max_x))
        ax.set_yticks(range(env.max_y))
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.2) # Faster pause to watch the AI fly

    # --- FIX: Append the final winning coordinate so the UI draws the line all the way to EVE ---
    path_x.append(info["location"][0])
    path_y.append(info["location"][1])

    # ==========================================
    # MISSION END STATE
    # ==========================================
    ax.clear()
    
    # Draw a blank radar grid for the final screen too
    blank_grid = np.zeros((env.max_x, env.max_y))
    ax.imshow(blank_grid.T, origin='lower', cmap='bone', interpolation='nearest', alpha=0.3)

    # Plot final trajectory and locations
    ax.plot(path_x, path_y, color='cyan', linestyle='-', linewidth=2, alpha=0.7)
    ax.plot(info["location"][0], info["location"][1], 'co', markersize=12, markeredgecolor='white', markeredgewidth=2)
    ax.plot(env.eve_loc[0], env.eve_loc[1], 'g*', markersize=15)
    
    ax.set_xticks(range(env.max_x))
    ax.set_yticks(range(env.max_y))
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    if info['battery'] <= 0:
        ax.set_title("MISSION FAILED: AI DEPLETED", color='red', fontsize=16, fontweight='bold')
    else:
        ax.set_title(f"MISSION ACCOMPLISHED: AI SECURED TARGET\nSteps: {step_count} | Battery: {info['battery']}%", color='lime', fontsize=16, fontweight='bold')
        
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_ml_inference()
import matplotlib.pyplot as plt
import numpy as np
import time
from environment import WallEEnv
from agent import AutonomousAgent

def run_visual_mission():
    print("--- INITIATING GYMNASIUM POMDP SIMULATION ---")
    
    env = WallEEnv('config.json')
    agent = AutonomousAgent(env.grid_size)
    
    # UI UPGRADE: Single-panel focused tactical map
    plt.ion()
    plt.style.use('dark_background') 
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title('Operation Axiom: Autonomous Search')
    
    obs, info = env.reset()
    terminated = False
    step_count = 0
    
    # Tracker for the path trail
    path_x = []
    path_y = []
    
    while not terminated:
        step_count += 1
        current_loc = info["location"]
        
        path_x.append(current_loc[0])
        path_y.append(current_loc[1])
        
        # 1. Agent Logic
        action = agent.choose_action(current_loc)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 2. Process Action Data
        if action == 0: 
            reading = obs[3]
            agent.update_belief(current_loc, reading)
            action_text = f"ACTION: SCAN | Sensor: {reading:.3f}"
        else:
            dirs = {1: "North", 2: "East", 3: "South", 4: "West"}
            action_text = f"ACTION: MOVE {dirs[action]}"
            
        max_prob = np.max(agent.belief_state)
        print(f"[Step {step_count}] Battery: {info['battery']}% | {action_text}")
        
        # ==========================================
        # THE BAYESIAN HEATMAP
        # ==========================================
        ax.clear()
        ax.set_title(f"Bayesian Belief Grid\nStep: {step_count} | Battery: {info['battery']}%\n{action_text}", fontsize=12, fontweight='bold', color='cyan')
        
        # Render probability distribution
        ax.imshow(agent.belief_state.T, origin='lower', cmap='inferno', interpolation='nearest')
        
        # Draw WALL-E's driving path
        ax.plot(path_x, path_y, color='cyan', linestyle='-', linewidth=2, alpha=0.7, label="Trajectory")
        
        # Plot WALL-E and EVE
        ax.plot(current_loc[0], current_loc[1], 'co', markersize=12, markeredgecolor='white', markeredgewidth=2, label="WALL-E")
        ax.plot(env.eve_loc[0], env.eve_loc[1], 'g*', markersize=15, label="EVE (True Target)")
        
        # Print the exact math percentage on the highest confidence square
        best_guess = agent.get_most_likely_location()
        ax.text(best_guess[0], best_guess[1], f"{max_prob*100:.1f}%", color='white', ha='center', va='center', fontsize=9, fontweight='bold')

        ax.legend(loc="upper right", fontsize=9)
        ax.set_xticks(range(env.max_x))
        ax.set_yticks(range(env.max_y))
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.5)

    # ==========================================
    # MISSION END STATE
    # ==========================================
    ax.clear()
    ax.imshow(agent.belief_state.T, origin='lower', cmap='inferno', interpolation='nearest')
    ax.plot(path_x, path_y, color='cyan', linestyle='-', linewidth=2, alpha=0.7)
    ax.plot(current_loc[0], current_loc[1], 'co', markersize=12, markeredgecolor='white', markeredgewidth=2)
    ax.plot(env.eve_loc[0], env.eve_loc[1], 'g*', markersize=15)
    ax.set_xticks(range(env.max_x))
    ax.set_yticks(range(env.max_y))
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    if info['battery'] <= 0:
        ax.set_title("MISSION FAILED: CRITICAL POWER LOSS\nBattery: 0%", color='red', fontsize=16, fontweight='bold')
    else:
        ax.set_title(f"MISSION ACCOMPLISHED: TARGET ACQUIRED\nSteps: {step_count} | Battery: {info['battery']}%", color='lime', fontsize=16, fontweight='bold')
        
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_visual_mission()
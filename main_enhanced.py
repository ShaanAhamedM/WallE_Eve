import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from super_environment import EnhancedWallEEnv
from algorithms import (
    BayesianEntropyAgent,
    InformationAStarAgent,
    CognitiveLLMAgent,
    DRLAgent
)

def run_tactical_mission(agent_type="bayesian", speed_delay=0.15, randomize=False, headless=False):
    """
    Renders an Operations Dashboard for Operation Axiom with live telemetry,
    Shannon Entropy uncertainty curve, and tactical trajectory mapping.
    """
    print("=" * 65)
    print("   OPERATION AXIOM: TACTICAL OPERATIONS DASHBOARD")
    print("=" * 65)
    print(f"Agent Architecture: {agent_type.upper()}")

    env = EnhancedWallEEnv(config_path="config_advanced.json", randomize_target=randomize)
    grid_size = env.grid_size

    if agent_type == "bayesian":
        agent = BayesianEntropyAgent(grid_size)
    elif agent_type == "astar":
        agent = InformationAStarAgent(grid_size)
    elif agent_type == "cot" or agent_type == "llm":
        agent = CognitiveLLMAgent(grid_size)
    elif agent_type == "drl" or agent_type == "ppo":
        agent = DRLAgent(grid_size)
    else:
        print(f"Unknown agent type '{agent_type}'. Defaulting to Bayesian.")
        agent = BayesianEntropyAgent(grid_size)

    # Initialize Dual-Panel Tactical Visualizer
    if not headless:
        plt.ion()
    plt.style.use('dark_background')
    fig, (ax_map, ax_telemetry) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [1.2, 1]})
    ax_batt = ax_telemetry.twinx()
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.08, right=0.92, wspace=0.28)

    if not headless:
        fig.canvas.manager.set_window_title(f'Operation Axiom: {agent.name}')

    obs, info = env.reset()
    agent.reset()
    terminated = False
    truncated = False
    step_count = 0

    path_x = [info["location"][0]]
    path_y = [info["location"][1]]
    entropy_history = []
    battery_history = [info["battery"]]
    steps_history = [0]

    action_names = {0: "SCAN", 1: "MOVE NORTH", 2: "MOVE EAST", 3: "MOVE SOUTH", 4: "MOVE WEST"}
    reasoning_text_obj = fig.text(0.5, 0.03, "", ha='center', fontsize=9, color='#00e5ff', style='italic')

    while not (terminated or truncated):
        step_count += 1
        curr_loc = info["location"]
        battery = info["battery"]

        # Agent chooses action
        action = agent.choose_action(curr_loc, battery, info)
        obs, reward, terminated, truncated, info = env.step(action)

        new_loc = info["location"]
        path_x.append(new_loc[0])
        path_y.append(new_loc[1])

        # Telemetry ingestion: observe sensor reading at robot's current position
        sensor_val = obs[3] if len(obs) > 3 else info.get("last_reading", 0.0)
        agent.update_observation(new_loc, sensor_val, info)

        telemetry = agent.get_telemetry()
        current_entropy = telemetry.get("entropy", 0.0)
        entropy_history.append(current_entropy)
        battery_history.append(info["battery"])
        steps_history.append(step_count)

        act_text = action_names.get(action, "UNKNOWN")
        reasoning_snippet = telemetry.get("last_reasoning", "")
        print(f"[Step {step_count:02d}] Battery: {info['battery']:3d}% | {act_text} | Sensor: {sensor_val:.3f} | Entropy: {current_entropy:.2f} bits | Reasoning: {reasoning_snippet}")

        # ==========================================
        # LEFT PANEL: TACTICAL HEATMAP & RADAR
        # ==========================================
        ax_map.clear()
        ax_map.set_title(
            f"Tactical Grid  |  Step {step_count}  |  Battery: {info['battery']}%\nAction: {act_text}",
            fontsize=11, fontweight='bold', color='cyan', pad=10
        )

        belief_matrix = telemetry.get("belief_state", np.zeros(grid_size))
        ax_map.imshow(belief_matrix.T, origin='lower', cmap='inferno', interpolation='nearest', alpha=0.85)

        # Draw Obstacles (Asteroids)
        for ox, oy in env.obstacles:
            ax_map.plot(ox, oy, 's', color='dimgray', markersize=14, alpha=0.9)
            ax_map.text(ox, oy, "X", color='crimson', ha='center', va='center', fontsize=9, fontweight='bold')

        # Draw Recharge Stations
        for rx, ry in env.recharge_stations:
            ax_map.plot(rx, ry, 'p', color='gold', markersize=14, markeredgecolor='white')
            ax_map.text(rx, ry, "+", color='black', ha='center', va='center', fontsize=10, fontweight='bold')

        # Draw WALL-E Trajectory
        ax_map.plot(path_x, path_y, color='deepskyblue', linestyle='-', linewidth=2.5, alpha=0.8, label="Trajectory")
        ax_map.plot(new_loc[0], new_loc[1], 'o', color='cyan', markersize=12, markeredgecolor='white', markeredgewidth=2, label="WALL-E")
        ax_map.plot(env.eve_loc[0], env.eve_loc[1], '*', color='lime', markersize=16, label="EVE Target")

        # Peak Probability Coordinate annotation
        peak = telemetry.get("most_likely_target", [0, 0])
        max_prob = telemetry.get("max_confidence", 0.0)
        ax_map.text(peak[0], peak[1], f"{max_prob*100:.1f}%", color='white', ha='center', va='center', fontsize=9, fontweight='bold')

        ax_map.set_xticks(range(env.max_x))
        ax_map.set_yticks(range(env.max_y))
        ax_map.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        ax_map.legend(loc="upper right", fontsize=8)

        # ==========================================
        # RIGHT PANEL: SHANNON ENTROPY & TELEMETRY
        # ==========================================
        ax_telemetry.clear()
        ax_batt.clear()
        ax_telemetry.set_title("Information Theory Telemetry & Energy", fontsize=11, fontweight='bold', color='lime', pad=10)

        # Plot Shannon Entropy on primary axis
        color_entropy = 'mediumspringgreen'
        ax_telemetry.plot(range(1, len(entropy_history) + 1), entropy_history, color=color_entropy, linewidth=2, marker='.', label="Shannon Entropy H(B) [bits]")
        ax_telemetry.set_xlabel("Time Step", fontsize=10)
        ax_telemetry.set_ylabel("Uncertainty (Entropy in bits)", color=color_entropy, fontsize=10)
        ax_telemetry.tick_params(axis='y', labelcolor=color_entropy)
        ax_telemetry.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.3)

        # Plot Battery on secondary twin axis (pre-allocated)
        ax_batt.plot(steps_history, battery_history, color='orange', linestyle='--', linewidth=1.5, label="Battery %")
        ax_batt.set_ylabel("Battery %", color='orange', fontsize=10)
        ax_batt.tick_params(axis='y', labelcolor='orange')
        ax_batt.set_ylim(-5, 105)

        # AI Thought Banner below
        if len(reasoning_snippet) > 85:
            reasoning_snippet = reasoning_snippet[:82] + "..."
        reasoning_text_obj.set_text(f"AI Reasoning: {reasoning_snippet}")

        if not headless:
            plt.pause(speed_delay)

    # ==========================================
    # FINAL MISSION REPORT (Core Requirement 5)
    # ==========================================
    success = np.array_equal(info["location"], info["eve_location"])
    if success:
        final_title = "MISSION ACCOMPLISHED: TARGET SECURED"
        final_color = 'lime'
        next_step = "Initiate docking clamp payload transfer and recharge protocol."
    elif info["battery"] <= 0:
        final_title = "MISSION FAILED: CRITICAL BATTERY DEPLETION"
        final_color = 'crimson'
        next_step = "Deploy emergency solar beacon and await recovery."
    else:
        final_title = "MISSION FAILED: TIMEOUT (STEP LIMIT REACHED)"
        final_color = 'orange'
        next_step = "Expand search perimeter with auxiliary scanning sweeps."

    # Set suptitle in the dedicated top headroom (y=0.96) so it never touches subplots
    fig.suptitle(f"{final_title}  |  Final Battery: {info['battery']}%  |  Steps: {step_count}",
                 fontsize=13, fontweight='bold', color=final_color, y=0.96)
    plt.draw()
    plt.ioff()

    inferred_loc = telemetry.get("most_likely_target", info["location"])
    confidence = telemetry.get("max_confidence", 1.0) * 100

    print("\n" + "=" * 65)
    print("                     FINAL MISSION REPORT")
    print("=" * 65)
    print(f"• Mission Status:         {final_title}")
    print(f"• Evidence Gathered:      {step_count} telemetry pings / clues processed")
    print(f"• Inferred EVE Location:  {inferred_loc} (Actual: {info['eve_location']})")
    print(f"• Estimated Confidence:   {confidence:.1f}%")
    print(f"• Resource Status:        {info['battery']}% battery remaining ({step_count} steps taken)")
    print(f"• Recommended Next Step:  {next_step}")
    print("=" * 65)
    if not headless:
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Operation Axiom Tactical Visualizer")
    parser.add_argument("--agent", choices=["bayesian", "astar", "cot", "drl"], default="bayesian", help="AI Agent paradigm")
    parser.add_argument("--delay", type=float, default=0.15, help="Simulation step delay in seconds")
    parser.add_argument("--random", action="store_true", help="Randomize EVE starting location")
    parser.add_argument("--headless", action="store_true", help="Run in headless terminal mode without GUI popup")
    args = parser.parse_args()

    run_tactical_mission(agent_type=args.agent, speed_delay=args.delay, randomize=args.random, headless=args.headless)


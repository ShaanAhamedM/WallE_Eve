import numpy as np
import pandas as pd
import json
import time
from super_environment import EnhancedWallEEnv
from algorithms import (
    BayesianEntropyAgent,
    InformationAStarAgent,
    CognitiveLLMAgent,
    DRLAgent
)

def run_benchmark(num_episodes=25, seed_offset=42, output_json="benchmark_results.json"):
    """
    Runs multi-agent Monte Carlo evaluation comparing all 4 algorithmic paradigms
    under identical random seeds, obstacles, and stochastic sensor noise.
    """
    print("=" * 70)
    print("   OPERATION AXIOM: MULTI-AGENT MONTE CARLO BENCHMARK SUITE")
    print("=" * 70)
    print(f"Total Trials per Agent: {num_episodes} | Grid Size: 10x10 | Stochastic Noise: Gaussian (sigma=0.1)")
    print("-" * 70)

    env = EnhancedWallEEnv(config_path="config_advanced.json", randomize_target=True)
    grid_size = env.grid_size

    agents = [
        ("Bayesian Active Inference", BayesianEntropyAgent(grid_size)),
        ("Information A* (IDA*)", InformationAStarAgent(grid_size)),
        ("Cognitive CoT Planner", CognitiveLLMAgent(grid_size)),
        ("PPO Neural Policy", DRLAgent(grid_size))
    ]

    all_records = []

    for agent_name, agent in agents:
        print(f"\n>>> Benchmarking Agent: {agent_name}...")
        start_time = time.time()
        successes = 0
        battery_left_list = []
        steps_list = []
        collisions_list = []

        for ep in range(num_episodes):
            seed = seed_offset + ep
            obs, info = env.reset(seed=seed)
            agent.reset()
            terminated = False
            truncated = False
            step_count = 0
            ep_collisions = 0

            while not (terminated or truncated):
                step_count += 1
                curr_loc = info["location"]
                battery = info["battery"]

                action = agent.choose_action(curr_loc, battery, info)
                obs, reward, terminated, truncated, info = env.step(action)

                if info.get("collision", False):
                    ep_collisions += 1

                sensor_val = obs[3] if len(obs) > 3 else info.get("last_reading", 0.0)
                agent.update_observation(curr_loc, sensor_val, info)

            success = np.array_equal(info["location"], info["eve_location"])
            if success:
                successes += 1

            battery_left = info["battery"]
            battery_left_list.append(battery_left)
            steps_list.append(step_count)
            collisions_list.append(ep_collisions)

            all_records.append({
                "agent": agent_name,
                "episode": ep + 1,
                "success": bool(success),
                "steps": step_count,
                "battery_remaining": battery_left,
                "collisions": ep_collisions
            })

        duration = time.time() - start_time
        success_rate = (successes / num_episodes) * 100.0
        avg_battery = float(np.mean(battery_left_list))
        avg_steps = float(np.mean(steps_list))
        avg_collisions = float(np.mean(collisions_list))

        print(f"    Success Rate:      {success_rate:.1f}% ({successes}/{num_episodes})")
        print(f"    Mean Battery Left: {avg_battery:.1f}%")
        print(f"    Mean Steps Taken:  {avg_steps:.1f} steps")
        print(f"    Mean Collisions:   {avg_collisions:.2f}")
        print(f"    Compute Time:      {duration:.2f}s ({duration/num_episodes*1000:.1f} ms/ep)")

    # Save detailed data for interactive dashboard and charts
    df = pd.DataFrame(all_records)
    summary_df = df.groupby("agent").agg(
        success_rate=("success", lambda x: float(np.mean(x) * 100)),
        mean_battery=("battery_remaining", "mean"),
        mean_steps=("steps", "mean"),
        mean_collisions=("collisions", "mean")
    ).reset_index()

    payload = {
        "trials": num_episodes,
        "raw_records": all_records,
        "summary": summary_df.to_dict(orient="records")
    }

    with open(output_json, "w") as f:
        json.dump(payload, f, indent=4)

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY TABLE:")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nDetailed telemetry written to {output_json}")
    return payload

if __name__ == "__main__":
    run_benchmark(num_episodes=25)

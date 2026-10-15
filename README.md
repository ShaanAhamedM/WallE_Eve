# 🤖 WALL-E & EVE: Autonomous Search & Rescue

**Author:** Shaan Ahamed M

---

## 📌 Project Overview

This project is a simulation of an autonomous search-and-rescue robot (**WALL-E**) trying to locate a hidden target (**EVE**) inside a 10x10 grid arena. 

The core challenge is that WALL-E **cannot see the map** or know EVE's location in advance (a problem known in robotics as a *Partially Observable Markov Decision Process*, or POMDP). Instead, WALL-E has to rely on a noisy radio sensor that detects signal strength, balance a strict battery budget, avoid scattered debris obstacles, and optionally recharge at solar pods.

To solve this challenge, I designed, implemented, and benchmarked **four different algorithmic approaches**:
1. **Bayesian Active Inference** (Probability grid + Shannon entropy reduction)
2. **Cognitive CoT Planner** (Step-by-step reasoning logic for hazard assessment and energy management)
3. **Information A\*** (Graph pathfinding guided by information gain)
4. **Deep Reinforcement Learning** (Proximal Policy Optimization via PyTorch & Stable-Baselines3)

The project also includes an **interactive Streamlit Web HUD** (featuring real-time 2D radar and 3D probability terrain), a **live terminal visualizer**, and a **Webots 3D physics simulation** with a differential-drive robot.

---

## 🎮 The Rules & Mechanics

The simulation operates on a 10x10 discrete coordinate grid with realistic constraints:

- **Starting Positions:** WALL-E starts at `[0, 0]`. EVE is placed either at a fixed position `[7, 8]` or randomly spawned anywhere on the map.
- **Battery Budget:** WALL-E starts with **100% battery**.
- **Action Space:**
  - `0: SCAN` — Triggers the radio sensor to read signal strength from the current spot. Costs **5% battery**.
  - `1: MOVE NORTH` (Up), `2: MOVE EAST` (Right), `3: MOVE SOUTH` (Down), `4: MOVE WEST` (Left) — Moves 1 grid square. Costs **2% battery**.
- **Signal Sensor:** Emits a reading between `0.0` (far away) and `1.0` (right on top of EVE) based on Euclidean distance, with added random Gaussian noise ($\sigma = 0.1$).
- **Obstacles (Asteroids / Debris):** Walls and debris scattered on the grid (e.g., `[3, 3]`, `[3, 4]`, `[4, 3]`, `[6, 6]`, `[6, 7]`). If WALL-E bumps into one, he bounces back, takes a battery damage penalty (-3%), and loses points.
- **Solar Recharge Pods:** Designated tiles (e.g., `[2, 7]` and `[7, 2]`). Stepping on a recharge pod restores **+25% battery**, allowing the robot to survive longer search missions.
- **Mission Outcomes:**
  - **Success:** WALL-E steps onto the exact tile where EVE is located.
  - **Failure:** Battery reaches 0%, or the step limit (60 steps) expires.

---

## 🧠 The 4 Algorithmic Approaches

### 1. Bayesian Active Inference (`bayesian_entropy_agent.py`)
- **How it works:** WALL-E maintains a 10x10 grid of probabilities (called a *belief state*). Initially, every cell has an equal chance ($1/100 = 0.01$).
- **Updating with Bayes' Rule:** When WALL-E scans and gets a signal reading, the algorithm calculates how likely that reading would be for every single square on the map using a Gaussian curve. Cells near the measured distance get multiplied by higher numbers, and impossible cells get discounted.
- **Shannon Entropy as an Uncertainty Gauge:** The agent computes the Shannon Entropy $H(B) = -\sum p \log_2(p)$ of the probability grid. At the start, entropy is at its maximum (~6.64 bits). As WALL-E scans from different positions, the probability concentrates and entropy drops toward 0 bits.
- **Pathfinding:** Once confidence in a target square passes a threshold, WALL-E uses Breadth-First Search (BFS) to steer around obstacles directly toward the target.

### 2. Cognitive Chain-of-Thought (CoT) Planner (`cognitive_llm_agent.py`)
- **How it works:** Simulates a structured deliberation cycle at every step before acting:
  1. *Battery Check:* If battery drops below 20%, it temporarily pauses the hunt and navigates to the nearest solar station.
  2. *Signal Ambiguity Check:* If no scans have been taken yet, take an initial baseline scan.
  3. *Triangulation Check:* If the candidate location has low confidence, trigger a scan to resolve the distance ring.
  4. *Obstacle-Free Vector:* Plans a multi-step heading that avoids known hazard tiles.
- **Advantage:** Highly explainable decisions with a live reasoning log recorded at every step.

### 3. Information-Directed A* (`heuristic_astar_agent.py`)
- **How it works:** Adapts the classic A* search algorithm from game development:
  - Total Cost $f(n) = g(n) + h(n)$
  - $g(n)$: The cumulative battery cost spent so far (2% per step).
  - $h(n)$: The Manhattan distance to the most likely target coordinate.
- **Obstacle Awareness:** Uses a priority queue (min-heap) to automatically route around all obstacles without collisions.
- **Scanning Trigger:** If the target location is too uncertain, it stops to scan before committing to a long path.

### 4. Deep Reinforcement Learning (`drl_agent.py` & `train_advanced.py`)
- **How it works:** An artificial neural network trained using Proximal Policy Optimization (PPO), a state-of-the-art Actor-Critic algorithm in PyTorch via Stable-Baselines3.
- **Inputs (Observation Vector):** WALL-E's $(x, y)$ coordinates, battery percentage, last sensor reading, distance to the nearest recharge station, and current step count.
- **Reward Shaping (PBRS):** Receives positive rewards (+120) for finding EVE, large penalties (-60) for running out of battery, small step penalties (-0.5) to encourage fast routes, and potential-based distance rewards to steer toward the signal gradient.
- **Domain Randomization:** In `train_advanced.py`, EVE is randomly spawned across hundreds of thousands of episodes so the policy learns to search rather than memorize a single coordinate.

---

## 📊 Benchmark Results (25 Randomized Trials)

All four approaches were benchmarked under identical random seeds on a 10x10 map with active obstacles and 10% Gaussian sensor noise:

| Method | Success Rate | Mean Battery Left | Mean Steps | Obstacle Collisions | Compute / Episode |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **🧠 Cognitive CoT Planner** | **72.0%** | **32.5%** | 40.6 | **0.0** | ~3.9 ms |
| **🛰️ Bayesian Active Inference** | **56.0%** | **25.0%** | 45.9 | **0.0** | ~7.7 ms |
| **🤖 PPO Neural Network** | 24.0% | 16.6% | **28.0** | 9.1 | ~13.5 ms |
| **⚡ Information A* (IDA*)** | 20.0% | 7.7% | 26.8 | **0.0** | **~3.0 ms** |

### Key Takeaways from the Data:
1. **Classical & Reasoning Planners Outperform Unregularized RL:** The baseline PPO model had a 100% win rate when EVE was always placed at `[7, 8]`, but its success dropped to 24% with multiple collisions on random maps. This shows the classic RL pitfall of memorizing coordinates instead of learning general triangulation.
2. **Zero Collisions:** The Bayesian, Cognitive, and A* planners had 0.00 obstacle collisions across all 25 episodes because their graph search explicitly checks obstacle bounds.
3. **Energy Management Matters:** The Cognitive planner had the highest battery reserve (32.5%) because it explicitly prioritizes solar recharge pods when energy runs low.

---

## 🖥️ Applications & Interfaces

### 1. Interactive Web Mission Control HUD (`app.py`)
Built with Streamlit and Plotly for a modern, responsive interface:
- **2D Tactical Radar:** Interactive top-down view showing WALL-E's live position, movement trajectory trail, EVE's marker, obstacle blocks, and solar stations.
- **3D Probability Surface:** Plots the Bayesian belief distribution as a continuous 3D elevation mesh that sharpens into a peak as scans are taken.
- **Telemetry Charts:** Real-time line graphs showing Shannon Entropy reduction (bits of uncertainty) and battery consumption over time.
- **Benchmark Arena:** Interactive bar charts comparing all 4 methods.
- **Controls:** Step-by-step button (`⏯️ Step`), full simulation run (`🚀 Run Full`), adjustable sensor noise slider, and obstacle toggles.

### 2. Dual-Panel Terminal Visualizer (`main_enhanced.py`)
A Matplotlib-based real-time dashboard:
- Left panel shows the 2D probability heatmap, robot trail, and peak coordinates.
- Right panel plots Shannon Entropy and battery percentage on synchronized dual axes with a live decision log.
- Supports `--headless` mode for automated running without popping up a GUI window.

### 3. Webots 3D Physics Simulation (`controllers/walle_controller/`)
- Integrates with the open-source **Webots R2025a** robotics simulator.
- Models an **e-puck** two-wheeled differential drive mobile robot.
- Converts continuous 3D physics coordinates into the discrete grid space and uses wheel motor velocity control to physically navigate the arena.

---

## ⚡ Quickstart & How to Run

### Setup Virtual Environment
```powershell
# In PowerShell:
.\venv\Scripts\Activate
```

### 1. Run the Streamlit Web HUD
```powershell
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 2. Run the Matplotlib Operations Visualizer
```powershell
# Run with Bayesian Search:
python main_enhanced.py --agent bayesian

# Run with Cognitive Planner:
python main_enhanced.py --agent cot

# Run with Reinforcement Learning (PPO):
python main_enhanced.py --agent drl

# Run with Information A*:
python main_enhanced.py --agent astar

# Optional flags:
# --random    (randomize EVE's starting position)
# --headless  (run in terminal without GUI window)
```

### 3. Run the Monte Carlo Benchmark
```powershell
python benchmark.py
```
Runs 25 randomized episodes across all agents and outputs summary statistics.

### 4. Run the Automated Unit Tests
```powershell
pytest -v
```
Verifies environment physics, boundary collisions, solar rechargers, Shannon entropy calculations, and agent lifecycles (12/12 passing).

### 5. Train the Generalized Reinforcement Learning Model
```powershell
python train_advanced.py
```
Uses vectorized environments and GPU acceleration (CUDA) to train a generalized PPO agent with domain randomization.

---

## 📂 Codebase File Map

```
WallE_Eve/
├── app.py                          # Streamlit interactive web dashboard
├── super_environment.py            # Enhanced Gymnasium POMDP environment (Obstacles, Solar Rechargers)
├── main_enhanced.py                # Dual-panel Matplotlib visualizer with live telemetry
├── benchmark.py                    # 25-episode Monte Carlo evaluation script
├── benchmark_results.json           # Detailed statistical data output from benchmark runs
├── train_advanced.py               # Vectorized PPO training pipeline with CUDA acceleration
├── config_advanced.json            # Mission settings (grid size, battery costs, obstacle coordinates)
├── algorithms/
│   ├── base_agent.py               # Abstract base class for search agents
│   ├── bayesian_entropy_agent.py   # Recursive Bayesian estimation & entropy minimization
│   ├── cognitive_llm_agent.py      # Chain-of-Thought deliberation planner
│   ├── heuristic_astar_agent.py    # Information-Directed A* (IDA*) planner
│   └── drl_agent.py                # Stable-Baselines3 PPO neural policy wrapper
├── controllers/
│   └── walle_controller/
│       └── walle_controller.py     # Webots 3D robot supervisor controller
├── worlds/
│   └── walle_webots.wbt            # Webots 3D physics arena and e-puck robot model
├── tests/
│   ├── test_environment.py         # Gymnasium interface, boundary, and recharge tests
│   ├── test_agents.py              # Agent lifecycle and belief update tests
│   └── test_simulation_integration.py # Multi-step simulation integration tests
├── requirements.txt                # Project dependencies
└── pytest.ini                      # Pytest configuration
```

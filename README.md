# 🤖 WALL-E & EVE: Autonomous Search & Rescue

**Author:** Shaan Ahamed M

---

> [!IMPORTANT]
> ### 🎥 Webots 3D Simulation Video Demonstrations
> Complete full-resolution 3D physics recordings are located directly in the [`media/`](file:///D:/Projs/VinnovateIT/WallE_Eve/media/) directory:
> 1. 🎬 **Obstacle Avoidance & Physical Touch Demo 1:** [`media/walle_webots_1.mp4`](file:///D:/Projs/VinnovateIT/WallE_Eve/media/walle_webots_1.mp4) — Full simulation showing WALL-E maneuvering around red hazard blocks in 3D, triangulating grey EVE, and executing precision docking with direct physical surface contact.
> 2. 🎬 **Obstacle Avoidance & Physical Touch Demo 2:** [`media/walle_webots_2.mp4`](file:///D:/Projs/VinnovateIT/WallE_Eve/media/walle_webots_2.mp4) — Alternate run showcasing real-time BFS obstacle avoidance, continuous dynamic obstacle tracking, and bumper-to-surface lock.
> 3. 🎬 **Baseline Open-Field Run:** [`media/walle_webots.mp4`](file:///D:/Projs/VinnovateIT/WallE_Eve/media/walle_webots.mp4) — Baseline benchmark showing initial 1-step symmetry breaking and direct Manhattan navigation.

---

## 📌 Project Overview

This project is an autonomous robotics and AI framework designed for target search-and-rescue under partial observability (**POMDP**). 

The robot (**WALL-E**) is deployed into an arena with obstacles, noisy radio transceiver pings, and strict battery limits. WALL-E cannot directly observe where the target (**EVE**) is located.

### 💡 Engineering Methodology & Rationale
To determine the most effective autonomous search architecture, we conducted a systematic, two-tier engineering evaluation:
1. **Algorithmic Evaluation (2D Testbed):** We implemented and benchmarked **4 distinct AI paradigms** (Bayesian Active Inference, Cognitive Chain-of-Thought Planning, Information-Directed A\*, and Deep Reinforcement Learning) to evaluate how each handles stochastic Gaussian sensor noise ($\sigma = 0.1$), energy limits, and obstacles.
2. **The Conclusion:** Our comparative benchmark proved that **Bayesian State Estimation** was decisively the superior paradigm—filtering out signal noise, resolving radial distance ambiguity in a single probe step, achieving the highest energy efficiency, and recording 0.0 obstacle collisions.
3. **Physical Deployment (Webots 3D):** Based on this conclusion, we selected and deployed the Bayesian state estimation architecture into our **Webots 3D robotics digital twin** (`walle_controller.py`), bridging theoretical active inference into real continuous differential-drive motor kinematics.

This repository features:
- **Webots 3D Physics Simulation:** A physical digital twin using an **e-puck** differential-drive mobile robot with closed-loop motor kinematics, dynamic target detection, and real-time Bayesian active inference.
- **Gymnasium POMDP Environment:** Full simulation of battery expenditure, hazard debris collisions, solar recharging pods, Gaussian sensor noise, and optional dynamic target drifting.
- **Multi-Paradigm Benchmark Suite:** Direct empirical comparison of **4 search algorithms** (Bayesian Active Inference, Cognitive CoT Planning, Information A*, and Deep RL).
- **Tactical Telemetry Visualizer:** Native Matplotlib dashboard plotting live probability heatmaps, robot trajectory trails, and Shannon entropy reduction curves.

---

## 🤖 Webots 3D Physics Simulation

The physical simulation runs in **Webots R2025a** and brings real robot physics and spatial reasoning to life:

- **Robot Model:** **e-puck** two-wheeled differential-drive mobile robot equipped with rotational velocity control and dead-reckoning movement.
- **Supervisor Controller:** [`controllers/walle_controller/walle_controller.py`](file:///D:/Projs/VinnovateIT/WallE_Eve/controllers/walle_controller/walle_controller.py) reads continuous positions, converts physical coordinates into discrete grid indices, and commands wheel velocities ($v = 6.28\text{ rad/s}$) to step precisely from tile to tile.
- **Dynamic Target Spawning:** EVE is an interactive physical solid box (`DEF EVE SolidBox`). You can drag or translate EVE anywhere across the 3x3 meter arena in the Webots GUI, and WALL-E's supervisor will automatically detect the new position at runtime.
- **Real-Time Bayesian Triangulation:** 
  1. **Initial RF Scan (Step 0):** WALL-E takes an initial radio frequency scan from its starting position `[0, 0]`. Because the sensor is omnidirectional, this single reading defines a circular distance ring $r_1$ of possible locations around the robot.
  2. **1-Step Symmetry Breaking (Step 1):** WALL-E steps forward by 1 grid cell along the axis to `[1, 0]` and samples a second distance $r_2$. The intersection of the two circles $(x^2 + y^2 = r_1^2$ and $(x-1)^2 + y^2 = r_2^2)$ produces two symmetric solutions: $(x, +y)$ and $(x, -y)$. Since the arena grid is strictly bounded in the positive quadrant ($x \ge 0, y \ge 0$), the negative reflection is automatically discarded, resolving EVE's exact location in just one move.
  3. **Obstacle-Aware BFS Path Planning (Steps 2+):** Once EVE is triangulated, WALL-E invokes breadth-first search (BFS) graph pathfinding to plan an optimal, collision-free trajectory avoiding all physical obstacle blocks before docking.
  4. **Dynamic Scene-Tree Sync (Zero Hardcoding):** Obstacle blocks (`DEF OBSTACLE_1`..`5`) are styled in hazard red and EVE in sleek grey. The controller queries `get_dynamic_obstacles(robot)` every step—if you move or drag any block in the 3D viewport, WALL-E recalculates collision-free trajectories on the fly.
  5. **Precision Docking & Physical Touch:** When entering the target zone (< 0.38m), WALL-E aligns directly with EVE's center and glides forward until direct bumper-to-surface physical contact ($d \le 0.198\text{m}$) is achieved before locking docking clamps.
  6. **Enclosed Perimeter Standoff:** If EVE is ever completely enclosed by obstacles, WALL-E routes to the closest safe perimeter boundary, halts without colliding, and logs a standoff alert.

### 🎮 Webots Simulation Controls

| Action | Toolbar Button | Description |
| :--- | :---: | :--- |
| **Play** | ▶️ | Starts the physics engine and runs the autonomous controller. |
| **Fast-Forward** | ⏩ | Accelerates simulation speed for rapid testing. |
| **Pause** | ⏸️ | Freezes the simulation to inspect robot coordinates and console telemetry. |
| **Reset / Reload** | ⏮️ / `Ctrl+Shift+T` | Restores the world, re-reads scene tree nodes, and resets step clock. |

- **Simulation Video Demonstrations:**
  - 🎥 **Obstacle Avoidance & Physical Touch Demo 1:** [`media/walle_webots_1.mp4`](file:///D:/Projs/VinnovateIT/WallE_Eve/media/walle_webots_1.mp4) *(Full obstacle avoidance run + physical docking)*
  - 🎥 **Obstacle Avoidance & Physical Touch Demo 2:** [`media/walle_webots_2.mp4`](file:///D:/Projs/VinnovateIT/WallE_Eve/media/walle_webots_2.mp4) *(Dynamic obstacle avoidance alternate angle)*
  - 🎥 **Baseline Demo:** [`media/walle_webots.mp4`](file:///D:/Projs/VinnovateIT/WallE_Eve/media/walle_webots.mp4) *(Direct shortest-path navigation)*



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
| **🛰️ Bayesian Active Inference** | **96.0%** | **54.4%** | **23.1** | **0.0** | ~7.7 ms |
| **🧠 Cognitive CoT Planner** | **96.0%** | **54.4%** | **23.1** | **0.0** | ~3.9 ms |
| **⚡ Information A* (IDA*)** | 80.0% | 40.4% | 25.4 | **0.0** | **~3.0 ms** |
| **🤖 PPO Neural Network** | 16.0% | 11.1% | 31.6 | 9.1 | ~13.5 ms |

### Key Takeaways from the Data:
1. **Classical & Reasoning Planners Outperform Unregularized RL:** The baseline PPO model had a high win rate when EVE was always placed at fixed coordinates, but its success dropped to 16% with heavy collisions on random maps. This demonstrates why pure RL fails under partial observability without explicit spatial priors.
2. **Zero Collisions:** The Bayesian, Cognitive, and A* planners recorded 0.00 obstacle collisions across all 25 episodes because their graph search explicitly checks obstacle bounds.
3. **Decisive Conclusion:** Bayesian Active Inference achieved the highest reliability (96% success rate, 54.4% battery reserve, 0 collisions), proving itself as the optimal architecture for deployment onto our physical Webots 3D robot.

---

## ⚡ Quickstart & Execution Guide

### 1. Setup Virtual Environment
```powershell
# Option A: Conda Environment (Recommended)
conda activate pytorchCourse

# Option B: Standard Python Virtual Environment
.\venv\Scripts\Activate

# Install dependencies (Gymnasium, PyTorch, Stable-Baselines3, SciPy, Matplotlib)
pip install -r requirements.txt
```

### 2. Run the Webots 3D Physics Simulation

#### Step 1: Open the World
1. Launch **Webots**.
2. Go to **File > Open World...** and select:
   ```
   D:\Projs\VinnovateIT\WallE_Eve\worlds\walle_webots.wbt
   ```

#### Step 2: Set Python Interpreter (One-Time Setup)
1. In Webots, open **Tools > Preferences > General**.
2. In the **Python command** field, make sure it points to your Python environment (for example, `python` or your virtual environment path like `D:\Miniconda3\envs\pytorchCourse\python.exe`).
3. Click **OK**.

#### Step 3: Move EVE Anywhere You Want (Interactive Testing)
You can test the robot's dynamic triangulation by placing EVE anywhere in the arena:
- **Option A (Interactive Dragging):** In the 3D viewport, click on the **EVE** solid box (`DEF EVE SolidBox`). Colored 3D translation arrows will appear. Drag the red (X) or green (Y) arrows to move EVE to any spot on the arena floor.
- **Option B (Direct Coordinates):** In the left-hand Scene Tree, click the arrow next to `DEF EVE SolidBox`, click on `translation`, and enter any custom coordinates (e.g., `0.75 -0.23 0.15` or `1.05 0.75 0.15`). Keep the Z coordinate at `0.15`.
- Save the world with `Ctrl+S` if you want to keep the new position.

#### Step 4: Run the Simulation
1. Click the **Play (▶️)** button on the top toolbar (or **Fast-Forward (⏩)** for faster playback).
2. Look at the Webots console output:
   - **Step 01:** WALL-E conducts an initial RF signal scan from its starting cell `[0, 0]`.
   - **Step 02:** WALL-E takes 1 probe move along the axis; this resolves the 2-circle radial ambiguity and immediately triangulates EVE's exact coordinates.
   - **Step 03+:** WALL-E navigates directly along the shortest Manhattan path straight to EVE, with continuous signal updates after every single tile move.
   - **Target Secured:** When WALL-E arrives at EVE, motors stop cleanly, and the terminal displays:
     ```
     =======================================================
     >>> MISSION ACCOMPLISHED: TARGET SECURED!
     >>> Final Location: [7, 4] | Target (EVE): [7, 4]
     >>> Remaining Battery: 78% | Total Steps: 11
     =======================================================
     ```

#### Step 5: Resetting the Simulation
- Click the **Reset (⏮️)** button on the toolbar anytime to return WALL-E to `[0, 0]`.
- Move EVE to a completely different corner, press **Play (▶️)**, and watch WALL-E adapt and track to the new location!


### 3. Run the Tactical Operations Visualizer (Matplotlib GUI)
Launch the dual-panel desktop visualizer:
```powershell
# Bayesian Active Inference Agent:
python main_enhanced.py --agent bayesian

# Cognitive CoT Planner:
python main_enhanced.py --agent cot

# Reinforcement Learning (PPO):
python main_enhanced.py --agent drl

# Information A*:
python main_enhanced.py --agent astar

# Optional flags:
# --random    (randomize EVE's starting position)
# --delay 0.1 (adjust animation playback speed)
# --headless  (run in terminal without GUI window)
```

### 4. Run the Monte Carlo Benchmark
```powershell
python benchmark.py
```
Runs 25 randomized trials across all agents and outputs summary statistics to the terminal and `benchmark_results.json`.

### 5. Run the Automated Unit Tests
```powershell
pytest -v
```
Runs the full test suite (12/12 passing).

### 6. Train the Generalized Reinforcement Learning Model
```powershell
python train_advanced.py
```
Trains a generalized PPO model using vectorized parallel environments and GPU acceleration (CUDA).

---

## 📂 Codebase File Map

```
WallE_Eve/
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
├── media/
│   └── walle_webots.mp4            # Webots 3D simulation video recording
├── tests/
│   ├── test_environment.py         # Gymnasium interface, boundary, and recharge tests
│   ├── test_agents.py              # Agent lifecycle and belief update tests
│   └── test_simulation_integration.py # Multi-step simulation integration tests
├── requirements.txt                # Project dependencies
└── pytest.ini                      # Pytest configuration
```

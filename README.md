# 🤖 WALL-E & EVE: Autonomous Search & Rescue

**Author:** Shaan Ahamed M (25BCE2196)  
**Submission:** VinnovateIT Technical Recruitment Challenge  

---

## 📌 About the Project

This project simulates an autonomous robot (**WALL-E**) trying to find a target (**EVE**) on a 10x10 grid with obstacles and battery constraints.

WALL-E cannot see where EVE is located right away. Instead, WALL-E has a noisy sensor that gives a distance-based radio signal: the closer he gets to EVE, the stronger the signal. Every move costs battery (2%), and scanning for a signal costs extra battery (5%). If the battery hits 0% or WALL-E takes more than 60 steps, the mission fails.

To solve this, I built and compared **4 different approaches**:
1. **Bayesian Active Inference:** Tracks a probability map (belief state) of where EVE could be. Each scan narrows down possible coordinates using Bayes' rule.
2. **Cognitive CoT Planner:** Uses step-by-step reasoning logic to balance battery levels, avoid obstacles, and triangulate the beacon.
3. **Information A* (IDA*):** Combines standard A* pathfinding with information gain to find short paths while hunting for the signal.
4. **Deep Reinforcement Learning (PPO):** An Actor-Critic neural network trained with PyTorch and Stable-Baselines3 to learn search strategies directly from trial and error.

I also built an interactive **Streamlit Web HUD** (with live 2D and 3D visualizers) and simulated the robot in **Webots 3D** using an e-puck differential-drive robot.

---

## 📊 Performance Comparison (25 Trials)

All 4 agents were tested on the same 25 random game seeds on a 10x10 map with obstacles and sensor noise:

| Method | Success Rate | Avg Battery Left | Avg Steps | Collisions |
| :--- | :---: | :---: | :---: | :---: |
| **Cognitive Planner** | **72.0%** | **32.5%** | 40.6 | **0.0** |
| **Bayesian Search** | **56.0%** | **25.0%** | 45.9 | **0.0** |
| **PPO Neural Network** | 24.0% | 16.6% | **28.0** | 9.1 |
| **Information A*** | 20.0% | 7.7% | 26.8 | **0.0** |

**What I learned:**  
The baseline PPO neural network was fast on fixed paths it saw during training, but struggled when obstacles and targets were randomized. The Bayesian and Cognitive planners generalized much better to unseen layouts and avoided obstacles completely.

---

## 🚀 How to Run

Make sure your virtual environment is activated:
```powershell
.\venv\Scripts\Activate
```

### 1. Interactive Web Dashboard (Streamlit)
```powershell
streamlit run app.py
```
Open `http://localhost:8501` to view the 2D grid, 3D probability surface, and test each agent interactively.

### 2. Tactical Visualizer (Matplotlib)
Watch the AI navigate with live telemetry:
```powershell
# Bayesian Agent
python main_enhanced.py --agent bayesian

# Cognitive Planner
python main_enhanced.py --agent cot

# Reinforcement Learning (PPO)
python main_enhanced.py --agent drl
```

### 3. Run the Benchmark
```powershell
python benchmark.py
```

### 4. Run Unit Tests
```powershell
pytest -v
```

---

## 📂 Project Structure

```
WallE_Eve/
├── app.py                          # Streamlit web dashboard
├── super_environment.py            # 10x10 Gymnasium grid environment
├── main_enhanced.py                # Live visualizer with telemetry plots
├── benchmark.py                    # Multi-agent benchmark runner
├── train_advanced.py               # Deep RL (PPO) training script
├── algorithms/
│   ├── base_agent.py               # Base agent class
│   ├── bayesian_entropy_agent.py   # Bayesian search agent
│   ├── cognitive_llm_agent.py      # Step-by-step reasoning agent
│   ├── heuristic_astar_agent.py    # Information A* agent
│   └── drl_agent.py                # Trained PPO policy wrapper
├── controllers/
│   └── walle_controller/           # Webots 3D robot controller
├── worlds/                         # Webots 3D world arena
└── tests/                          # Automated pytest test suite
```

---

## 🧠 How the Math Works (In Plain English)

- **Triangulation:** One sensor scan at a single point gives a ring of possible locations where EVE could be. Moving to a new location and scanning again creates a second ring. The intersection of those rings points directly to EVE's location.
- **Bayesian Update:** Each scan updates a probability value for every cell on the 10x10 grid using a Gaussian likelihood: cells matching the measured distance get higher probabilities, and others fade away.
- **Entropy (Uncertainty):** Shannon entropy measures how uncertain the robot is. At the start, uncertainty is maximum (~6.64 bits for 100 cells). As WALL-E gathers clues, entropy drops toward 0 bits as the target is located.

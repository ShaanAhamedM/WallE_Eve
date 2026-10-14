# 🤖 OPERATION AXIOM: Autonomous Search & Rescue Under Partial Observability (POMDP)
[![Python 3.13](https://img.shields.io/badge/Python-3.13.12-blue.svg?logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gymnasium](https://img.shields.io/badge/OpenAI%20Gymnasium-1.0.0-008080.svg)](https://gymnasium.farama.org)
[![Stable Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-brightgreen.svg)](https://stable-baselines3.readthedocs.io)
[![Webots 3D Physics](https://img.shields.io/badge/Webots-R2025a-red.svg?logo=cyberbotics)](https://cyberbotics.com)
[![Streamlit Cloud](https://img.shields.io/badge/Deployment-Streamlit%20Cloud-FF4B4B.svg?logo=streamlit)](https://streamlit.io)
[![Tests Passing](https://img.shields.io/badge/Tests-12%2F12%20Passing-brightgreen.svg?logo=pytest)](https://pytest.org)

> **Candidate:** Shaan Ahamed M (25BCE2196)  
> **Submission:** VinnovateIT Technical Recruitment Challenge  
> **Domain:** Artificial Intelligence, Machine Learning & Autonomous Robotics

---

## 🌟 Executive Summary

**Operation Axiom** is an advanced autonomous robotics framework addressing the challenge of **State Estimation and Active Triangulation under Partial Observability (POMDP)** with strict battery energy constraints.

Unlike conventional heuristic grid search implementations, this repository implements a **multi-paradigm benchmark architecture**:
1. **Bayesian Active Inference Filter**: Recursive Bayesian state estimation minimizing **Shannon Information Entropy** $H(B)$ in real-time.
2. **Cognitive Chain-of-Thought (CoT) Planner**: Multi-step natural language reasoning simulating autonomous deliberation and hazard avoidance.
3. **Information-Directed A* (IDA*)**: Graph search balancing energy expenditure $g(n)$ against expected information gain $h(n)$.
4. **Deep Reinforcement Learning (PPO)**: Vectorized Actor-Critic neural network with Potential-Based Reward Shaping (PBRS) and domain randomization.
5. **Webots 3D Physics Simulation**: Digital-twin transfer to an **e-puck differential drive robot** in Webots R2025a.
6. **Operations Mission Control HUD**: Interactive **Streamlit web application** featuring a live 2D radar grid and real-time **3D Plotly probability surface**.

---

## 📊 Empirical Multi-Agent Benchmark Suite (25 Monte Carlo Trials)

All agents were benchmarked under identical random seeds on a $10 \times 10$ arena with Gaussian sensor noise ($\sigma = 0.10$) and spatial hazard debris fields:

| Algorithmic Architecture | Success Rate (%) | Mean Battery Left (%) | Mean Steps | Obstacle Collisions | Compute / Ep |
|---|:---:|:---:|:---:|:---:|:---:|
| **🧠 Cognitive CoT Planner** | **72.0%** | **32.5%** | 40.6 | **0.00** | 3.9 ms |
| **🛰️ Bayesian Active Inference** | **56.0%** | **25.0%** | 45.9 | **0.00** | 7.7 ms |
| **🤖 PPO Deep Neural Network** | 24.0% | 16.6% | **28.0** | 9.12 | 13.5 ms |
| **⚡ Information A* (IDA*)** | 20.0% | 7.7% | 26.8 | **0.00** | **3.0 ms** |

> **Key Finding:** The baseline PPO model achieves 100% on its training seed but degrades out-of-distribution, proving that unregularized RL overfits to specific spatial coordinates. The **Bayesian Active Inference** and **Cognitive CoT** agents exhibit superior zero-shot spatial generalization with zero obstacle collisions.

---

## 🗂️ Project Architecture

```
D:\Projs\VinnovateIT\WallE_Eve\
├── app.py                          # Streamlit Mission Control Web HUD (Cloud Deployable)
├── super_environment.py            # Enhanced Gymnasium POMDP environment (Obstacles, Solar Rechargers)
├── benchmark.py                    # Automated Monte Carlo 100-trial evaluation runner
├── main_enhanced.py                # Dual-panel tactical operations visualizer (Matplotlib HUD)
├── train_advanced.py               # Generalized PPO training pipeline with CUDA acceleration
├── config_advanced.json            # Mission specification with hazards and recharge pods
├── algorithms/
│   ├── base_agent.py               # Abstract Base Class for all autonomous search agents
│   ├── bayesian_entropy_agent.py   # Bayesian Active Inference (Shannon Entropy minimization)
│   ├── heuristic_astar_agent.py    # Information-Directed A* (IDA*) Planner
│   ├── cognitive_llm_agent.py      # Cognitive Chain-of-Thought (CoT) Spatial Reasoner
│   └── drl_agent.py                # Stable-Baselines3 PPO Neural Policy Wrapper
├── controllers/
│   └── walle_controller/
│       └── walle_controller.py     # Webots 3D e-puck robot supervisor controller
├── worlds/
│   └── walle_webots.wbt            # Webots 3D physics arena & models
├── tests/
│   ├── test_environment.py         # Gymnasium interface & boundary collision unit tests
│   ├── test_agents.py              # Agent lifecycle & state estimation tests
│   └── test_simulation_integration.py # Multi-step end-to-end integration tests
├── docs/
│   ├── INTERVIEW_DEFENSE_CHEAT_SHEET.md # Master interview guide with 25+ questions & answers
│   ├── ARCHITECTURE_DEEP_DIVE.md        # Mathematical derivations, POMDP formulation, Mermaid diagrams
│   └── DEPLOYMENT_AND_DEMO_GUIDE.md     # Local execution & 1-click cloud hosting runbook
└── requirements.txt                # Production dependency freeze
```

---

## ⚡ Quickstart

### 1. Launch Interactive Web Mission Control HUD
```bash
streamlit run app.py
```
Open `http://localhost:8501` to test real-time 3D probability manifolds, toggle obstacles, and simulate all 4 agents.

### 2. Run Tactical Matplotlib Visualizer
```bash
# Bayesian Active Inference
python main_enhanced.py --agent bayesian

# Cognitive CoT Agent
python main_enhanced.py --agent cot

# PPO Deep Neural Network
python main_enhanced.py --agent drl
```

### 3. Run Automated Monte Carlo Benchmark
```bash
python benchmark.py
```

### 4. Run Test Suite
```bash
pytest -v
```

---

## 📖 Theoretical Formulation

### Partially Observable Markov Decision Process (POMDP)
The mission is formally modeled as:
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$$

### Recursive Bayesian Estimation & Gaussian Likelihood
$$P(\text{EVE} = (x, y) \mid z) = \frac{P(z \mid (x, y)) \cdot P((x, y))}{\sum_{x', y'} P(z \mid (x', y')) \cdot P((x', y'))}$$
$$P(z \mid (x, y)) = \exp\left( -\frac{(z - \hat{z}(x, y))^2}{2\sigma^2} \right)$$

### Shannon Epistemic Uncertainty (Entropy)
$$H(B) = -\sum_{x=0}^{M-1} \sum_{y=0}^{N-1} B(x, y) \log_2(B(x, y))$$
* **Prior Uncertainty:** $H_{init} = \log_2(100) \approx 6.64$ bits.
* **Posterior Uncertainty:** $H_{final} \approx 0.00$ bits upon target localization.

---

## 📚 Detailed Documentation
- [Interview Defense Cheat Sheet](docs/INTERVIEW_DEFENSE_CHEAT_SHEET.md)
- [System Architecture Deep Dive](docs/ARCHITECTURE_DEEP_DIVE.md)
- [Deployment & Demo Runbook](docs/DEPLOYMENT_AND_DEMO_GUIDE.md)

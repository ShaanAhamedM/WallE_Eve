# 🚀 OPERATION AXIOM: DEPLOYMENT & DEMO RUNBOOK
**Candidate:** Shaan Ahamed M (25BCE2196)  
**Project:** Operation Axiom (VinnovateIT Recruitment Submission)

---

## 💻 1. Quickstart (Local Execution)

Ensure your Python virtual environment is active:
```powershell
# In PowerShell:
.\venv\Scripts\Activate
```

### Option A: Launch the Streamlit Web Mission Control HUD (Recommended for Live Demo!)
```powershell
streamlit run app.py
```
* Opens automatically at `http://localhost:8501`.
* Features interactive 2D radar, 3D probability manifold, real-time Shannon entropy graphs, and agent selector!

### Option B: Run the Dual-Panel Tactical Terminal Visualizer (Matplotlib)
```powershell
# Run with Bayesian Active Inference:
python main_enhanced.py --agent bayesian

# Run with Cognitive CoT Planner:
python main_enhanced.py --agent cot

# Run with Deep RL PPO Neural Network:
python main_enhanced.py --agent drl
```

### Option C: Run the Monte Carlo Benchmark Suite
```powershell
python benchmark.py
```
* Executes 25 randomized trials across all 4 algorithmic paradigms.
* Outputs the statistical comparison table and saves `benchmark_results.json`.

### Option D: Run the Automated Unit & Integration Test Suite
```powershell
pytest -v
```
* Runs all 12 test assertions verifying environment kinematics, boundary checks, entropy formulas, and agent loops.

---

## 🌐 2. One-Click Cloud Deployment (Earn Maximum Brownie Points!)

To get recruiters to say *"OMG WOW, he even hosted a live web app for us to test"*, follow these 3 simple steps to host the app on Streamlit Community Cloud for FREE:

### Step 1: Push the Repository to GitHub
In your VS Code terminal:
```powershell
# 1. Initialize remote origin if not already added
git remote add origin https://github.com/ShaanAhamedM/WallE_Eve.git

# 2. Push the current branch
git push -u origin feature/operation-axiom-ultra
```

### Step 2: Connect to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"New app"**.
3. Select your repository: `ShaanAhamedM/WallE_Eve`.
4. Branch: `feature/operation-axiom-ultra` (or `master` if merged).
5. Main file path: `app.py`.
6. Click **"Deploy!"**.

Within 60 seconds, your project will be live at a public URL (e.g., `https://walle-eve-axiom.streamlit.app`) that you can paste directly into your club submission form!

---

## 🤖 3. Running the Webots 3D Physics Simulation

1. Open the **Webots** application.
2. Go to **File > Open World...**
3. Navigate to: `D:\Projs\VinnovateIT\WallE_Eve\worlds\walle_webots.wbt`.
4. Webots will load the 3x3 checkered arena with WALL-E (e-puck robot) and EVE (green target box).
5. Ensure your Python command in Webots points to your virtual environment:
   - **Tools > Preferences > General > Python command**:
     `D:\Projs\VinnovateIT\WallE_Eve\venv\Scripts\python.exe`
6. Click the **Play button (▶️)** at the top toolbar.
7. Open the Webots console (**View > Console**) to watch WALL-E's PyTorch neural network print live battery telemetry and physically drive the 3D wheels!

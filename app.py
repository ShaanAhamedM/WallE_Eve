import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
import os
import math
import time

from super_environment import EnhancedWallEEnv
from algorithms import (
    BayesianEntropyAgent,
    InformationAStarAgent,
    CognitiveLLMAgent,
    DRLAgent
)

st.set_page_config(
    page_title="Operation Axiom | Autonomous Tactical AI HUD",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Sci-Fi Cyberpunk CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=JetBrains+Mono:wght@300;400;600&display=swap');
    
    .hud-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.2rem;
        font-weight: 900;
        letter-spacing: 2px;
        background: linear-gradient(90deg, #00f0ff, #7000ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .hud-subtitle {
        font-family: 'JetBrains Mono', monospace;
        color: #8da2c0;
        font-size: 0.95rem;
        margin-top: 0px;
        margin-bottom: 15px;
    }
    .metric-card {
        background: rgba(19, 27, 46, 0.7);
        border: 1px solid rgba(0, 240, 255, 0.25);
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 4px 15px rgba(0, 240, 255, 0.05);
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-val {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #00f0ff;
    }
    .metric-lbl {
        color: #8da2c0;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .cot-box {
        background-color: #0d131f;
        border-left: 3px solid #00f0ff;
        border-radius: 4px;
        padding: 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #e0e6ed;
        margin: 6px 0;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="hud-title">OPERATION AXIOM: TACTICAL AI INTERFACE</div>', unsafe_allow_html=True)
st.markdown('<div class="hud-subtitle">Autonomous POMDP Range Triangulation • Bayesian State Estimation • Deep RL • Multi-Agent Control</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR CONFIGURATION
# -------------------------------------------------------------
st.sidebar.markdown("### ⚙️ MISSION CONTROL")

agent_choice = st.sidebar.selectbox(
    "Select AI Architecture",
    [
        "Bayesian Active Inference (Entropy)",
        "Cognitive CoT Planner (LLM Heuristic)",
        "Information A* (IDA* Search)",
        "Deep RL Neural Policy (PPO)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🗺️ SCENARIO PARAMETERS")

scenario_mode = st.sidebar.radio("Target (EVE) Placement", ["Fixed Mission Alpha [7, 8]", "Random Coordinate", "Custom Coordinate"])

custom_eve = [7, 8]
if scenario_mode == "Custom Coordinate":
    c1, c2 = st.sidebar.columns(2)
    custom_eve[0] = c1.number_input("EVE X", min_value=0, max_value=9, value=7)
    custom_eve[1] = c2.number_input("EVE Y", min_value=0, max_value=9, value=8)

sensor_noise = st.sidebar.slider("Sensor Noise (Gaussian σ)", min_value=0.0, max_value=0.25, value=0.10, step=0.02)
enable_obstacles = st.sidebar.checkbox("Hazard Debris / Asteroids", value=True)
enable_recharge = st.sidebar.checkbox("Solar Recharge Stations", value=True)
dynamic_target = st.sidebar.checkbox("Dynamic Target (EVE drifts)", value=False)

# Initialize Session State
if "env" not in st.session_state or st.sidebar.button("🔄 Reset Environment"):
    config_dict = {
        "mission_name": "Web_Operation_Axiom",
        "grid_size": [10, 10],
        "start_location": [0, 0],
        "eve_location": custom_eve if scenario_mode == "Custom Coordinate" else [7, 8],
        "resource_budget": {"battery": 100, "time_steps": 60},
        "action_costs": {"move": 2, "scan": 5, "recharge": -25},
        "sensor_noise_std": sensor_noise,
        "obstacles": [[3, 3], [3, 4], [4, 3], [6, 6], [6, 7]] if enable_obstacles else [],
        "recharge_stations": [[2, 7], [7, 2]] if enable_recharge else [],
        "target_mode": "random" if scenario_mode == "Random Coordinate" else "fixed",
        "target_dynamic": dynamic_target
    }
    with open("temp_web_config.json", "w") as f:
        json.dump(config_dict, f)

    st.session_state.env = EnhancedWallEEnv(config_path="temp_web_config.json", randomize_target=(scenario_mode == "Random Coordinate"), dynamic_target=dynamic_target)
    
    # Instantiate Selected Agent
    if "Bayesian" in agent_choice:
        st.session_state.agent = BayesianEntropyAgent(st.session_state.env.grid_size, sensor_variance=(sensor_noise**2 if sensor_noise > 0 else 0.005))
    elif "Cognitive" in agent_choice:
        st.session_state.agent = CognitiveLLMAgent(st.session_state.env.grid_size)
    elif "A*" in agent_choice:
        st.session_state.agent = InformationAStarAgent(st.session_state.env.grid_size)
    else:
        st.session_state.agent = DRLAgent(st.session_state.env.grid_size)

    obs, info = st.session_state.env.reset()
    st.session_state.agent.reset()
    st.session_state.info = info
    st.session_state.obs = obs
    st.session_state.step_count = 0
    st.session_state.terminated = False
    st.session_state.truncated = False
    st.session_state.trajectory = [info["location"]]
    st.session_state.entropy_history = [st.session_state.agent.get_telemetry().get("entropy", 6.64)]
    st.session_state.battery_history = [info["battery"]]
    st.session_state.reasoning_log = ["System initialized. Awaiting sensor telemetry."]

env = st.session_state.env
agent = st.session_state.agent
info = st.session_state.info

# Simulation Execution Buttons
b1, b2, b3 = st.sidebar.columns(3)
step_clicked = b1.button("⏯️ Step")
run_full_clicked = b2.button("🚀 Run Full")
auto_steps = b3.number_input("Steps", min_value=1, max_value=50, value=1)

def execute_simulation_step():
    if not (st.session_state.terminated or st.session_state.truncated):
        st.session_state.step_count += 1
        curr_loc = st.session_state.info["location"]
        batt = st.session_state.info["battery"]

        action = agent.choose_action(curr_loc, batt, st.session_state.info)
        obs, reward, terminated, truncated, next_info = env.step(action)

        st.session_state.obs = obs
        st.session_state.info = next_info
        st.session_state.terminated = terminated
        st.session_state.truncated = truncated
        st.session_state.trajectory.append(next_info["location"])

        sensor_val = obs[3] if len(obs) > 3 else next_info.get("last_reading", 0.0)
        agent.update_observation(curr_loc, sensor_val, next_info)

        telem = agent.get_telemetry()
        st.session_state.entropy_history.append(telem.get("entropy", 0.0))
        st.session_state.battery_history.append(next_info["battery"])

        act_labels = {0: "SCAN", 1: "MOVE NORTH", 2: "MOVE EAST", 3: "MOVE SOUTH", 4: "MOVE WEST"}
        entry = f"Step {st.session_state.step_count:02d} | Action: {act_labels.get(action, 'UNKNOWN')} | Sensor: {sensor_val:.3f} | {telem.get('last_reasoning', '')}"
        st.session_state.reasoning_log.append(entry)

if step_clicked:
    for _ in range(auto_steps):
        execute_simulation_step()

if run_full_clicked:
    while not (st.session_state.terminated or st.session_state.truncated):
        execute_simulation_step()

# -------------------------------------------------------------
# TOP HUD METRICS
# -------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

success = np.array_equal(st.session_state.info["location"], st.session_state.info["eve_location"])
status_text = "🎯 TARGET SECURED" if success else ("⚠️ POWER CRITICAL" if st.session_state.info["battery"] <= 0 else "🟢 MISSION ACTIVE")
status_color = "#00ff88" if success else ("#ff3366" if st.session_state.info["battery"] <= 0 else "#00f0ff")

col1.markdown(f'<div class="metric-card"><div class="metric-lbl">Status</div><div class="metric-val" style="color:{status_color}; font-size:1.1rem;">{status_text}</div></div>', unsafe_allow_html=True)
col2.markdown(f'<div class="metric-card"><div class="metric-lbl">Battery Reserve</div><div class="metric-val">{st.session_state.info["battery"]}%</div></div>', unsafe_allow_html=True)
col3.markdown(f'<div class="metric-card"><div class="metric-lbl">Target Distance</div><div class="metric-val">{st.session_state.info["true_distance"]:.2f} m</div></div>', unsafe_allow_html=True)
latest_entropy = st.session_state.entropy_history[-1] if st.session_state.entropy_history else 0.0
col4.markdown(f'<div class="metric-card"><div class="metric-lbl">Shannon Entropy</div><div class="metric-val">{latest_entropy:.2f} <span style="font-size:0.9rem">bits</span></div></div>', unsafe_allow_html=True)
col5.markdown(f'<div class="metric-card"><div class="metric-lbl">Time Steps</div><div class="metric-val">{st.session_state.step_count}</div></div>', unsafe_allow_html=True)

st.write("")

# -------------------------------------------------------------
# MAIN TABS DISPLAY
# -------------------------------------------------------------
tab_map, tab_3d, tab_telemetry, tab_benchmark, tab_theory = st.tabs([
    "🗺️ 2D Tactical Radar",
    "🌐 3D Probability Surface",
    "📈 Telemetry & Entropy",
    "🏆 Multi-Agent Benchmark Arena",
    "📚 Theoretical Mathematics & Formulas"
])

telem = agent.get_telemetry()
belief_matrix = telem.get("belief_state", np.zeros(env.grid_size))

with tab_map:
    m_col1, m_col2 = st.columns([2.5, 1])

    with m_col1:
        # Plotly 2D Tactical Map
        fig_map = go.Figure()

        # Heatmap layer (Probability Distribution)
        fig_map.add_trace(go.Heatmap(
            z=belief_matrix.T,
            colorscale='Inferno',
            showscale=True,
            colorbar=dict(title=dict(text="Probability P(EVE)", side="right"), tickfont=dict(color="#00f0ff")),
            hoverongaps=False,
            opacity=0.85
        ))

        # Trajectory Path line
        tx = [loc[0] for loc in st.session_state.trajectory]
        ty = [loc[1] for loc in st.session_state.trajectory]
        fig_map.add_trace(go.Scatter(
            x=tx, y=ty,
            mode='lines+markers',
            line=dict(color='#00f0ff', width=3, dash='solid'),
            marker=dict(size=6, color='#00f0ff'),
            name='WALL-E Trajectory'
        ))

        # Agent Marker (WALL-E)
        curr = st.session_state.info["location"]
        fig_map.add_trace(go.Scatter(
            x=[curr[0]], y=[curr[1]],
            mode='markers+text',
            marker=dict(size=18, color='#00f0ff', symbol='circle', line=dict(color='white', width=2)),
            text=["WALL-E"],
            textposition="top center",
            name='WALL-E Position'
        ))

        # Target Marker (EVE)
        eve = st.session_state.info["eve_location"]
        fig_map.add_trace(go.Scatter(
            x=[eve[0]], y=[eve[1]],
            mode='markers+text',
            marker=dict(size=20, color='#00ff88', symbol='star', line=dict(color='white', width=2)),
            text=["EVE"],
            textposition="bottom center",
            name='EVE Target'
        ))

        # Obstacles
        if env.obstacles:
            ox = [o[0] for o in env.obstacles]
            oy = [o[1] for o in env.obstacles]
            fig_map.add_trace(go.Scatter(
                x=ox, y=oy,
                mode='markers+text',
                marker=dict(size=16, color='#ff3366', symbol='square'),
                text=["X" for _ in ox],
                textposition="middle center",
                textfont=dict(color="white", size=10),
                name='Hazard Debris'
            ))

        # Recharge Stations
        if env.recharge_stations:
            rx = [r[0] for r in env.recharge_stations]
            ry = [r[1] for r in env.recharge_stations]
            fig_map.add_trace(go.Scatter(
                x=rx, y=ry,
                mode='markers+text',
                marker=dict(size=16, color='#ffd700', symbol='diamond'),
                text=["+" for _ in rx],
                textposition="middle center",
                textfont=dict(color="black", size=12, family="Orbitron"),
                name='Solar Recharge'
            ))

        fig_map.update_layout(
            title=dict(text="TACTICAL ACTIVE SCANNER GRID (10x10)", font=dict(family="Orbitron", color="#00f0ff")),
            xaxis=dict(range=[-0.5, 9.5], dtick=1, gridcolor='rgba(255,255,255,0.1)', color="#8da2c0"),
            yaxis=dict(range=[-0.5, 9.5], dtick=1, gridcolor='rgba(255,255,255,0.1)', color="#8da2c0", scaleanchor="x", scaleratio=1),
            paper_bgcolor='rgba(10, 14, 23, 0.95)',
            plot_bgcolor='rgba(19, 27, 46, 0.9)',
            height=580,
            legend=dict(orientation="h", y=-0.15, font=dict(color="#8da2c0")),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with m_col2:
        st.markdown("#### 📡 LIVE DECISION STREAM")
        st.markdown(f"**Controller:** `{agent.name}`")
        st.markdown(f"**Peak Hypothesis:** `{telem.get('most_likely_target', [0,0])}` ({telem.get('max_confidence', 0.0)*100:.1f}%)")
        
        # Display latest reasoning logs
        st.markdown("**Chain-of-Thought Telemetry:**")
        log_container = st.container(height=420)
        with log_container:
            for log in reversed(st.session_state.reasoning_log[-12:]):
                st.markdown(f'<div class="cot-box">{log}</div>', unsafe_allow_html=True)

with tab_3d:
    st.markdown("### 🌐 3D BAYESIAN PROBABILITY TOPOLOGY")
    st.markdown("Visualizing the mathematical belief state as a continuous 3D probability manifold. As the AI scans, rings collapse into a localized Gaussian peak over EVE's location.")
    
    x_grid = np.arange(env.max_x)
    y_grid = np.arange(env.max_y)
    
    fig_3d = go.Figure(data=[go.Surface(
        z=belief_matrix.T,
        x=x_grid,
        y=y_grid,
        colorscale='Viridis',
        contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
    )])
    
    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(title='Grid X', gridcolor='rgba(255,255,255,0.2)', color='#00f0ff'),
            yaxis=dict(title='Grid Y', gridcolor='rgba(255,255,255,0.2)', color='#00f0ff'),
            zaxis=dict(title='P(Target)', gridcolor='rgba(255,255,255,0.2)', color='#00ff88'),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2))
        ),
        paper_bgcolor='rgba(10, 14, 23, 0.95)',
        height=600,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig_3d, use_container_width=True)

with tab_telemetry:
    st.markdown("### 📈 INFORMATION GAIN & RESOURCE TELEMETRY")
    t_c1, t_c2 = st.columns(2)

    with t_c1:
        # Shannon Entropy Curve
        df_ent = pd.DataFrame({
            "Step": list(range(len(st.session_state.entropy_history))),
            "Shannon Entropy (bits)": st.session_state.entropy_history
        })
        fig_ent = px.line(df_ent, x="Step", y="Shannon Entropy (bits)",
                          title="SHANNON ENTROPY H(B) REDUCTION [UNCERTAINTY IN BITS]",
                          markers=True)
        fig_ent.update_traces(line_color="#00ff88", line_width=3)
        fig_ent.update_layout(paper_bgcolor='rgba(10, 14, 23, 0.95)', plot_bgcolor='rgba(19, 27, 46, 0.8)',
                              font=dict(color="#8da2c0"))
        st.plotly_chart(fig_ent, use_container_width=True)

    with t_c2:
        # Battery Depletion Curve
        df_bat = pd.DataFrame({
            "Step": list(range(len(st.session_state.battery_history))),
            "Battery (%)": st.session_state.battery_history
        })
        fig_bat = px.line(df_bat, x="Step", y="Battery (%)",
                          title="BATTERY ENERGY CONSUMPTION TRAJECTORY",
                          markers=True)
        fig_bat.update_traces(line_color="#ff9900", line_width=3)
        fig_bat.update_layout(paper_bgcolor='rgba(10, 14, 23, 0.95)', plot_bgcolor='rgba(19, 27, 46, 0.8)',
                              font=dict(color="#8da2c0"), yaxis=dict(range=[0, 105]))
        st.plotly_chart(fig_bat, use_container_width=True)

with tab_benchmark:
    st.markdown("### 🏆 MULTI-AGENT MONTE CARLO BENCHMARK ARENA")
    st.markdown("Statistical evaluation across **25 randomized trials** under Gaussian sensor noise ($\sigma=0.1$) and hazard debris fields.")

    # Load precomputed or live benchmark results
    bench_file = "benchmark_results.json"
    if os.path.exists(bench_file):
        with open(bench_file, "r") as f:
            bench_data = json.load(f)
        df_summary = pd.DataFrame(bench_data["summary"])

        bc1, bc2 = st.columns(2)
        with bc1:
            fig_succ = px.bar(
                df_summary, x="agent", y="success_rate",
                color="agent",
                title="SUCCESS RATE COMPARISON (%)",
                text_auto='.1f',
                color_discrete_sequence=['#00f0ff', '#00ff88', '#ff3366', '#ff9900']
            )
            fig_succ.update_layout(paper_bgcolor='rgba(10, 14, 23, 0.95)', plot_bgcolor='rgba(19, 27, 46, 0.8)',
                                  font=dict(color="#8da2c0"), showlegend=False, yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig_succ, use_container_width=True)

        with bc2:
            fig_batt = px.bar(
                df_summary, x="agent", y="mean_battery",
                color="agent",
                title="MEAN BATTERY EFFICIENCY AT TERMINATION (%)",
                text_auto='.1f',
                color_discrete_sequence=['#00f0ff', '#00ff88', '#ff3366', '#ff9900']
            )
            fig_batt.update_layout(paper_bgcolor='rgba(10, 14, 23, 0.95)', plot_bgcolor='rgba(19, 27, 46, 0.8)',
                                   font=dict(color="#8da2c0"), showlegend=False, yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig_batt, use_container_width=True)

        st.dataframe(df_summary, use_container_width=True)
    else:
        st.info("Run `python benchmark.py` in the terminal to generate comparative statistical data.")

with tab_theory:
    st.markdown("### 📚 MATHEMATICAL FOUNDATIONS & RECRUITMENT DEFENSE")

    st.markdown(r"""
    #### 1. Partially Observable Markov Decision Process (POMDP)
    The navigation problem is formally structured as a 7-tuple:
    $$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$$
    - $\mathcal{S}$: True hidden state $(x_{agent}, y_{agent}, x_{target}, y_{target}, b)$.
    - $\mathcal{A}$: Discrete action space $\{0:\text{Scan}, 1:\text{North}, 2:\text{East}, 3:\text{South}, 4:\text{West}\}$.
    - $\Omega$: Observation space $z \in [0.0, 1.0]$ representing noisy distance attenuation.
    - $\mathcal{O}(z \mid s, a)$: Observation probability function conditioned on state.

    #### 2. Recursive Bayesian Estimation & Gaussian Likelihood
    Given prior belief $b_{t-1}(s)$ and observation $z_t$, the posterior update follows Bayes' Law:
    $$b_t(s) = \frac{p(z_t \mid s) b_{t-1}(s)}{\sum_{s'} p(z_t \mid s') b_{t-1}(s')}$$
    where the sensor likelihood is modeled as a normal distribution over Euclidean distance:
    $$p(z_t \mid s) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(z_t - \hat{z}(s))^2}{2\sigma^2} \right)$$

    #### 3. Information-Theoretic Uncertainty (Shannon Entropy)
    The agent quantifies its spatial epistemic uncertainty via the Shannon Entropy of the belief state:
    $$H(B) = -\sum_{x=0}^{M-1} \sum_{y=0}^{N-1} B(x, y) \log_2(B(x, y) + \epsilon)$$
    Active inference actions minimize expected future entropy: $\arg\max_a \mathbb{E}[\Delta H(B)]$.

    #### 4. Proximal Policy Optimization (PPO) Clipped Surrogate Objective
    For the deep neural policy $\pi_\theta$, parameters are optimized using the clipped surrogate advantage function:
    $$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$
    where $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$ prevents destructively large policy updates.
    """)

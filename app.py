import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
import os
import math

from super_environment import EnhancedWallEEnv
from algorithms import (
    BayesianEntropyAgent,
    InformationAStarAgent,
    CognitiveLLMAgent,
    DRLAgent
)

st.set_page_config(
    page_title="Autonomous Search & Localization Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, professional styling (minimalist light theme, no neon/sci-fi elements)
st.markdown("""
<style>
    .reportview-container, .main, .block-container {
        color: #0f172a;
        background-color: #ffffff;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 12px 16px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
        margin: 2px 0;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 600;
    }
    .log-container {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 10px 14px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.82rem;
        color: #334155;
        line-height: 1.5;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Dashboard Header
st.markdown("""
<div style="padding-bottom: 12px; border-bottom: 1px solid #e2e8f0; margin-bottom: 20px;">
    <h2 style="margin: 0; font-size: 1.6rem; font-weight: 700; color: #0f172a;">Autonomous Search & Localization Dashboard</h2>
    <p style="margin: 4px 0 0 0; font-size: 0.9rem; color: #64748b;">POMDP Grid Simulation with Bayesian Estimation and Reinforcement Learning</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR CONFIGURATION
# -------------------------------------------------------------
st.sidebar.subheader("Configuration")

agent_choice = st.sidebar.selectbox(
    "Algorithm",
    [
        "Bayesian Active Inference",
        "Cognitive CoT Planner",
        "Information A*",
        "Deep RL (PPO)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Environment Settings")

scenario_mode = st.sidebar.radio("Target Placement", ["Fixed [7, 8]", "Random Coordinate", "Custom Coordinate"])

custom_eve = [7, 8]
if scenario_mode == "Custom Coordinate":
    c1, c2 = st.sidebar.columns(2)
    custom_eve[0] = int(c1.number_input("Target X", min_value=0, max_value=9, value=7))
    custom_eve[1] = int(c2.number_input("Target Y", min_value=0, max_value=9, value=8))

sensor_noise = st.sidebar.slider("Sensor Noise (Std Dev)", min_value=0.0, max_value=0.25, value=0.10, step=0.02)
enable_obstacles = st.sidebar.checkbox("Obstacles", value=True)
enable_recharge = st.sidebar.checkbox("Solar Recharge Pods", value=True)
dynamic_target = st.sidebar.checkbox("Dynamic Target (Drifting)", value=False)

# Session State Initialization
if "env" not in st.session_state or st.sidebar.button("Reset Environment"):
    config_dict = {
        "mission_name": "Web_Simulation",
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

    st.session_state.env = EnhancedWallEEnv(
        config_path="temp_web_config.json",
        randomize_target=(scenario_mode == "Random Coordinate"),
        dynamic_target=dynamic_target
    )

    if "Bayesian" in agent_choice:
        st.session_state.agent = BayesianEntropyAgent(
            st.session_state.env.grid_size,
            sensor_variance=(sensor_noise**2 if sensor_noise > 0 else 0.005)
        )
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
    st.session_state.reasoning_log = ["System initialized. Awaiting sensor inputs."]

env = st.session_state.env
agent = st.session_state.agent
info = st.session_state.info

# Simulation Controls in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Controls")
col_b1, col_b2 = st.sidebar.columns(2)
step_clicked = col_b1.button("Step")
run_full_clicked = col_b2.button("Run Full")
steps_per_click = st.sidebar.slider("Steps per click", min_value=1, max_value=20, value=1)

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

        act_labels = {0: "Scan", 1: "Move North", 2: "Move East", 3: "Move South", 4: "Move West"}
        entry = f"Step {st.session_state.step_count:02d} | Action: {act_labels.get(action, 'Unknown')} | Sensor: {sensor_val:.3f} | {telem.get('last_reasoning', '')}"
        st.session_state.reasoning_log.append(entry)

if step_clicked:
    for _ in range(steps_per_click):
        execute_simulation_step()

if run_full_clicked:
    while not (st.session_state.terminated or st.session_state.truncated):
        execute_simulation_step()

# -------------------------------------------------------------
# TOP SUMMARY METRICS
# -------------------------------------------------------------
success = np.array_equal(st.session_state.info["location"], st.session_state.info["eve_location"])
if success:
    status_text = "Target Located"
    status_color = "#16a34a"
elif st.session_state.info["battery"] <= 0:
    status_text = "Battery Depleted"
    status_color = "#dc2626"
elif st.session_state.truncated:
    status_text = "Timeout (60 Steps)"
    status_color = "#d97706"
elif st.session_state.terminated:
    status_text = "Terminated"
    status_color = "#dc2626"
else:
    status_text = "In Progress"
    status_color = "#2563eb"

m1, m2, m3, m4, m5 = st.columns(5)
m1.markdown(f'<div class="metric-card"><div class="metric-label">Status</div><div class="metric-value" style="color: {status_color}; font-size: 1.15rem;">{status_text}</div></div>', unsafe_allow_html=True)
m2.markdown(f'<div class="metric-card"><div class="metric-label">Battery Remaining</div><div class="metric-value">{st.session_state.info["battery"]}%</div></div>', unsafe_allow_html=True)
m3.markdown(f'<div class="metric-card"><div class="metric-label">Target Distance</div><div class="metric-value">{st.session_state.info["true_distance"]:.2f} m</div></div>', unsafe_allow_html=True)
latest_ent = st.session_state.entropy_history[-1] if st.session_state.entropy_history else 0.0
m4.markdown(f'<div class="metric-card"><div class="metric-label">Shannon Entropy</div><div class="metric-value">{latest_ent:.2f} <span style="font-size:0.9rem; font-weight: normal; color: #64748b;">bits</span></div></div>', unsafe_allow_html=True)
m5.markdown(f'<div class="metric-card"><div class="metric-label">Steps Elapsed</div><div class="metric-value">{st.session_state.step_count} <span style="font-size:0.9rem; font-weight: normal; color: #64748b;">/ 60</span></div></div>', unsafe_allow_html=True)

st.write("")

# -------------------------------------------------------------
# MAIN VIEW TABS
# -------------------------------------------------------------
tab_arena, tab_telemetry, tab_benchmarks = st.tabs([
    "Arena & Belief State",
    "Telemetry & Energy",
    "Benchmark Results"
])

telem = agent.get_telemetry()
belief_matrix = telem.get("belief_state", np.zeros(env.grid_size))

with tab_arena:
    col_2d, col_3d = st.columns([1.3, 1])

    with col_2d:
        fig_2d = go.Figure()

        # Heatmap of belief state
        fig_2d.add_trace(go.Heatmap(
            z=belief_matrix.T,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title=dict(text="Probability", side="right"), tickfont=dict(color="#334155")),
            opacity=0.75,
            hoverongaps=False
        ))

        # Trajectory Path line
        tx = [loc[0] for loc in st.session_state.trajectory]
        ty = [loc[1] for loc in st.session_state.trajectory]
        fig_2d.add_trace(go.Scatter(
            x=tx, y=ty,
            mode='lines+markers',
            line=dict(color='#0f172a', width=2),
            marker=dict(size=5, color='#0f172a'),
            name='Path'
        ))

        # Agent Marker (WALL-E)
        curr = st.session_state.info["location"]
        fig_2d.add_trace(go.Scatter(
            x=[curr[0]], y=[curr[1]],
            mode='markers+text',
            marker=dict(size=16, color='#2563eb', symbol='circle', line=dict(color='#ffffff', width=2)),
            text=["WALL-E"],
            textposition="top center",
            name='WALL-E'
        ))

        # Target Marker (EVE)
        eve = st.session_state.info["eve_location"]
        fig_2d.add_trace(go.Scatter(
            x=[eve[0]], y=[eve[1]],
            mode='markers+text',
            marker=dict(size=18, color='#16a34a', symbol='star', line=dict(color='#ffffff', width=2)),
            text=["EVE"],
            textposition="bottom center",
            name='EVE'
        ))

        # Obstacles
        if env.obstacles:
            ox = [o[0] for o in env.obstacles]
            oy = [o[1] for o in env.obstacles]
            fig_2d.add_trace(go.Scatter(
                x=ox, y=oy,
                mode='markers+text',
                marker=dict(size=15, color='#64748b', symbol='square'),
                text=["X" for _ in ox],
                textposition="middle center",
                textfont=dict(color="white", size=9),
                name='Obstacle'
            ))

        # Recharge Stations
        if env.recharge_stations:
            rx = [r[0] for r in env.recharge_stations]
            ry = [r[1] for r in env.recharge_stations]
            fig_2d.add_trace(go.Scatter(
                x=rx, y=ry,
                mode='markers+text',
                marker=dict(size=15, color='#d97706', symbol='diamond'),
                text=["+" for _ in rx],
                textposition="middle center",
                textfont=dict(color="white", size=10),
                name='Recharge Pod'
            ))

        fig_2d.update_layout(
            title=dict(text="2D Arena & Belief Probability", font=dict(color="#0f172a", size=15)),
            xaxis=dict(range=[-0.5, 9.5], dtick=1, gridcolor='#f1f5f9', color="#64748b"),
            yaxis=dict(range=[-0.5, 9.5], dtick=1, gridcolor='#f1f5f9', color="#64748b", scaleanchor="x", scaleratio=1),
            paper_bgcolor='#ffffff',
            plot_bgcolor='#ffffff',
            height=540,
            legend=dict(orientation="h", y=-0.12, font=dict(color="#64748b")),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_2d, use_container_width=True)

    with col_3d:
        x_grid = np.arange(env.max_x)
        y_grid = np.arange(env.max_y)

        fig_3d = go.Figure(data=[go.Surface(
            z=belief_matrix.T,
            x=x_grid,
            y=y_grid,
            colorscale='Blues',
            contours_z=dict(show=True, usecolormap=True, project_z=True)
        )])

        fig_3d.update_layout(
            title=dict(text="3D Probability Surface", font=dict(color="#0f172a", size=15)),
            scene=dict(
                xaxis=dict(title='Grid X', gridcolor='#e2e8f0', color='#64748b'),
                yaxis=dict(title='Grid Y', gridcolor='#e2e8f0', color='#64748b'),
                zaxis=dict(title='Probability', gridcolor='#e2e8f0', color='#64748b'),
                camera=dict(eye=dict(x=1.4, y=-1.4, z=1.1))
            ),
            paper_bgcolor='#ffffff',
            height=540,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_3d, use_container_width=True)

with tab_telemetry:
    tc1, tc2 = st.columns(2)

    with tc1:
        df_entropy = pd.DataFrame({
            "Step": list(range(len(st.session_state.entropy_history))),
            "Entropy (bits)": st.session_state.entropy_history
        })
        fig_entropy = px.line(
            df_entropy, x="Step", y="Entropy (bits)",
            title="Shannon Entropy Reduction (Uncertainty in Bits)",
            markers=True
        )
        fig_entropy.update_traces(line_color="#2563eb", line_width=2.5)
        fig_entropy.update_layout(
            template="plotly_white",
            paper_bgcolor='#ffffff',
            plot_bgcolor='#ffffff',
            height=340
        )
        st.plotly_chart(fig_entropy, use_container_width=True)

    with tc2:
        df_battery = pd.DataFrame({
            "Step": list(range(len(st.session_state.battery_history))),
            "Battery (%)": st.session_state.battery_history
        })
        fig_battery = px.line(
            df_battery, x="Step", y="Battery (%)",
            title="Battery Consumption Profile",
            markers=True
        )
        fig_battery.update_traces(line_color="#d97706", line_width=2.5)
        fig_battery.update_layout(
            template="plotly_white",
            paper_bgcolor='#ffffff',
            plot_bgcolor='#ffffff',
            height=340,
            yaxis=dict(range=[0, 105])
        )
        st.plotly_chart(fig_battery, use_container_width=True)

    st.subheader("Step-by-Step Decision Log")
    log_box = st.container(height=260)
    with log_box:
        for item in reversed(st.session_state.reasoning_log):
            st.markdown(f'<div class="log-container">{item}</div>', unsafe_allow_html=True)

with tab_benchmarks:
    st.subheader("Multi-Agent Monte Carlo Benchmark (25 Trials)")
    st.caption("Statistical performance across 25 identical random seeds with Gaussian sensor noise (sigma=0.1) and obstacle fields.")

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
                title="Success Rate (%)",
                text_auto='.1f',
                color_discrete_sequence=['#2563eb', '#16a34a', '#dc2626', '#d97706']
            )
            fig_succ.update_layout(
                template="plotly_white",
                showlegend=False,
                yaxis=dict(range=[0, 100]),
                height=320
            )
            st.plotly_chart(fig_succ, use_container_width=True)

        with bc2:
            fig_batt = px.bar(
                df_summary, x="agent", y="mean_battery",
                color="agent",
                title="Mean Battery Left (%)",
                text_auto='.1f',
                color_discrete_sequence=['#2563eb', '#16a34a', '#dc2626', '#d97706']
            )
            fig_batt.update_layout(
                template="plotly_white",
                showlegend=False,
                yaxis=dict(range=[0, 100]),
                height=320
            )
            st.plotly_chart(fig_batt, use_container_width=True)

        st.dataframe(df_summary, use_container_width=True)
    else:
        st.info("Run 'python benchmark.py' in the terminal to generate comparative statistical data.")

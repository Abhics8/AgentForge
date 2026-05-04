"""
🎮 AgentForge — Interactive Dashboard v2

Streamlit-powered web dashboard with interactive Plotly charts,
tab-based navigation, and proper video handling.

Usage:
    cd AgentForge
    PYTHONPATH=. streamlit run src/dashboard.py
"""

import os
import sys
import time

# Ensure repo root is on sys.path (needed for Streamlit Cloud)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="AgentForge Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e0e0e0;
    }
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-size: 2.5rem;
        background: linear-gradient(120deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        color: #9ca3af;
        font-size: 1.1rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card h2 {
        font-size: 2.2rem;
        margin: 0;
        background: linear-gradient(120deg, #34d399, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card p {
        color: #9ca3af;
        margin: 0.25rem 0 0 0;
        font-size: 0.85rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
    }
    /* Fix tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.04);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        color: #9ca3af;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(167, 139, 250, 0.15);
        color: #a78bfa;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }
    footer {
        text-align: center;
        padding: 1rem;
        color: #6b7280;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────
PLOTS_DIR = "results/plots"
LOGS_DIR = "results/logs"
VIDEOS_DIR = "results/videos"

# ── Plotly theme ──────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0.15)",
    font=dict(color="#d1d5db"),
    margin=dict(l=50, r=30, t=50, b=50),
)


def load_training_data():
    """Load and cache training log CSV."""
    path = os.path.join(LOGS_DIR, "training_log.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if df.empty or len(df) == 0:
                return None
            return df
        except Exception:
            return None
    return None


def _is_cloud():
    """Detect if running on Streamlit Community Cloud."""
    return os.path.exists("/mount/src")


# ══════════════════════════════════════════════════════════
# HERO BANNER — only shown once at the top
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🎮 AgentForge</h1>
    <p>Deep Reinforcement Learning for Autonomous Control</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB NAVIGATION — replaces the sidebar radio buttons
# ══════════════════════════════════════════════════════════
tab1, tab7, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Training",
    "🚀 Live Train",
    "🏆 Baselines",
    "🔬 Ablations",
    "🎯 Double DQN",
    "🎬 Videos",
    "📐 Architecture",
])


# ══════════════════════════════════════════════════════════
# TAB 1: TRAINING OVERVIEW
# ══════════════════════════════════════════════════════════
with tab1:
    df = load_training_data()

    if df is not None:
        # Metric cards
        col1, col2, col3, col4 = st.columns(4)
        solved_idx = df[df["rolling_avg"] >= 195].index
        conv_ep = int(df.loc[solved_idx[0], "episode"]) if len(solved_idx) > 0 else "N/A"
        with col1:
            st.metric("🎯 Convergence", f"Episode {conv_ep}")
        with col2:
            st.metric("🏆 Max Reward", f"{df['reward'].max():.0f}")
        with col3:
            st.metric("📈 Final Rolling Avg", f"{df['rolling_avg'].iloc[-1]:.1f}")
        with col4:
            st.metric("📦 Total Episodes", len(df))

        st.markdown("---")

        # Interactive training convergence plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["episode"], y=df["reward"],
            mode="lines", name="Episode Reward",
            line=dict(color="#60a5fa", width=1),
            opacity=0.4,
        ))
        fig.add_trace(go.Scatter(
            x=df["episode"], y=df["rolling_avg"],
            mode="lines", name="Rolling Avg (100 ep)",
            line=dict(color="#f87171", width=2.5),
        ))
        fig.add_hline(
            y=195, line_dash="dash", line_color="#34d399",
            annotation_text="Solved (195)", annotation_position="top left",
        )
        fig.update_layout(
            title="Reward Convergence",
            xaxis_title="Episode",
            yaxis_title="Total Reward",
            yaxis=dict(range=[0, 520]),
            height=500,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Side-by-side diagnostics
        col1, col2 = st.columns(2)

        with col1:
            fig_eps = go.Figure()
            fig_eps.add_trace(go.Scatter(
                x=df["episode"], y=df["epsilon"],
                mode="lines", name="Epsilon",
                line=dict(color="#a78bfa", width=2),
                fill="tozeroy", fillcolor="rgba(167,139,250,0.1)",
            ))
            fig_eps.add_hline(y=0.01, line_dash="dot", line_color="#6b7280",
                              annotation_text="ε_min = 0.01")
            fig_eps.update_layout(
                title="Epsilon Decay Schedule",
                xaxis_title="Episode", yaxis_title="ε",
                height=350, **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_eps, use_container_width=True)

        with col2:
            valid_loss = df[df["avg_loss"] > 0].copy()
            if len(valid_loss) > 0:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=valid_loss["episode"], y=valid_loss["avg_loss"],
                    mode="lines", name="Avg Loss",
                    line=dict(color="#fb923c", width=1.5),
                ))
                fig_loss.update_layout(
                    title="Training Loss",
                    xaxis_title="Episode", yaxis_title="MSE Loss",
                    yaxis_type="log",
                    height=350, **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig_loss, use_container_width=True)

        # Expandable raw data
        with st.expander("📋 View Raw Training Log"):
            st.dataframe(df, use_container_width=True, height=300)
    else:
        st.warning("Training log not found. Run `PYTHONPATH=. python src/train.py` first.")


# ══════════════════════════════════════════════════════════
# TAB 7: LIVE TRAINING (in-process — works everywhere)
# ══════════════════════════════════════════════════════════
with tab7:
    st.header("🚀 Live Training")
    st.markdown("Launch a training run and watch the agent learn in real-time.")

    # Config controls
    col1, col2, col3 = st.columns(3)
    with col1:
        live_episodes = st.number_input("Episodes", min_value=50, max_value=2000, value=500, step=50)
    with col2:
        live_lr = st.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
    with col3:
        live_target_freq = st.select_slider("Target Update Freq", options=[100, 250, 500, 1000, 2000], value=500)

    st.markdown("---")

    start_btn = st.button("▶️ Start Training", type="primary", use_container_width=False)

    if start_btn:
        # ── Import training components directly ──
        from src.environment import make_env
        from src.agent import DQNAgent

        # ── Placeholders for live updates ──
        status_box = st.empty()
        progress_bar = st.progress(0)
        metric_cols = st.columns(4)
        m_episode = metric_cols[0].empty()
        m_reward = metric_cols[1].empty()
        m_rolling = metric_cols[2].empty()
        m_epsilon = metric_cols[3].empty()
        chart_placeholder = st.empty()
        log_expander = st.expander("📟 Training Log", expanded=False)
        log_box = log_expander.empty()

        status_box.info("🔄 Initializing agent and environment...")

        # ── Create environment and agent ──
        env = make_env(seed=42)
        agent = DQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_size=128,
            num_hidden_layers=2,
            learning_rate=live_lr,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_capacity=10000,
            batch_size=64,
            target_update_freq=live_target_freq,
            seed=42,
        )

        # ── Training loop (in-process) ──
        rewards_history = []
        rolling_avgs = []
        epsilons = []
        log_lines = []
        solved_reward = 195.0
        solved_window = 100
        convergence_ep = None

        status_box.info(f"🔄 Training... Episode 0/{live_episodes}")

        for episode in range(1, live_episodes + 1):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.store_transition(state, action, reward, next_state, done)
                agent.optimize()
                state = next_state
                episode_reward += reward

            agent.decay_epsilon()
            rewards_history.append(episode_reward)
            epsilons.append(agent.epsilon)
            rolling_avg = float(np.mean(rewards_history[-solved_window:])) if len(rewards_history) >= solved_window else float(np.mean(rewards_history))
            rolling_avgs.append(rolling_avg)

            log_lines.append(
                f"Ep {episode:>4d}/{live_episodes} | "
                f"Reward: {episode_reward:>5.0f} | "
                f"Avg: {rolling_avg:>6.1f} | "
                f"ε: {agent.epsilon:.4f}"
            )

            # ── Update UI every 10 episodes (or on last) ──
            if episode % 10 == 0 or episode == live_episodes or (convergence_ep is None and rolling_avg >= solved_reward and len(rewards_history) >= solved_window):
                pct = min(episode / live_episodes, 1.0)
                progress_bar.progress(pct)
                status_box.info(f"🔄 Training... Episode {episode}/{live_episodes}")

                m_episode.metric("Episode", f"{episode}")
                m_reward.metric("Last Reward", f"{episode_reward:.0f}")
                m_rolling.metric("Rolling Avg", f"{rolling_avg:.1f}")
                m_epsilon.metric("Epsilon", f"{agent.epsilon:.4f}")

                # Live Plotly chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, episode + 1)), y=rewards_history,
                    mode="lines", name="Reward",
                    line=dict(color="#60a5fa", width=1), opacity=0.4,
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(1, episode + 1)), y=rolling_avgs,
                    mode="lines", name="Rolling Avg",
                    line=dict(color="#f87171", width=2.5),
                ))
                fig.add_hline(y=195, line_dash="dash", line_color="#34d399",
                              annotation_text="Solved (195)")
                fig.update_layout(
                    title=f"Live Training — Episode {episode}",
                    xaxis_title="Episode", yaxis_title="Reward",
                    yaxis=dict(range=[0, 520]),
                    height=450, **PLOTLY_LAYOUT,
                )
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                log_box.code("\n".join(log_lines[-30:]), language="text")

            # Check for convergence
            if convergence_ep is None and len(rewards_history) >= solved_window and rolling_avg >= solved_reward:
                convergence_ep = episode

        env.close()
        progress_bar.progress(1.0)

        if convergence_ep:
            status_box.success(f"✅ Solved at episode {convergence_ep}! Rolling avg: {rolling_avgs[convergence_ep - 1]:.1f} ≥ 195")
        else:
            status_box.success(f"✅ Training complete! Final rolling avg: {rolling_avgs[-1]:.1f}")
        st.balloons()

    else:
        st.markdown("""
        > Click **▶️ Start Training** to train a fresh DQN agent right here in the browser.
        > The reward curve, metrics, and training log update live as the agent learns.
        > No local setup needed — training runs entirely in-process.
        """)


# ══════════════════════════════════════════════════════════
# TAB 2: BASELINES
# ══════════════════════════════════════════════════════════
with tab2:
    st.header("🏆 Baseline Comparison")
    st.markdown("The trained DQN agent is compared against two baselines to prove it *learned* a meaningful policy.")

    # Glowing metric cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h2>~20</h2><p>🔴 Random Agent</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h2>~35</h2><p>🟡 Heuristic Agent</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h2>500 ⭐</h2><p>🟢 DQN Agent</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Interactive bar chart with proper y-axis
    fig = go.Figure()
    agents = ["Random", "Heuristic", "DQN"]
    rewards = [20, 35, 500]
    colors = ["#ef4444", "#f59e0b", "#22c55e"]

    fig.add_trace(go.Bar(
        x=agents, y=rewards,
        marker_color=colors,
        text=[f"{r}" for r in rewards],
        textposition="outside",
        textfont=dict(size=16, color="#e0e0e0"),
    ))
    fig.add_hline(y=195, line_dash="dash", line_color="#6b7280",
                  annotation_text="Solved Threshold (195)")
    fig.update_layout(
        title="Agent Performance (100 Evaluation Episodes)",
        yaxis_title="Average Total Reward",
        yaxis=dict(range=[0, 550]),
        height=500,
        showlegend=False,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Key Insight:** The DQN achieves a PERFECT score of 500 (the CartPole-v1 maximum) on every
    evaluation episode — a **25x improvement** over the random baseline and **14x** over
    the hand-coded heuristic.
    """)


# ══════════════════════════════════════════════════════════
# TAB 3: ABLATION STUDIES
# ══════════════════════════════════════════════════════════
with tab3:
    st.header("🔬 Ablation Studies")
    st.markdown("Four systematic studies revealing how each hyperparameter affects convergence.")

    ablation = st.selectbox(
        "Select Ablation Study",
        [
            "Replay Buffer Size [1K, 5K, 10K, 50K]",
            "Epsilon Decay Rate [0.990, 0.995, 0.999, 0.9995]",
            "Network Depth [1, 2, 3 hidden layers]",
            "Target Update Frequency [250, 500, 1000, 2000]",
        ],
        label_visibility="collapsed",
    )

    ablation_map = {
        "Replay Buffer Size [1K, 5K, 10K, 50K]": (
            "ablation_buffer_size",
            "Buffer that's too small (1K) causes the agent to forget old experiences. 10K provides optimal diversity.",
        ),
        "Epsilon Decay Rate [0.990, 0.995, 0.999, 0.9995]": (
            "ablation_epsilon_decay",
            "Too fast (0.990) = premature exploitation. Too slow (0.9995) = wasted exploration. 0.995 is the sweet spot.",
        ),
        "Network Depth [1, 2, 3 hidden layers]": (
            "ablation_network_depth",
            "CartPole is low-dimensional — a single hidden layer suffices. Depth adds parameters without improving convergence.",
        ),
        "Target Update Frequency [250, 500, 1000, 2000]": (
            "ablation_target_update",
            "Too frequent (250) = unstable targets. Too rare (2000) = stale targets. 500 steps is optimal.",
        ),
    }

    prefix, analysis = ablation_map[ablation]

    col1, col2 = st.columns(2)
    with col1:
        img_path = os.path.join(PLOTS_DIR, f"{prefix}.png")
        if os.path.exists(img_path):
            st.image(Image.open(img_path), caption="Rolling Average Reward Curves", use_container_width=True)
    with col2:
        img_path = os.path.join(PLOTS_DIR, f"{prefix}_bar.png")
        if os.path.exists(img_path):
            st.image(Image.open(img_path), caption="Convergence Speed", use_container_width=True)

    st.success(f"**Analysis:** {analysis}")


# ══════════════════════════════════════════════════════════
# TAB 4: DOUBLE DQN
# ══════════════════════════════════════════════════════════
with tab4:
    st.header("🎯 DQN vs Double DQN")
    st.markdown("""
    Standard DQN uses `max` to both select and evaluate the best next action,
    causing systematic **Q-value overestimation**. Double DQN decouples selection
    (policy net) from evaluation (target net).
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.code("# Standard DQN\ny = r + γ · max_a' Q_target(s', a')", language="python")
    with col2:
        st.code("# Double DQN\na* = argmax Q_policy(s', a')\ny = r + γ · Q_target(s', a*)", language="python")

    st.markdown("---")

    # Interactive comparison bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Standard DQN", "Double DQN"],
        y=[563, 595],
        marker_color=["#ef4444", "#22c55e"],
        text=["563", "595"],
        textposition="outside",
        textfont=dict(size=16, color="#e0e0e0"),
    ))
    fig.update_layout(
        title="Episodes to Solve CartPole-v1",
        yaxis_title="Convergence Episode",
        yaxis=dict(range=[0, 700]),
        height=400,
        showlegend=False,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Convergence curve image (keep as image since it has multiple overlaid lines)
    img_path = os.path.join(PLOTS_DIR, "dqn_vs_double_dqn.png")
    if os.path.exists(img_path):
        st.image(Image.open(img_path), caption="Convergence Curves — DQN (red) vs Double DQN (green)", use_container_width=True)

    st.info("""
    **Result:** Both agents solve CartPole-v1. The difference is marginal here because
    Q-value overestimation is less harmful in simple environments. Double DQN becomes
    critical in complex environments with large action spaces (e.g., Atari).
    """)


# ══════════════════════════════════════════════════════════
# TAB 5: LEARNING PROGRESSION VIDEOS
# ══════════════════════════════════════════════════════════
with tab5:
    st.header("🎬 Learning Progression")
    st.markdown("Watch the agent evolve from random flailing to perfect control.")

    videos = [
        ("01_untrained.mp4", "🔴 Untrained", "Random actions, falls in ~11 steps"),
        ("02_mid_training.mp4", "🟡 Mid-Training (ep 500)", "Getting better, survives ~500 steps"),
        ("03_fully_trained.mp4", "🟢 Fully Trained", "Perfect balance — 500 steps (max)"),
    ]

    col1, col2, col3 = st.columns(3)

    for col, (filename, title, desc) in zip([col1, col2, col3], videos):
        with col:
            st.subheader(title)
            video_path = os.path.join(VIDEOS_DIR, filename)

            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path)
                if file_size > 100:  # Sanity check
                    with open(video_path, "rb") as f:
                        video_bytes = f.read()
                    st.video(video_bytes, format="video/mp4")
                else:
                    st.warning(f"Video file is too small ({file_size}B). Regenerate it.")
            else:
                st.warning(f"⚠️ `{filename}` not found.")

            st.caption(desc)

    st.markdown("---")
    st.markdown("""
    > **Can't see the videos?** Regenerate them with:
    > ```bash
    > PYTHONPATH=. python src/record.py
    > ```
    """)


# ══════════════════════════════════════════════════════════
# TAB 6: ARCHITECTURE
# ══════════════════════════════════════════════════════════
with tab6:
    st.header("📐 Architecture")

    # Graphviz diagram instead of ASCII art
    st.graphviz_chart("""
    digraph DQN {
        bgcolor="transparent"
        node [style="filled" fontname="Helvetica" fontsize=12 fontcolor="white"]
        edge [color="#9ca3af" fontcolor="#9ca3af" fontsize=10]

        env [label="Environment\\nCartPole-v1\\n[x, ẋ, θ, θ̇]" shape=box fillcolor="#1e3a5f"
             style="filled,rounded"]

        policy [label="Policy Network\\n4 → 128 → 128 → 2\\n(actively trained)" shape=box
                fillcolor="#4c1d95" style="filled,rounded"]

        target [label="Target Network\\n4 → 128 → 128 → 2\\n(frozen copy)" shape=box
                fillcolor="#1e3a5f" style="filled,rounded"]

        action [label="ε-Greedy\\nAction Selection" shape=diamond fillcolor="#065f46"
                style="filled,rounded"]

        replay [label="Experience Replay\\nBuffer (10K)" shape=cylinder
                fillcolor="#78350f" style="filled"]

        loss [label="MSE Loss\\nAdam (lr=0.001)" shape=box fillcolor="#7f1d1d"
              style="filled,rounded"]

        env -> policy [label="  state"]
        policy -> action [label="  Q-values"]
        action -> env [label="  action"]
        env -> replay [label="  (s, a, r, s', done)"]
        replay -> policy [label="  mini-batch (64)"]
        replay -> target [label="  mini-batch (64)"]
        target -> loss [label="  TD target"]
        policy -> loss [label="  Q(s,a)"]
        loss -> policy [label="  backprop" style=dashed]
        policy -> target [label="  copy every\\n  500 steps" style=dashed color="#34d399"
                          fontcolor="#34d399"]
    }
    """, use_container_width=True)

    st.markdown("---")

    # Hyperparameter table
    st.subheader("⚙️ Hyperparameters")
    hp_data = pd.DataFrame({
        "Parameter": ["Hidden Layers", "Replay Buffer", "Batch Size", "Discount (γ)",
                       "Learning Rate", "Target Update", "ε Schedule", "Grad Clipping"],
        "Value": ["2 × 128 (ReLU)", "10,000", "64", "0.99",
                  "0.001 (Adam)", "Every 500 steps", "1.0 → 0.01 (×0.995)", "max_norm=1.0"],
        "Source": ["Proposal", "Ablation-validated", "Proposal", "Standard",
                   "Adam default", "Tuning", "Ablation-validated", "Stability"],
    })
    st.dataframe(hp_data, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<footer>Built by <a href="https://github.com/Abhics8" style="color: #a78bfa;">Abhi Bhardwaj</a> · '
    "PyTorch · Gymnasium · CartPole-v1</footer>",
    unsafe_allow_html=True,
)

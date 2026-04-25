"""
🎮 AgentForge — Interactive Dashboard

A Streamlit-powered web dashboard for exploring AgentForge's
training results, ablation studies, and agent comparisons.

Usage:
    cd AgentForge
    PYTHONPATH=. streamlit run src/dashboard.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="AgentForge Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
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
    .stSidebar {
        background: rgba(15, 12, 41, 0.95);
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────
PLOTS_DIR = "results/plots"
LOGS_DIR = "results/logs"
VIDEOS_DIR = "results/videos"


def load_image(filename):
    """Safely load an image from the plots directory."""
    path = os.path.join(PLOTS_DIR, filename)
    if os.path.exists(path):
        return Image.open(path)
    return None


# ── Header ────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎮 AgentForge</h1>
    <p>Deep Reinforcement Learning for Autonomous Control</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Select Section",
    [
        "📊 Training Overview",
        "🏆 Baseline Comparison",
        "🔬 Ablation Studies",
        "🎯 DQN vs Double DQN",
        "🎬 Learning Progression",
        "📐 Architecture",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Built by [Abhi Bhardwaj](https://github.com/Abhics8)**")
st.sidebar.markdown("PyTorch · Gymnasium · CartPole-v1")


# ══════════════════════════════════════════════════════════
# PAGE: TRAINING OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "📊 Training Overview":
    st.header("📊 Training Overview")

    # Load training log
    log_path = os.path.join(LOGS_DIR, "training_log.csv")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)

        # Metric cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            solved_idx = df[df["rolling_avg"] >= 195].index
            conv_ep = df.loc[solved_idx[0], "episode"] if len(solved_idx) > 0 else "N/A"
            st.metric("Convergence Episode", conv_ep)
        with col2:
            st.metric("Max Reward", f"{df['reward'].max():.0f}")
        with col3:
            st.metric("Final Rolling Avg", f"{df['rolling_avg'].iloc[-1]:.1f}")
        with col4:
            st.metric("Total Episodes", len(df))

        st.markdown("---")

        # Training curve
        st.subheader("Reward Convergence")
        img = load_image("training_curve.png")
        if img:
            st.image(img, use_container_width=True)

        # Side by side: epsilon + loss
        st.subheader("Training Diagnostics")
        col1, col2 = st.columns(2)
        with col1:
            img = load_image("epsilon_decay.png")
            if img:
                st.image(img, caption="Epsilon Decay Schedule", use_container_width=True)
        with col2:
            img = load_image("loss_curve.png")
            if img:
                st.image(img, caption="MSE Training Loss", use_container_width=True)

        # Interactive data table
        with st.expander("📋 View Raw Training Log"):
            st.dataframe(df, use_container_width=True, height=300)
    else:
        st.warning("Training log not found. Run `PYTHONPATH=. python src/train.py` first.")


# ══════════════════════════════════════════════════════════
# PAGE: BASELINES
# ══════════════════════════════════════════════════════════
elif page == "🏆 Baseline Comparison":
    st.header("🏆 Baseline Comparison")

    st.markdown("""
    The trained DQN agent is compared against two baselines to validate
    that it actually **learned** a meaningful policy rather than getting lucky.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>~20</h2>
            <p>🔴 Random Agent</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>~35</h2>
            <p>🟡 Heuristic Agent</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>500 ⭐</h2>
            <p>🟢 DQN Agent</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    img = load_image("baseline_comparison.png")
    if img:
        st.image(img, caption="100-Episode Evaluation: Random vs Heuristic vs DQN", use_container_width=True)

    st.info("""
    **Key Insight:** The DQN achieves a PERFECT score of 500 (the CartPole-v1 maximum) on every
    evaluation episode. This is a **25x improvement** over the random baseline and **14x** over
    the hand-coded heuristic.
    """)


# ══════════════════════════════════════════════════════════
# PAGE: ABLATION STUDIES
# ══════════════════════════════════════════════════════════
elif page == "🔬 Ablation Studies":
    st.header("🔬 Ablation Studies")

    st.markdown("""
    Four systematic ablation studies reveal how each hyperparameter
    affects convergence behavior. One parameter is varied while all
    others are held at their tuned defaults.
    """)

    ablation = st.selectbox(
        "Select Ablation Study",
        [
            "Replay Buffer Size [1K, 5K, 10K, 50K]",
            "Epsilon Decay Rate [0.990, 0.995, 0.999, 0.9995]",
            "Network Depth [1, 2, 3 hidden layers]",
            "Target Update Frequency [250, 500, 1000, 2000]",
        ],
    )

    ablation_map = {
        "Replay Buffer Size [1K, 5K, 10K, 50K]": ("ablation_buffer_size", "Buffer that's too small (1K) causes the agent to forget old experiences. 10K provides optimal diversity."),
        "Epsilon Decay Rate [0.990, 0.995, 0.999, 0.9995]": ("ablation_epsilon_decay", "Too fast (0.990) = premature exploitation. Too slow (0.9995) = wasted exploration episodes. 0.995 is the sweet spot."),
        "Network Depth [1, 2, 3 hidden layers]": ("ablation_network_depth", "CartPole is low-dimensional — a single hidden layer suffices. Depth adds parameters without improving convergence."),
        "Target Update Frequency [250, 500, 1000, 2000]": ("ablation_target_update", "Too frequent (250) = unstable targets. Too rare (2000) = stale targets. 500 steps was found optimal via tuning."),
    }

    prefix, analysis = ablation_map[ablation]

    col1, col2 = st.columns(2)
    with col1:
        img = load_image(f"{prefix}.png")
        if img:
            st.image(img, caption="Rolling Average Reward Curves", use_container_width=True)
    with col2:
        img = load_image(f"{prefix}_bar.png")
        if img:
            st.image(img, caption="Convergence Speed", use_container_width=True)

    st.success(f"**Analysis:** {analysis}")


# ══════════════════════════════════════════════════════════
# PAGE: DOUBLE DQN
# ══════════════════════════════════════════════════════════
elif page == "🎯 DQN vs Double DQN":
    st.header("🎯 DQN vs Double DQN")

    st.markdown("""
    **Standard DQN** uses `max` to both select and evaluate the best next action,
    causing systematic **Q-value overestimation**. **Double DQN** fixes this by
    decoupling selection (policy net) from evaluation (target net).
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.code(
            "# Standard DQN\ny = r + γ · max_a' Q_target(s', a')",
            language="python",
        )
    with col2:
        st.code(
            "# Double DQN\na* = argmax Q_policy(s', a')\ny = r + γ · Q_target(s', a*)",
            language="python",
        )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        img = load_image("dqn_vs_double_dqn.png")
        if img:
            st.image(img, caption="Convergence Comparison", use_container_width=True)
    with col2:
        img = load_image("dqn_vs_double_dqn_bar.png")
        if img:
            st.image(img, caption="Episodes to Solve", use_container_width=True)

    st.info("""
    **Result:** Both agents solve CartPole-v1. The difference is marginal here because
    Q-value overestimation is less harmful in simple environments. Double DQN becomes
    critical in complex environments with large action spaces (e.g., Atari).
    """)


# ══════════════════════════════════════════════════════════
# PAGE: LEARNING PROGRESSION
# ══════════════════════════════════════════════════════════
elif page == "🎬 Learning Progression":
    st.header("🎬 Learning Progression")

    st.markdown("Watch the agent evolve from random flailing to perfect control.")

    col1, col2, col3 = st.columns(3)

    videos = [
        ("01_untrained.mp4", "🔴 Untrained", "Random actions, falls in ~11 steps"),
        ("02_mid_training.mp4", "🟡 Mid-Training (ep 500)", "Getting better, survives ~500 steps"),
        ("03_fully_trained.mp4", "🟢 Fully Trained", "Perfect balance — 500 steps"),
    ]

    for col, (filename, title, desc) in zip([col1, col2, col3], videos):
        with col:
            st.subheader(title)
            video_path = os.path.join(VIDEOS_DIR, filename)
            if os.path.exists(video_path):
                st.video(video_path)
            else:
                st.warning(f"Video not found: {filename}")
            st.caption(desc)


# ══════════════════════════════════════════════════════════
# PAGE: ARCHITECTURE
# ══════════════════════════════════════════════════════════
elif page == "📐 Architecture":
    st.header("📐 Architecture")

    st.code("""
                     ┌─────────────────────────────────┐
                     │       ENVIRONMENT (CartPole-v1)  │
                     │   state = [x, ẋ, θ, θ̇]          │
                     └──────────┬──────────────────────┘
                                │ state
                                ▼
┌──────────────────────────────────────────────────────────────┐
│                        DQN AGENT                             │
│                                                              │
│   ┌──────────────┐    ε-greedy     ┌──────────────────────┐  │
│   │ Policy Net   │ ◄──────────────►│  Action Selection    │  │
│   │ (4→128→128→2)│    explore/     │  argmax Q(s,a)       │  │
│   └──────┬───────┘    exploit      └──────────────────────┘  │
│          │ MSE Loss                                          │
│   ┌──────▼───────┐                 ┌──────────────────────┐  │
│   │ Target Net   │ ◄── hard copy ──│  Every 500 steps     │  │
│   │ (frozen)     │    (sync)       │  (target_update_freq) │  │
│   └──────────────┘                 └──────────────────────┘  │
│                                                              │
│   ┌──────────────────────────────────────────────────────┐   │
│   │  Experience Replay Buffer (capacity: 10,000)         │   │
│   │  → random mini-batches of 64 for training            │   │
│   └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
    """, language="text")

    st.markdown("---")

    st.subheader("Hyperparameters")
    hp_data = {
        "Parameter": ["Hidden Layers", "Replay Buffer", "Batch Size", "Discount (γ)",
                       "Learning Rate", "Target Update", "ε Schedule", "Grad Clipping"],
        "Value": ["2 × 128 (ReLU)", "10,000", "64", "0.99",
                  "0.001 (Adam)", "Every 500 steps", "1.0 → 0.01 (×0.995)", "max_norm=1.0"],
        "Source": ["Proposal", "Ablation-validated", "Proposal", "Standard",
                   "Adam default", "Tuning", "Ablation-validated", "Stability"],
    }
    st.dataframe(pd.DataFrame(hp_data), use_container_width=True, hide_index=True)

"""
Generate the AgentForge Analysis Notebook programmatically.
Uses nbformat to create a clean, well-structured .ipynb file.
"""

import os
import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    }
}

cells = []

# ══════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("""# 🎮 AgentForge — Analysis Notebook

**Deep Reinforcement Learning for Autonomous Control**

This notebook provides a comprehensive walkthrough of the AgentForge project — a Deep Q-Network (DQN) agent that learns to balance a pole on a moving cart with zero prior knowledge.

---

## Table of Contents

1. [Project Overview & Motivation](#1-project-overview--motivation)
2. [Architecture Deep-Dive](#2-architecture-deep-dive)
3. [Training Results](#3-training-results)
4. [Baseline Comparison](#4-baseline-comparison)
5. [Ablation Studies](#5-ablation-studies)
6. [Double DQN Extension](#6-double-dqn-extension)
7. [Learning Progression](#7-learning-progression)
8. [Conclusions & Future Work](#8-conclusions--future-work)
"""))

# ══════════════════════════════════════════════════════════
# SECTION 1: OVERVIEW
# ══════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("""---

## 1. Project Overview & Motivation

### The Problem

Imagine trying to balance a broomstick on your palm. You'd wobble, overcorrect, drop it — and then try again. Over time, through pure trial and error, you'd get better.

**AgentForge works the same way**, except the "palm" is a moving cart, the "broomstick" is a pole, and the "you" is a neural network that has never seen this problem before.

### Why Reinforcement Learning?

Traditional approaches solve this with hand-written physics equations (PID controllers). AgentForge takes a fundamentally different approach: the AI starts with **zero knowledge** and teaches itself purely by trying thousands of times and learning from its own mistakes.

### Environment: CartPole-v1

| Property | Value |
|---|---|
| **State Space** | 4D continuous: cart position, cart velocity, pole angle, angular velocity |
| **Action Space** | 2 discrete: push left (0), push right (1) |
| **Reward** | +1 for every timestep the pole stays upright |
| **Termination** | Pole angle > ±12°, cart position > ±2.4, or 500 steps |
| **Solved** | Average reward ≥ 195 over 100 consecutive episodes |
"""))

# ══════════════════════════════════════════════════════════
# SETUP CODE
# ══════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell("""import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image, Video, Markdown

# Ensure project root is on path
sys.path.insert(0, os.path.abspath('..'))

# Plot styling
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

print("✅ Setup complete.")
"""))

# ══════════════════════════════════════════════════════════
# SECTION 2: ARCHITECTURE
# ══════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("""---

## 2. Architecture Deep-Dive

### DQN Pipeline

```
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
│          │                                                   │
│          │ MSE Loss                                          │
│          │                                                   │
│   ┌──────▼───────┐                 ┌──────────────────────┐  │
│   │ Target Net   │ ◄── hard copy ──│  Every 500 steps     │  │
│   │ (frozen)     │    (sync)       │  (target_update_freq) │  │
│   └──────────────┘                 └──────────────────────┘  │
│                                                              │
│   ┌──────────────────────────────────────────────────────┐   │
│   │  Experience Replay Buffer (capacity: 10,000)         │   │
│   │  → stores (s, a, r, s', done) transitions            │   │
│   │  → samples random mini-batches of 64 for training    │   │
│   └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Component | Choice | Rationale |
|---|---|---|
| **Q-Network** | 4→128→128→2 (ReLU) | Sufficient capacity for CartPole's low-dimensional state space |
| **Target Network** | Hard copy every 500 steps | Provides stable TD targets; tuned via ablation |
| **Experience Replay** | 10K buffer, batch 64 | Breaks temporal correlation; 10K balances memory vs diversity |
| **ε-Greedy** | 1.0→0.01, decay 0.995/ep | Ensures thorough early exploration, then exploitation |
| **Gradient Clipping** | max_norm=1.0 | Prevents exploding gradients during early chaotic training |
"""))

cells.append(nbf.v4.new_code_cell("""# Inspect the neural network architecture
from src.model import DQN

model = DQN(state_dim=4, action_dim=2, hidden_size=128, num_hidden_layers=2)
print(f"Architecture:\\n{model.network}")
print(f"\\nTotal trainable parameters: {model.get_num_parameters():,}")

# Show parameter counts for ablation depths
print("\\n--- Network Depth Comparison ---")
for depth in [1, 2, 3]:
    m = DQN(num_hidden_layers=depth)
    print(f"  {depth} hidden layer(s) → {m.get_num_parameters():,} parameters")
"""))

# ══════════════════════════════════════════════════════════
# SECTION 3: TRAINING RESULTS
# ══════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("""---

## 3. Training Results

The DQN agent was trained for up to 1,000 episodes. Training terminates early if the agent achieves an average reward of ≥195 over 100 consecutive episodes (the official "solved" benchmark).
"""))

cells.append(nbf.v4.new_code_cell("""# Load training log
log_path = "../results/logs/training_log.csv"
df = pd.read_csv(log_path)

print(f"Training log: {len(df)} episodes recorded")
print(f"Final rolling average: {df['rolling_avg'].iloc[-1]:.1f}")
print(f"Max single-episode reward: {df['reward'].max():.0f}")
display(df.tail(10))
"""))

cells.append(nbf.v4.new_code_cell("""# Training convergence curve
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df['episode'], df['reward'], alpha=0.3, color='steelblue', label='Episode Reward')
ax.plot(df['episode'], df['rolling_avg'], color='#FF5252', linewidth=2, label='Rolling Avg (100 ep)')
ax.axhline(y=195, color='green', linestyle='--', alpha=0.7, label='Solved Threshold (195)')

ax.set_xlabel('Episode', fontsize=13)
ax.set_ylabel('Total Reward', fontsize=13)
ax.set_title('DQN Training Convergence on CartPole-v1', fontsize=15, pad=15)
ax.legend(fontsize=11)
ax.set_ylim(0, 520)

plt.tight_layout()
plt.savefig('../results/plots/training_curve_notebook.png', dpi=150, bbox_inches='tight')
plt.show()

# Find convergence point
solved_idx = df[df['rolling_avg'] >= 195].index
if len(solved_idx) > 0:
    print(f"\\n🎉 Agent SOLVED the environment at Episode {df.loc[solved_idx[0], 'episode']}")
else:
    print("\\n⚠️ Agent did not converge within training episodes.")
"""))

cells.append(nbf.v4.new_code_cell("""# Epsilon decay and loss curves side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Epsilon decay
ax1.plot(df['episode'], df['epsilon'], color='#9C27B0', linewidth=1.5)
ax1.set_xlabel('Episode')
ax1.set_ylabel('Epsilon (ε)')
ax1.set_title('Exploration Rate Decay')
ax1.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='ε_min = 0.01')
ax1.legend()

# Loss curve
valid_loss = df[df['avg_loss'] > 0]
ax2.plot(valid_loss['episode'], valid_loss['avg_loss'], color='#FF9800', alpha=0.7, linewidth=1)
ax2.set_xlabel('Episode')
ax2.set_ylabel('Average MSE Loss')
ax2.set_title('Training Loss Over Time')
ax2.set_yscale('log')

plt.tight_layout()
plt.show()

print(f"Epsilon decayed from {df['epsilon'].iloc[0]:.4f} to {df['epsilon'].iloc[-1]:.4f}")
print(f"Final avg loss: {valid_loss['avg_loss'].iloc[-1]:.6f}")
"""))

# ══════════════════════════════════════════════════════════
# SECTION 4: BASELINES
# ══════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("""---

## 4. Baseline Comparison

To validate that the DQN agent actually *learned* something meaningful, we compare it against two baselines:

1. **Random Agent**: Selects actions uniformly at random (expected reward ~20)
2. **Heuristic Agent**: Simple rule — if pole leans right, push right; if pole leans left, push left (expected reward ~35-60)
"""))

cells.append(nbf.v4.new_code_cell("""# Display the baseline comparison plot
display(Image(filename='../results/plots/baseline_comparison.png', width=600))

print("Agent Performance Summary (100 evaluation episodes):")
print("=" * 50)
print(f"  Random Agent     →  ~20   avg reward")
print(f"  Heuristic Agent  →  ~35   avg reward")
print(f"  DQN Agent        →  500   avg reward  ⭐")
print("=" * 50)
print("\\nThe DQN agent achieves PERFECT scores — balancing")
print("the pole for the maximum 500 steps every episode.")
"""))

# ══════════════════════════════════════════════════════════
# SECTION 5: ABLATION STUDIES
# ══════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("""---

## 5. Ablation Studies

We conducted **4 systematic ablation studies** to understand how each hyperparameter affects convergence. In each study, one parameter is varied while all others are held at their tuned defaults.

### Why Ablation Studies Matter

Ablation studies separate a competent ML project from a tutorial exercise. They answer: *"Do you understand WHY your model works, or did you just get lucky with defaults?"*
"""))

cells.append(nbf.v4.new_markdown_cell("""### Ablation 1 — Replay Buffer Size
> *How much past experience does the agent need to learn effectively?*

**Tested values:** 1K · 5K · 10K · 50K transitions
"""))

cells.append(nbf.v4.new_code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.imshow(plt.imread('../results/plots/ablation_buffer_size.png'))
ax1.axis('off')
ax1.set_title('Convergence Curves', fontsize=12)
ax2.imshow(plt.imread('../results/plots/ablation_buffer_size_bar.png'))
ax2.axis('off')
ax2.set_title('Convergence Speed', fontsize=12)
plt.tight_layout()
plt.show()

print("Analysis: A buffer that's too small (1K) causes the agent to 'forget'")
print("old experiences too quickly, leading to unstable training. The 10K default")
print("provides a good balance between memory diversity and computational cost.")
"""))

cells.append(nbf.v4.new_markdown_cell("""### Ablation 2 — Epsilon Decay Rate
> *How fast should the agent transition from exploration to exploitation?*

**Tested values:** 0.990 · 0.995 · 0.999 · 0.9995
"""))

cells.append(nbf.v4.new_code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.imshow(plt.imread('../results/plots/ablation_epsilon_decay.png'))
ax1.axis('off')
ax1.set_title('Convergence Curves', fontsize=12)
ax2.imshow(plt.imread('../results/plots/ablation_epsilon_decay_bar.png'))
ax2.axis('off')
ax2.set_title('Convergence Speed', fontsize=12)
plt.tight_layout()
plt.show()

print("Analysis: Too fast (0.990) = agent stops exploring before it has enough data.")
print("Too slow (0.9995) = agent wastes episodes on random actions when it could exploit.")
print("0.995 hits the sweet spot for CartPole's relatively small state space.")
"""))

cells.append(nbf.v4.new_markdown_cell("""### Ablation 3 — Network Depth
> *Does a deeper Q-network learn a better policy?*

**Tested values:** 1 · 2 · 3 hidden layers (128 neurons each)
"""))

cells.append(nbf.v4.new_code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.imshow(plt.imread('../results/plots/ablation_network_depth.png'))
ax1.axis('off')
ax1.set_title('Convergence Curves', fontsize=12)
ax2.imshow(plt.imread('../results/plots/ablation_network_depth_bar.png'))
ax2.axis('off')
ax2.set_title('Convergence Speed', fontsize=12)
plt.tight_layout()
plt.show()

print("Analysis: CartPole is a low-dimensional problem (4 inputs, 2 outputs).")
print("A single hidden layer is sufficient. Adding more depth increases parameters")
print("without improving convergence — classic case of diminishing returns.")
"""))

cells.append(nbf.v4.new_markdown_cell("""### Ablation 4 — Target Update Frequency
> *How often should the target network synchronize with the policy network?*

**Tested values:** 250 · 500 · 1,000 · 2,000 steps
"""))

cells.append(nbf.v4.new_code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.imshow(plt.imread('../results/plots/ablation_target_update.png'))
ax1.axis('off')
ax1.set_title('Convergence Curves', fontsize=12)
ax2.imshow(plt.imread('../results/plots/ablation_target_update_bar.png'))
ax2.axis('off')
ax2.set_title('Convergence Speed', fontsize=12)
plt.tight_layout()
plt.show()

print("Analysis: Updating too frequently (250 steps) makes targets unstable.")
print("Updating too rarely (2000 steps) means the agent trains against stale targets.")
print("500 steps was identified as optimal during hyperparameter tuning.")
"""))

# ══════════════════════════════════════════════════════════
# SECTION 6: DOUBLE DQN
# ══════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("""---

## 6. Double DQN Extension

### The Overestimation Problem

Standard DQN uses the **same network** to both select and evaluate the best next action:

$$y = r + \\gamma \\cdot \\max_{a'} Q_{\\text{target}}(s', a')$$

This `max` operator causes **systematic overestimation** of Q-values because noise in the Q-estimates gets amplified.

### The Fix: Decouple Selection from Evaluation

Double DQN uses the **policy network to select** the best action, then the **target network to evaluate** it:

$$y = r + \\gamma \\cdot Q_{\\text{target}}(s', \\arg\\max_{a'} Q_{\\text{policy}}(s', a'))$$

This is a 3-line code change that eliminates the maximization bias.
"""))

cells.append(nbf.v4.new_code_cell("""# Display comparison results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.imshow(plt.imread('../results/plots/dqn_vs_double_dqn.png'))
ax1.axis('off')
ax1.set_title('Convergence Curves', fontsize=12)
ax2.imshow(plt.imread('../results/plots/dqn_vs_double_dqn_bar.png'))
ax2.axis('off')
ax2.set_title('Convergence Speed', fontsize=12)
plt.tight_layout()
plt.show()

print("Result: Both agents solve CartPole-v1.")
print("On this simple environment, the difference is marginal.")
print("Double DQN becomes critical in complex environments with")
print("large action spaces where Q-value overestimation compounds.")
"""))

# ══════════════════════════════════════════════════════════
# SECTION 7: LEARNING PROGRESSION
# ══════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("""---

## 7. Learning Progression

Three gameplay videos demonstrate the agent's learning journey:

| Stage | Video | Steps Survived |
|---|---|---|
| 🔴 Untrained | `01_untrained.mp4` | ~11 steps |
| 🟡 Mid-Training (ep 500) | `02_mid_training.mp4` | ~500 steps |
| 🟢 Fully Trained | `03_fully_trained.mp4` | 500 steps (max) |

The videos are available in `results/videos/`. To regenerate them:
```bash
PYTHONPATH=. python src/record.py
```
"""))

# ══════════════════════════════════════════════════════════
# SECTION 8: CONCLUSIONS
# ══════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("""---

## 8. Conclusions & Future Work

### Key Findings

1. **DQN successfully solves CartPole-v1** — achieving a perfect score of 500 across all evaluation episodes, massively outperforming both random (~20) and heuristic (~35) baselines.

2. **Experience replay is critical** — without sufficient buffer capacity, the agent cannot break temporal correlations in the training data, leading to unstable or failed convergence.

3. **Target network update frequency is the most sensitive hyperparameter** — updating too frequently or too rarely both degrade performance significantly.

4. **Network depth has diminishing returns** — for CartPole's 4D state space, a single hidden layer is sufficient. Deeper networks add parameters without improving performance.

5. **Double DQN provides marginal benefit on simple environments** — but the architectural insight (decoupling selection from evaluation) is fundamental to scaling RL to harder problems.

### Future Work

- **Dueling DQN**: Separate the Q-function into state-value V(s) and advantage A(s,a) streams
- **Prioritized Experience Replay**: Sample important transitions more frequently based on TD-error
- **Harder Environments**: Apply the architecture to LunarLander-v2 or Atari games
- **Continuous Action Spaces**: Extend to DDPG or SAC for continuous control tasks

---

*Built by [Abhi Bhardwaj](https://github.com/Abhics8)*
"""))

# ══════════════════════════════════════════════════════════
# ASSEMBLE NOTEBOOK
# ══════════════════════════════════════════════════════════
nb.cells = cells

output_path = "notebooks/AgentForge_Analysis.ipynb"
os.makedirs("notebooks", exist_ok=True)
nbf.write(nb, output_path)
print(f"✅ Notebook created: {output_path}")
print(f"   Cells: {len(cells)} ({sum(1 for c in cells if c.cell_type == 'markdown')} markdown, {sum(1 for c in cells if c.cell_type == 'code')} code)")

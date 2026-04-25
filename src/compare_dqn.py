"""
Day 15: DQN vs Double DQN Head-to-Head Comparison.

Trains both agents under identical conditions and produces:
  1. Overlaid convergence curves (rolling average reward)
  2. Bar chart comparing convergence speed
  3. Console summary of results
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from src.environment import make_env
from src.agent import DQNAgent
from src.double_dqn_agent import DoubleDQNAgent
from src.utils import load_config


def train_agent(agent, env, num_episodes: int, label: str):
    """Train an agent and return episode rewards."""
    rewards = []
    print(f"\n  Training {label}...")

    for episode in range(1, num_episodes + 1):
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
        rewards.append(episode_reward)

        if episode % 100 == 0:
            avg = np.mean(rewards[-100:])
            print(f"    [{label}] Episode {episode:>4d} | Avg(100): {avg:>6.1f} | ε: {agent.epsilon:.4f}")

            if avg >= 195.0:
                print(f"    [{label}] ✅ Solved at episode {episode}!")
                break

    return rewards


def find_convergence(rewards, threshold=195.0, window=100):
    """Find the episode at which the agent first converges."""
    for i in range(window, len(rewards)):
        if np.mean(rewards[i - window : i]) >= threshold:
            return i
    return None  # Never converged


def run_comparison():
    """Run head-to-head DQN vs Double DQN comparison."""
    print("=" * 55)
    print("  Day 15: DQN vs Double DQN Comparison")
    print("=" * 55)

    config = load_config("configs/default.yaml")
    num_episodes = config["training"]["episodes"]
    seed = config["training"]["seed"]

    agent_params = dict(
        state_dim=4,
        action_dim=2,
        hidden_size=config["network"]["hidden_size"],
        num_hidden_layers=config["network"]["num_hidden_layers"],
        learning_rate=config["agent"]["learning_rate"],
        gamma=config["agent"]["gamma"],
        epsilon_start=config["agent"]["epsilon_start"],
        epsilon_end=config["agent"]["epsilon_end"],
        epsilon_decay=config["agent"]["epsilon_decay"],
        buffer_capacity=config["agent"]["replay_buffer_size"],
        batch_size=config["agent"]["batch_size"],
        target_update_freq=config["agent"]["target_update_freq"],
        seed=seed,
    )

    # ── Train Standard DQN ────────────────────────────────────────
    env_dqn = make_env(seed=seed)
    dqn_agent = DQNAgent(**agent_params)
    dqn_rewards = train_agent(dqn_agent, env_dqn, num_episodes, "DQN")
    env_dqn.close()

    # ── Train Double DQN ──────────────────────────────────────────
    env_ddqn = make_env(seed=seed)
    ddqn_agent = DoubleDQNAgent(**agent_params)
    ddqn_rewards = train_agent(ddqn_agent, env_ddqn, num_episodes, "Double DQN")
    env_ddqn.close()

    # ── Find convergence points ───────────────────────────────────
    dqn_conv = find_convergence(dqn_rewards)
    ddqn_conv = find_convergence(ddqn_rewards)

    print(f"\n  DQN converged at:        Episode {dqn_conv or 'DNF'}")
    print(f"  Double DQN converged at: Episode {ddqn_conv or 'DNF'}")

    # ── Plot 1: Overlaid Convergence Curves ───────────────────────
    plots_dir = "results/plots"
    os.makedirs(plots_dir, exist_ok=True)

    window = 50
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1a1f3a")
    ax.set_facecolor("#1a1f3a")

    # DQN curve
    dqn_rolling = [np.mean(dqn_rewards[max(0, i - window + 1):i + 1]) for i in range(len(dqn_rewards))]
    ax.plot(dqn_rolling, label="DQN", color="#FF5252", linewidth=1.5)

    # Double DQN curve
    ddqn_rolling = [np.mean(ddqn_rewards[max(0, i - window + 1):i + 1]) for i in range(len(ddqn_rewards))]
    ax.plot(ddqn_rolling, label="Double DQN", color="#4CAF50", linewidth=1.5)

    ax.axhline(y=195.0, color="gray", linestyle="--", alpha=0.6, label="Solved (195)")
    ax.set_title("DQN vs Double DQN — Convergence Comparison", fontsize=14, pad=12, color="#e0e0e0")
    ax.set_xlabel("Episode", fontsize=12, color="#9ca3af")
    ax.set_ylabel(f"Rolling Avg Reward ({window} ep)", fontsize=12, color="#9ca3af")
    ax.legend(fontsize=11, facecolor="#252b48", edgecolor="#3a4068", labelcolor="#d1d5db")
    ax.grid(True, alpha=0.15, color="#4a5078")
    ax.tick_params(colors="#9ca3af")
    for spine in ax.spines.values():
        spine.set_color("#3a4068")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/dqn_vs_double_dqn.png", dpi=300, bbox_inches="tight", facecolor="#1a1f3a")
    plt.close()
    print(f"\n  Convergence plot saved → {plots_dir}/dqn_vs_double_dqn.png")

    # ── Plot 2: Convergence Speed Bar Chart ───────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#1a1f3a")
    ax.set_facecolor("#1a1f3a")

    labels = ["DQN", "Double DQN"]
    episodes = [dqn_conv or num_episodes, ddqn_conv or num_episodes]
    colors = ["#FF5252", "#4CAF50"]

    bars = ax.bar(labels, episodes, color=colors, alpha=0.85, width=0.5)

    for bar, ep, conv in zip(bars, episodes, [dqn_conv, ddqn_conv]):
        lbl = str(ep) if conv else "DNF"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                 lbl, ha="center", fontsize=12, fontweight="bold", color="#e0e0e0")

    ax.set_title("Convergence Speed: DQN vs Double DQN", fontsize=14, pad=12, color="#e0e0e0")
    ax.set_ylabel("Episodes to Solve", fontsize=12, color="#9ca3af")
    ax.set_ylim(0, max(episodes) * 1.2)
    ax.grid(axis="y", linestyle="--", alpha=0.15, color="#4a5078")
    ax.tick_params(colors="#9ca3af")
    for spine in ax.spines.values():
        spine.set_color("#3a4068")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/dqn_vs_double_dqn_bar.png", dpi=300, bbox_inches="tight", facecolor="#1a1f3a")
    plt.close()
    print(f"  Bar chart saved → {plots_dir}/dqn_vs_double_dqn_bar.png")

    # ── Save checkpoints ──────────────────────────────────────────
    ckpt_dir = "results/checkpoints"
    dqn_agent.save(f"{ckpt_dir}/dqn_final.pt")
    ddqn_agent.save(f"{ckpt_dir}/double_dqn_final.pt")

    print(f"\n{'=' * 55}")
    print(f"  ✅ Day 15 Complete — Double DQN Extension")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    run_comparison()

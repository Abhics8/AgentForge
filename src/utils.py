"""
Utility functions for AgentForge.

Provides:
  - Config loading from YAML files
  - (More utilities like plotting will be added here in Phase 2)
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """
    Load hyperparameters from a YAML config file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Dictionary of configuration values
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def plot_training_curve(
    rewards: List[float],
    window: int = 100,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Plot episode rewards with rolling average."""
    episodes = range(1, len(rewards) + 1)

    rolling_avg = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        rolling_avg.append(np.mean(rewards[start : i + 1]))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, alpha=0.3, color="#4A90D9", label="Reward")
    ax.plot(episodes, rolling_avg, color="#E74C3C", linewidth=2.0, label=f"Avg ({window})")
    ax.axhline(y=195, color="#2ECC71", linestyle="--", label="Solved (195)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("AgentForge: Training Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_epsilon_decay(
    epsilons: List[float],
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Plot epsilon decay over episodes."""
    episodes = range(1, len(epsilons) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, epsilons, color="#9B59B6", linewidth=2.0)
    ax.fill_between(episodes, epsilons, alpha=0.2, color="#9B59B6")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon (ε)")
    ax.set_title("Epsilon-Greedy Exploration Decay")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_loss_curve(
    losses: List[float],
    window: int = 50,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Plot training loss over optimization steps with smoothing."""
    steps = range(1, len(losses) + 1)

    smoothed = []
    for i in range(len(losses)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(losses[start : i + 1]))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, alpha=0.2, color="#3498DB", linewidth=0.5, label="Raw Loss")
    ax.plot(steps, smoothed, color="#E67E22", linewidth=2.0, label=f"Smoothed ({window})")

    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()

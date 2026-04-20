"""
Day 12 & 13 Ablation Studies for AgentForge.

Tests how the DQN agent's convergence behavior changes when we
systematically vary individual hyperparameters while keeping
all others fixed at their tuned defaults.

Ablation 1: Replay Buffer Size    [1K, 5K, 10K, 50K]
Ablation 2: Epsilon Decay Rate    [0.990, 0.995, 0.999, 0.9995]
Ablation 3: Network Depth         [1, 2, 3 hidden layers]
Ablation 4: Target Update Freq    [250, 500, 1000, 2000 steps]
"""

import os
import copy
import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from src.train import train


def run_single_ablation(
    base_config_path: str,
    param_section: str,
    param_name: str,
    values: list,
    display_name: str,
) -> Dict[str, Any]:
    """
    Run a single ablation study by varying one parameter across
    multiple values while keeping all other hyperparameters fixed.

    Args:
        base_config_path: Path to the base YAML config.
        param_section: Top-level section in the YAML (e.g. 'agent', 'network').
        param_name: Key within that section to vary.
        values: List of values to test.
        display_name: Human-readable name for plots.

    Returns:
        Dictionary mapping each value to its reward curve.
    """
    results = {}

    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    for val in values:
        print(f"\n--- {display_name} = {val} ---")

        # Deep copy and patch the config
        cfg = copy.deepcopy(base_config)
        cfg[param_section][param_name] = val

        # Write temporary config
        tmp_path = f"configs/_ablation_tmp.yaml"
        with open(tmp_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        # Train with this config
        try:
            result = train(config_path=tmp_path)
            results[val] = result["episode_rewards"]
        except Exception as e:
            print(f"  FAILED: {e}")
            results[val] = []

    # Clean up temp file
    if os.path.exists("configs/_ablation_tmp.yaml"):
        os.remove("configs/_ablation_tmp.yaml")

    return results


def plot_ablation(
    results: Dict,
    title: str,
    xlabel: str,
    save_path: str,
    window: int = 50,
):
    """
    Plot overlaid rolling-average reward curves for each ablation value.
    """
    plt.figure(figsize=(10, 6))

    colors = ["#FF5252", "#FFA000", "#4CAF50", "#2196F3", "#9C27B0"]

    for i, (val, rewards) in enumerate(results.items()):
        if not rewards:
            continue
        # Compute rolling average
        rolling = []
        for j in range(len(rewards)):
            start = max(0, j - window + 1)
            rolling.append(np.mean(rewards[start : j + 1]))

        color = colors[i % len(colors)]
        plt.plot(rolling, label=f"{xlabel}={val}", color=color, linewidth=1.5)

    plt.axhline(y=195.0, color="gray", linestyle="--", alpha=0.6, label="Solved (195)")
    plt.title(title, fontsize=14, pad=12)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel(f"Rolling Avg Reward ({window} ep)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {save_path}")


def plot_convergence_bar(
    results: Dict,
    title: str,
    xlabel: str,
    save_path: str,
    solved_threshold: float = 195.0,
    solved_window: int = 100,
):
    """
    Bar chart showing at which episode each configuration converged.
    If it never converged, show 1000 (max episodes) in red.
    """
    labels = []
    convergence_episodes = []
    bar_colors = []

    for val, rewards in results.items():
        labels.append(str(val))

        # Find convergence episode
        conv_ep = len(rewards)  # default: never converged
        for i in range(solved_window, len(rewards)):
            if np.mean(rewards[i - solved_window : i]) >= solved_threshold:
                conv_ep = i
                break

        convergence_episodes.append(conv_ep)
        bar_colors.append("#4CAF50" if conv_ep < len(rewards) else "#FF5252")

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, convergence_episodes, color=bar_colors, alpha=0.85)

    # Add value labels
    for bar, ep in zip(bars, convergence_episodes):
        yval = bar.get_height()
        lbl = f"{ep}" if ep < 1000 else "DNF"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 15,
            lbl,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.title(title, fontsize=14, pad=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Convergence Episode", fontsize=12)
    plt.ylim(0, 1100)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {save_path}")


def run_all_ablations():
    """Run all 4 ablation studies from the proposal."""

    config_path = "configs/default.yaml"
    plots_dir = "results/plots"

    # ── Ablation 1: Replay Buffer Size ────────────────────────────
    print("\n" + "=" * 55)
    print("  ABLATION 1: Replay Buffer Size")
    print("=" * 55)

    buf_results = run_single_ablation(
        config_path, "agent", "replay_buffer_size",
        values=[1000, 5000, 10000, 50000],
        display_name="Buffer Size",
    )
    plot_ablation(
        buf_results,
        "Ablation: Effect of Replay Buffer Size on Convergence",
        "Buffer",
        f"{plots_dir}/ablation_buffer_size.png",
    )
    plot_convergence_bar(
        buf_results,
        "Convergence Speed vs. Replay Buffer Size",
        "Buffer Size",
        f"{plots_dir}/ablation_buffer_size_bar.png",
    )

    # ── Ablation 2: Epsilon Decay Rate ────────────────────────────
    print("\n" + "=" * 55)
    print("  ABLATION 2: Epsilon Decay Rate")
    print("=" * 55)

    eps_results = run_single_ablation(
        config_path, "agent", "epsilon_decay",
        values=[0.990, 0.995, 0.999, 0.9995],
        display_name="Epsilon Decay",
    )
    plot_ablation(
        eps_results,
        "Ablation: Effect of Epsilon Decay Rate on Convergence",
        "ε-decay",
        f"{plots_dir}/ablation_epsilon_decay.png",
    )
    plot_convergence_bar(
        eps_results,
        "Convergence Speed vs. Epsilon Decay Rate",
        "Epsilon Decay Rate",
        f"{plots_dir}/ablation_epsilon_decay_bar.png",
    )

    # ── Ablation 3: Network Depth ─────────────────────────────────
    print("\n" + "=" * 55)
    print("  ABLATION 3: Network Depth")
    print("=" * 55)

    depth_results = run_single_ablation(
        config_path, "network", "num_hidden_layers",
        values=[1, 2, 3],
        display_name="Hidden Layers",
    )
    plot_ablation(
        depth_results,
        "Ablation: Effect of Network Depth on Convergence",
        "Layers",
        f"{plots_dir}/ablation_network_depth.png",
    )
    plot_convergence_bar(
        depth_results,
        "Convergence Speed vs. Network Depth",
        "Number of Hidden Layers",
        f"{plots_dir}/ablation_network_depth_bar.png",
    )

    # ── Ablation 4: Target Update Frequency ───────────────────────
    print("\n" + "=" * 55)
    print("  ABLATION 4: Target Update Frequency")
    print("=" * 55)

    target_results = run_single_ablation(
        config_path, "agent", "target_update_freq",
        values=[250, 500, 1000, 2000],
        display_name="Target Update Freq",
    )
    plot_ablation(
        target_results,
        "Ablation: Effect of Target Update Frequency on Convergence",
        "Steps",
        f"{plots_dir}/ablation_target_update.png",
    )
    plot_convergence_bar(
        target_results,
        "Convergence Speed vs. Target Update Frequency",
        "Target Update Frequency (steps)",
        f"{plots_dir}/ablation_target_update_bar.png",
    )

    print("\n" + "=" * 55)
    print("  ✅ All 4 Ablation Studies Complete!")
    print(f"  Plots saved to {plots_dir}/")
    print("=" * 55)


if __name__ == "__main__":
    run_all_ablations()

"""
Day 14: Video Recording — Before, During, and After Learning.

Records the DQN agent's gameplay at three stages to visually
demonstrate the learning progression:

  1. Untrained  (random flailing, falls in ~10 steps)
  2. Mid-train  (improving but unstable, ~100-200 steps)
  3. Fully trained (perfect balance, 500 steps)

Uses Gymnasium's RecordVideo wrapper to output .mp4 files.
"""

import os
import shutil
import numpy as np
import torch

from src.environment import make_env
from src.agent import DQNAgent
from src.utils import load_config


def record_agent(agent, label: str, video_dir: str, epsilon: float = 0.0):
    """
    Record a single episode of the agent playing CartPole.

    Args:
        agent: The DQNAgent (or None for random).
        label: Label for the video filename.
        video_dir: Directory to save the video.
        epsilon: Exploration rate to use during recording.
    """
    staging_dir = os.path.join(video_dir, f"_staging_{label}")
    os.makedirs(staging_dir, exist_ok=True)

    env = make_env(
        seed=42,
        record_video=True,
        video_dir=staging_dir,
        video_episode_trigger=lambda ep: True,  # Record every episode
    )

    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        if agent is None:
            action = env.sample_random_action()
        else:
            original_eps = agent.epsilon
            agent.epsilon = epsilon
            action = agent.select_action(state)
            agent.epsilon = original_eps

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    env.close()

    # Rename the generated video to a clean filename
    for f in os.listdir(staging_dir):
        if f.endswith(".mp4"):
            src = os.path.join(staging_dir, f)
            dst = os.path.join(video_dir, f"{label}.mp4")
            shutil.move(src, dst)
            break

    # Clean up staging directory
    shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"  [{label}] Recorded: {steps} steps, reward: {total_reward:.0f}")
    return steps, total_reward


def record_all():
    """Record the three-stage learning progression."""
    print("=" * 55)
    print("  Day 14: Agent Gameplay Video Recordings")
    print("=" * 55)

    video_dir = "results/videos"
    os.makedirs(video_dir, exist_ok=True)

    config = load_config("configs/default.yaml")

    # ── Stage 1: Untrained Agent (random flailing) ────────────────
    print("\n📹 Recording Stage 1: Untrained Agent (random actions)...")
    record_agent(agent=None, label="01_untrained", video_dir=video_dir)

    # ── Stage 2: Mid-Training Agent (~500 episodes) ───────────────
    print("\n📹 Recording Stage 2: Mid-Training Agent...")
    mid_agent = DQNAgent(
        hidden_size=config["network"]["hidden_size"],
        num_hidden_layers=config["network"]["num_hidden_layers"],
    )
    # Try to load a mid-training checkpoint
    mid_checkpoint = "results/checkpoints/agent_ep500.pt"
    if os.path.exists(mid_checkpoint):
        mid_agent.load(mid_checkpoint)
        print(f"  Loaded checkpoint: {mid_checkpoint}")
    else:
        # Fallback: load ep300 or ep400
        for fallback in ["agent_ep400.pt", "agent_ep300.pt", "agent_ep200.pt"]:
            path = f"results/checkpoints/{fallback}"
            if os.path.exists(path):
                mid_agent.load(path)
                print(f"  Loaded fallback checkpoint: {path}")
                break
    record_agent(agent=mid_agent, label="02_mid_training", video_dir=video_dir, epsilon=0.05)

    # ── Stage 3: Fully Trained Agent (perfect balance) ────────────
    print("\n📹 Recording Stage 3: Fully Trained Agent...")
    trained_agent = DQNAgent(
        hidden_size=config["network"]["hidden_size"],
        num_hidden_layers=config["network"]["num_hidden_layers"],
    )
    final_checkpoint = "results/checkpoints/agent_final.pt"
    if os.path.exists(final_checkpoint):
        trained_agent.load(final_checkpoint)
        print(f"  Loaded checkpoint: {final_checkpoint}")
    else:
        print("  ⚠️  No final checkpoint found! Run training first.")
        return
    record_agent(agent=trained_agent, label="03_fully_trained", video_dir=video_dir, epsilon=0.0)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  ✅ All videos saved to {video_dir}/")
    print(f"{'=' * 55}")

    videos = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    for v in sorted(videos):
        size = os.path.getsize(os.path.join(video_dir, v))
        print(f"  🎬 {v}  ({size / 1024:.1f} KB)")


if __name__ == "__main__":
    record_all()

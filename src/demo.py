"""
🎮 AgentForge — Live Demo Script

Run this during your presentation to show the trained DQN agent
balancing the pole in real-time with a visual pygame window.

Usage:
    PYTHONPATH=. python src/demo.py              # Trained agent (perfect balance)
    PYTHONPATH=. python src/demo.py --untrained   # Untrained agent (random flailing)
    PYTHONPATH=. python src/demo.py --side-by-side # Both agents side by side
"""

import argparse
import time
import numpy as np
import gymnasium as gym

from src.agent import DQNAgent
from src.utils import load_config


def run_demo(trained: bool = True, speed: float = 0.02):
    """
    Run a single visual demo episode.

    Args:
        trained: If True, load the trained checkpoint. If False, use random actions.
        speed: Delay between frames in seconds (lower = faster).
    """
    label = "TRAINED DQN Agent" if trained else "UNTRAINED Agent (Random)"
    print(f"\n{'='*50}")
    print(f"  🎮 Live Demo: {label}")
    print(f"{'='*50}")

    env = gym.make("CartPole-v1", render_mode="human")

    agent = None
    if trained:
        config = load_config("configs/default.yaml")
        agent = DQNAgent(
            hidden_size=config["network"]["hidden_size"],
            num_hidden_layers=config["network"]["num_hidden_layers"],
        )
        agent.load("results/checkpoints/agent_final.pt")
        agent.epsilon = 0.0  # Pure exploitation, no randomness
        print("  ✅ Loaded trained model from results/checkpoints/agent_final.pt")

    state, _ = env.reset(seed=42)
    state = np.array(state, dtype=np.float32)
    done = False
    total_reward = 0
    steps = 0

    print(f"  ▶ Starting episode...\n")

    while not done:
        if agent is not None:
            action = agent.select_action(state)
        else:
            action = env.action_space.sample()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = np.array(next_state, dtype=np.float32)
        total_reward += reward
        steps += 1

        time.sleep(speed)  # Control playback speed

    print(f"  🏁 Episode finished: {steps} steps, reward: {total_reward:.0f}")

    if trained and total_reward >= 195:
        print(f"  🎉 PERFECT — Agent balanced for {steps} steps!")
    elif not trained:
        print(f"  💀 Agent fell after just {steps} steps (as expected)")

    env.close()
    return steps, total_reward


def run_side_by_side():
    """Run untrained then trained back-to-back for dramatic effect."""
    print("\n" + "=" * 50)
    print("  🎬 Side-by-Side Comparison")
    print("  First: UNTRAINED agent (watch it fail)")
    print("  Then:  TRAINED agent (watch it succeed)")
    print("=" * 50)

    input("\n  Press ENTER to start the UNTRAINED agent...")
    steps1, reward1 = run_demo(trained=False, speed=0.03)

    input("\n  Press ENTER to start the TRAINED agent...")
    steps2, reward2 = run_demo(trained=True, speed=0.01)

    print(f"\n{'='*50}")
    print(f"  📊 Comparison")
    print(f"  Untrained: {steps1:>4} steps  |  Trained: {steps2:>4} steps")
    print(f"  That's a {steps2/max(steps1,1):.0f}x improvement from zero-knowledge learning!")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgentForge Live Demo")
    parser.add_argument("--untrained", action="store_true", help="Show untrained agent")
    parser.add_argument("--side-by-side", action="store_true", help="Show both agents back to back")
    parser.add_argument("--speed", type=float, default=0.02, help="Frame delay in seconds")
    args = parser.parse_args()

    if args.side_by_side:
        run_side_by_side()
    elif args.untrained:
        run_demo(trained=False, speed=args.speed)
    else:
        run_demo(trained=True, speed=args.speed)

"""
Training loop for AgentForge DQN agent.

Implements the fundamental reinforcement learning cycle:
  1. Observe state
  2. Select action 
  3. Execute action
  4. Store transition
  5. Optimize policy network
  6. Decay epsilon
"""

import os
import csv
import time
import numpy as np
from typing import Optional, Dict, Any

from src.environment import make_env
from src.agent import DQNAgent
from src.utils import (
    load_config,
    plot_training_curve,
    plot_epsilon_decay,
    plot_loss_curve,
)


def train(
    config_path: str = "configs/default.yaml",
    episodes_override: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Train the DQN agent on CartPole-v1.

    Args:
        config_path: Path to hyperparameter config file
        episodes_override: Optional override for number of episodes 
                           (useful for short test runs)

    Returns:
        Dictionary containing simple training results
    """
    config = load_config(config_path)

    # Extract config values
    seed = config["training"]["seed"]
    num_episodes = episodes_override or config["training"]["episodes"]

    batch_size = config["agent"]["batch_size"]
    gamma = config["agent"]["gamma"]
    epsilon_start = config["agent"]["epsilon_start"]
    epsilon_end = config["agent"]["epsilon_end"]
    epsilon_decay = config["agent"]["epsilon_decay"]
    learning_rate = config["agent"]["learning_rate"]
    buffer_capacity = config["agent"]["replay_buffer_size"]
    target_update_freq = config["agent"]["target_update_freq"]

    hidden_size = config["network"]["hidden_size"]
    num_hidden_layers = config["network"]["num_hidden_layers"]
    log_interval = config["logging"]["log_interval"]
    save_interval = config["logging"]["save_interval"]
    results_dir = config["logging"]["results_dir"]
    solved_reward = config["environment"]["solved_reward"]
    solved_window = config["environment"]["solved_window"]

    # Create environment
    env = make_env(seed=seed)
    
    # Create agent
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        seed=seed,
    )

    episode_rewards = []
    epsilons = []
    convergence_episode = None
    
    # Ensure results directories exist
    os.makedirs(f"{results_dir}/plots", exist_ok=True)
    os.makedirs(f"{results_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{results_dir}/logs", exist_ok=True)

    # Setup CSV logging
    log_file = f"{results_dir}/logs/training_log.csv"
    csv_file = open(log_file, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "reward", "epsilon", "avg_loss", "steps", "rolling_avg"])

    print(f"\n{'='*50}")
    print(f"  AgentForge — Training Initialized")
    print(f"  Episodes: {num_episodes}")
    print(f"{'='*50}\n")

    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        steps = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            agent.optimize()

            state = next_state
            episode_reward += reward
            steps += 1

        agent.decay_epsilon()
        episode_rewards.append(episode_reward)
        epsilons.append(agent.epsilon)

        # Compute metrics
        rolling_avg = np.mean(episode_rewards[-solved_window:]) if len(episode_rewards) >= solved_window else np.mean(episode_rewards)
        avg_loss = np.mean(agent.training_losses[-steps:]) if agent.training_losses and steps > 0 else 0.0

        # Log to CSV
        csv_writer.writerow([episode, episode_reward, agent.epsilon, avg_loss, steps, rolling_avg])

        # Check for convergence
        if len(episode_rewards) >= solved_window and rolling_avg >= solved_reward:
            convergence_episode = episode
            print(f"\n🎉 SOLVED at episode {episode}! Rolling avg: {rolling_avg:.1f} >= {solved_reward}")
            # Save final model state before breaking
            agent.save(f"{results_dir}/checkpoints/agent_final.pt")
            break

        if episode % log_interval == 0:
            print(
                f"Episode {episode:>4d}/{num_episodes} | "
                f"Reward: {episode_reward:>5.1f} | "
                f"Avg({solved_window}): {rolling_avg:>5.1f} | "
                f"ε: {agent.epsilon:.4f}"
            )
            
        # Save checkpoint periodically
        if episode % save_interval == 0:
            agent.save(f"{results_dir}/checkpoints/agent_ep{episode}.pt")

    csv_file.close()
    
    # Save final model if we didn't exit early via convergence
    if convergence_episode is None:
        agent.save(f"{results_dir}/checkpoints/agent_final.pt")

    elapsed = time.time() - start_time
    env.close()

    print(f"\n{'='*50}")
    print(f"  Training Complete in {elapsed:.1f}s")
    if convergence_episode:
        print(f"  Converged at episode {convergence_episode}")
    else:
        print(f"  Failed to converge.")
    print(f"{'='*50}\n")
    
    # Generate plots
    plot_training_curve(
        episode_rewards,
        window=solved_window,
        save_path=f"{results_dir}/plots/training_curve.png",
    )
    plot_epsilon_decay(
        epsilons,
        save_path=f"{results_dir}/plots/epsilon_decay.png",
    )
    if agent.training_losses:
        plot_loss_curve(
            agent.training_losses,
            save_path=f"{results_dir}/plots/loss_curve.png",
        )
    print(f"Plots saved to {results_dir}/plots/")

    return {
        "episode_rewards": episode_rewards,
        "epsilons": epsilons,
        "convergence_episode": convergence_episode,
        "total_time": elapsed,
    }


if __name__ == "__main__":
    # Run full 1000-episode training run
    train()

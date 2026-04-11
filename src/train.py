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

import time
import numpy as np
from typing import Optional, Dict, Any

from src.environment import make_env
from src.agent import DQNAgent
from src.utils import load_config


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

        if episode % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            print(
                f"Episode {episode:>4d}/{num_episodes} | "
                f"Reward: {episode_reward:>5.1f} | "
                f"Avg(last 10): {avg_reward:>5.1f} | "
                f"ε: {agent.epsilon:.4f}"
            )

    elapsed = time.time() - start_time
    env.close()

    print(f"\n{'='*50}")
    print(f"  Training Complete in {elapsed:.1f}s")
    print(f"{'='*50}\n")

    return {
        "episode_rewards": episode_rewards,
    }


if __name__ == "__main__":
    # Day 6 requirement: first 100-episode test run
    train(episodes_override=100)

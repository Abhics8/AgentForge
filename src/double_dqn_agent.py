"""
Double DQN Agent — Day 15 Extension.

The key insight: standard DQN uses the same network to both SELECT
and EVALUATE the best next action, which causes systematic
overestimation of Q-values.

Double DQN fixes this by decoupling selection from evaluation:
  - Policy network SELECTS the best action:  a* = argmax_a Q_policy(s', a)
  - Target network EVALUATES that action:    Q_target(s', a*)

This small change significantly reduces overestimation bias and
often leads to more stable, faster convergence.

Reference: van Hasselt et al., "Deep Reinforcement Learning with
           Double Q-learning", AAAI 2016.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from src.model import DQN
from src.replay_buffer import ReplayBuffer


class DoubleDQNAgent:
    """
    Double DQN Agent for CartPole-v1.

    Identical to DQNAgent except for the target computation in optimize():
      - DQN:        y = r + γ · max_a' Q_target(s', a')
      - Double DQN: y = r + γ · Q_target(s', argmax_a' Q_policy(s', a'))

    Args:
        state_dim: Dimension of the state space (4)
        action_dim: Number of possible actions (2)
        hidden_size: Neurons per hidden layer (128)
        num_hidden_layers: Number of hidden layers (2)
        learning_rate: Adam optimizer learning rate (0.001)
        gamma: Discount factor for future rewards (0.99)
        epsilon_start: Initial exploration rate (1.0)
        epsilon_end: Minimum exploration rate (0.01)
        epsilon_decay: Multiplicative decay per episode (0.995)
        buffer_capacity: Replay buffer size (10,000)
        batch_size: Mini-batch size for training (64)
        target_update_freq: Steps between target network updates (500)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        hidden_size: int = 128,
        num_hidden_layers: int = 2,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        seed: int = 42,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy network (actively trained)
        self.policy_net = DQN(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
        ).to(self.device)

        # Target network (frozen copy)
        self.target_net = DQN(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, seed=seed)

        self.total_steps = 0
        self.training_losses = []

    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def optimize(self) -> Optional[float]:
        """
        Perform a single optimization step using DOUBLE DQN targets.

        The critical difference from standard DQN:
          Standard DQN:  y = r + γ · max_a' Q_target(s', a')
          Double DQN:    y = r + γ · Q_target(s', argmax_a' Q_policy(s', a'))

        The policy net picks the action, the target net evaluates it.
        This decoupling eliminates the maximization bias.
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Q(s, a) — current Q-value prediction
        current_q = self.policy_net(states).gather(1, actions)

        # ═══ DOUBLE DQN TARGET ═══
        with torch.no_grad():
            # Step 1: Policy net SELECTS the best action for next state
            best_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)

            # Step 2: Target net EVALUATES that action's Q-value
            next_q = self.target_net(next_states).gather(1, best_actions)

            # Compute target
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_value = loss.item()
        self.training_losses.append(loss_value)

        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self.update_target_network()

        return loss_value

    def update_target_network(self):
        """Hard update: copy policy weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """Save agent checkpoint."""
        checkpoint = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "training_losses": self.training_losses,
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.total_steps = checkpoint["total_steps"]
        self.training_losses = checkpoint.get("training_losses", [])

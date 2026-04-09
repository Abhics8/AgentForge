"""
DQN Agent — the core of AgentForge.

Combines the Q-Network, Target Network, Experience Replay, and
Epsilon-Greedy exploration into a single cohesive agent.

From the proposal:
  - Experience Replay: buffer of 10,000 transitions, sample batches of 64
  - Target Network: updated every 1,000 steps (hard copy)
  - Epsilon-Greedy: ε starts at 1.0, decays to 0.01 (factor 0.995/episode)
  - Optimizer: Adam, lr=0.001
  - Loss: MSE between predicted Q(s,a) and target y = r + γ·max Q'(s',a')
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from src.model import DQN
from src.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network Agent for CartPole-v1.

    The agent maintains two networks:
      1. Policy Network: actively trained, used for action selection
      2. Target Network: frozen copy, used for computing stable TD targets

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
        target_update_freq: Steps between target network updates (1,000)
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
        target_update_freq: int = 1000,
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

        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy network (actively trained)
        self.policy_net = DQN(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
        ).to(self.device)

        # Target network (frozen copy, updated periodically)
        self.target_net = DQN(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
        ).to(self.device)

        # Initialize target with same weights as policy
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net is never in training mode

        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, seed=seed)

        # Step counter for target network updates
        self.total_steps = 0

        # Training metrics
        self.training_losses = []

    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action using epsilon-greedy policy.

        With probability ε: random action (exploration)
        With probability 1-ε: highest Q-value action (exploitation)

        Args:
            state: Current state observation (4,)

        Returns:
            action: 0 (push left) or 1 (push right)
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def optimize(self) -> Optional[float]:
        """
        Perform a single optimization step on a random mini-batch.

        From the proposal:
          "Compute the target Q-value: y = r + γ · max Q'(s', a')
           using the target network. Update the policy network by
           minimizing MSE loss between Q(s, a) and the target y,
           using Adam (lr = 0.001)"

        Returns:
            loss value, or None if buffer isn't ready
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        # Sample random mini-batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Q(s, a) — Q-value of the action we actually took
        current_q = self.policy_net(states).gather(1, actions)

        # Target: y = r + γ · max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # MSE loss
        loss = self.criterion(current_q, target_q)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_value = loss.item()
        self.training_losses.append(loss_value)

        # Update target network every target_update_freq steps
        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self.update_target_network()

        return loss_value

    def update_target_network(self) -> None:
        """
        Hard update: copy policy network weights to target network.

        From the proposal:
          "Every 1,000 steps, copy the policy network's weights
           into the target network"
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        """
        Decay epsilon after each episode.

        ε = max(ε_end, ε × ε_decay)
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str) -> None:
        """Save agent checkpoint to disk."""
        checkpoint = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "training_losses": self.training_losses,
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str) -> None:
        """Load agent checkpoint from disk."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.total_steps = checkpoint["total_steps"]
        self.training_losses = checkpoint.get("training_losses", [])

    def __repr__(self) -> str:
        return (
            f"DQNAgent(\n"
            f"  epsilon={self.epsilon:.4f},\n"
            f"  total_steps={self.total_steps:,},\n"
            f"  buffer_size={len(self.replay_buffer):,},\n"
            f"  device={self.device}\n"
            f")"
        )


if __name__ == "__main__":
    agent = DQNAgent()
    print(agent)
    print(f"\nPolicy Network:\n{agent.policy_net}")

    # Test action selection
    state = np.random.randn(4).astype(np.float32)
    action = agent.select_action(state)
    print(f"\nState: {state}")
    print(f"Action: {action}")

    # Test optimize
    for i in range(100):
        s = np.random.randn(4).astype(np.float32)
        agent.store_transition(s, np.random.randint(2), 1.0, s, False)

    loss = agent.optimize()
    print(f"\nOptimization loss: {loss}")
    print(f"After: {agent}")

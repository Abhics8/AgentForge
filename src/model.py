"""
Deep Q-Network (DQN) Architecture.

A multi-layer feedforward neural network that approximates the Q-value
function: Q(s, a) ≈ expected future reward for taking action a in state s.

From the proposal:
  "The Q-network is a simple feedforward neural network built in PyTorch"

  Layer       | Configuration                | Purpose
  ------------|------------------------------|----------------------------------
  Input       | 4 neurons (state dimensions) | Takes in the observation vector
  Hidden 1    | 128 neurons, ReLU activation | First layer of feature extraction
  Hidden 2    | 128 neurons, ReLU activation | Second layer of feature extraction
  Output      | 2 neurons (one per action)   | Outputs Q-value for each action
"""

import torch
import torch.nn as nn
from typing import List


class DQN(nn.Module):
    """
    Deep Q-Network for CartPole-v1.

    The network takes a 4-dimensional state as input and outputs
    Q-values for each of the 2 possible actions. The agent selects
    the action with the highest Q-value.

    Architecture is configurable to support ablation studies on
    network depth (1, 2, 3 hidden layers).

    Args:
        state_dim: Dimension of the state space (4 for CartPole)
        action_dim: Number of possible actions (2 for CartPole)
        hidden_size: Number of neurons per hidden layer (128)
        num_hidden_layers: Number of hidden layers (2)
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        hidden_size: int = 128,
        num_hidden_layers: int = 2,
    ):
        super(DQN, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        # Build layers dynamically to support ablation on network depth
        layers: List[nn.Module] = []

        # Input layer → first hidden layer
        layers.append(nn.Linear(state_dim, hidden_size))
        layers.append(nn.ReLU())

        # Additional hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Output layer (Q-values for each action)
        layers.append(nn.Linear(hidden_size, action_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights using Xavier uniform for stable training
        self._initialize_weights()

    def _initialize_weights(self):
        """Apply Xavier uniform initialization to all linear layers."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: state → Q-values.

        Args:
            state: Batch of states, shape (batch_size, state_dim)

        Returns:
            Q-values for each action, shape (batch_size, action_dim)
        """
        return self.network(state)

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"DQN(\n"
            f"  state_dim={self.state_dim},\n"
            f"  action_dim={self.action_dim},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  num_hidden_layers={self.num_hidden_layers},\n"
            f"  parameters={self.get_num_parameters():,}\n"
            f")\n{self.network}"
        )


if __name__ == "__main__":
    # Smoke test
    model = DQN(state_dim=4, action_dim=2, hidden_size=128, num_hidden_layers=2)
    print(model)

    # Test forward pass
    dummy_state = torch.randn(1, 4)
    q_values = model(dummy_state)
    print(f"\nSingle input:  {dummy_state.shape} → {q_values.shape}")
    print(f"Q-values: {q_values.detach().numpy()}")

    # Test batch
    batch = torch.randn(64, 4)
    batch_q = model(batch)
    print(f"Batch input:   {batch.shape} → {batch_q.shape}")

    # Ablation preview: different depths
    print("\n--- Network Depth Ablation ---")
    for depth in [1, 2, 3]:
        m = DQN(num_hidden_layers=depth)
        print(f"  {depth} hidden layers → {m.get_num_parameters():,} parameters")

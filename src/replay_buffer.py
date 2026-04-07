"""
Experience Replay Buffer for DQN.

Stores transitions (state, action, reward, next_state, done) in a circular
buffer and provides random mini-batch sampling to break temporal correlations
during training.

From the proposal:
  "A replay buffer — a memory bank that stores the last 10,000 transitions.
   During each training step, we sample a random batch of 64 transitions
   from this buffer. Because they come from different points in time, the
   correlations are broken and the network gets a much more stable learning
   signal."
"""

import random
import numpy as np
import torch
from collections import deque
from typing import Tuple, NamedTuple


class Transition(NamedTuple):
    """A single experience transition."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Fixed-size circular buffer to store experience transitions.

    Uses collections.deque for O(1) append and automatic eviction of
    oldest transitions when capacity is reached.

    Args:
        capacity: Maximum number of transitions to store (default: 10,000)
        seed: Random seed for reproducible sampling
    """

    def __init__(self, capacity: int = 10000, seed: int = 42):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        random.seed(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a transition in the buffer.

        Args:
            state: Current state observation (4,)
            action: Action taken (0 or 1)
            reward: Reward received
            next_state: Resulting state observation (4,)
            done: Whether the episode ended
        """
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int = 64) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random mini-batch of transitions.

        Returns tensors ready for PyTorch training:
            states:      (batch_size, state_dim)
            actions:     (batch_size, 1)
            rewards:     (batch_size, 1)
            next_states: (batch_size, state_dim)
            dones:       (batch_size, 1)

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) tensors

        Raises:
            ValueError: If buffer has fewer transitions than batch_size
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Not enough transitions in buffer ({len(self.buffer)}) "
                f"to sample batch of {batch_size}"
            )

        transitions = random.sample(self.buffer, batch_size)

        # Unzip the batch of transitions
        batch = Transition(*zip(*transitions))

        # Convert to PyTorch tensors
        states = torch.FloatTensor(np.array(batch.state))
        actions = torch.LongTensor(np.array(batch.action)).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(batch.reward)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(batch.next_state))
        dones = torch.FloatTensor(np.array(batch.done, dtype=np.float32)).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def is_ready(self, batch_size: int = 64) -> bool:
        """Check if buffer has enough transitions for sampling."""
        return len(self.buffer) >= batch_size

    def __len__(self) -> int:
        """Return current number of transitions stored."""
        return len(self.buffer)

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(size={len(self.buffer)}, "
            f"capacity={self.capacity})"
        )


if __name__ == "__main__":
    # Quick smoke test
    buffer = ReplayBuffer(capacity=100, seed=42)
    print(f"Empty buffer: {buffer}")

    # Push some transitions
    for i in range(50):
        state = np.random.randn(4).astype(np.float32)
        action = random.randint(0, 1)
        reward = 1.0
        next_state = np.random.randn(4).astype(np.float32)
        done = i == 49
        buffer.push(state, action, reward, next_state, done)

    print(f"After 50 pushes: {buffer}")
    print(f"Ready for batch of 64? {buffer.is_ready(64)}")
    print(f"Ready for batch of 32? {buffer.is_ready(32)}")

    # Sample a batch
    states, actions, rewards, next_states, dones = buffer.sample(16)
    print(f"\nSampled batch shapes:")
    print(f"  states:      {states.shape}")
    print(f"  actions:     {actions.shape}")
    print(f"  rewards:     {rewards.shape}")
    print(f"  next_states: {next_states.shape}")
    print(f"  dones:       {dones.shape}")

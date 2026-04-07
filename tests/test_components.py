"""
Unit tests for AgentForge components.
"""

import pytest
import numpy as np
import torch

from src.replay_buffer import ReplayBuffer


class TestReplayBuffer:
    """Tests for Experience Replay Buffer."""

    def test_initialization(self):
        buffer = ReplayBuffer(capacity=100)
        assert len(buffer) == 0
        assert buffer.capacity == 100

    def test_push_single(self):
        buffer = ReplayBuffer(capacity=100)
        state = np.zeros(4, dtype=np.float32)
        buffer.push(state, 0, 1.0, state, False)
        assert len(buffer) == 1

    def test_push_multiple(self):
        buffer = ReplayBuffer(capacity=100)
        state = np.zeros(4, dtype=np.float32)
        for _ in range(25):
            buffer.push(state, 0, 1.0, state, False)
        assert len(buffer) == 25

    def test_capacity_limit(self):
        """Buffer should never exceed capacity — oldest evicted first."""
        buffer = ReplayBuffer(capacity=10)
        state = np.zeros(4, dtype=np.float32)
        for i in range(20):
            buffer.push(state, 0, float(i), state, False)
        assert len(buffer) == 10

    def test_sample_shapes(self):
        """Sampled tensors must have correct shapes for DQN training."""
        buffer = ReplayBuffer(capacity=100)
        for _ in range(50):
            state = np.random.randn(4).astype(np.float32)
            buffer.push(state, np.random.randint(2), 1.0, state, False)

        states, actions, rewards, next_states, dones = buffer.sample(16)

        assert states.shape == (16, 4)
        assert actions.shape == (16, 1)
        assert rewards.shape == (16, 1)
        assert next_states.shape == (16, 4)
        assert dones.shape == (16, 1)

    def test_sample_dtypes(self):
        """Tensor dtypes must match what PyTorch expects."""
        buffer = ReplayBuffer(capacity=100)
        for _ in range(50):
            state = np.random.randn(4).astype(np.float32)
            buffer.push(state, 0, 1.0, state, False)

        states, actions, rewards, next_states, dones = buffer.sample(16)

        assert states.dtype == torch.float32
        assert actions.dtype == torch.int64
        assert rewards.dtype == torch.float32
        assert next_states.dtype == torch.float32
        assert dones.dtype == torch.float32

    def test_is_ready_false(self):
        buffer = ReplayBuffer(capacity=100)
        assert not buffer.is_ready(64)

    def test_is_ready_true(self):
        buffer = ReplayBuffer(capacity=100)
        for _ in range(64):
            state = np.zeros(4, dtype=np.float32)
            buffer.push(state, 0, 1.0, state, False)
        assert buffer.is_ready(64)

    def test_sample_raises_on_insufficient_data(self):
        """Should raise ValueError if not enough transitions."""
        buffer = ReplayBuffer(capacity=100)
        for _ in range(5):
            state = np.zeros(4, dtype=np.float32)
            buffer.push(state, 0, 1.0, state, False)

        with pytest.raises(ValueError):
            buffer.sample(10)

    def test_done_flag_preserved(self):
        """Done=True transitions should have dones=1.0 in tensor."""
        buffer = ReplayBuffer(capacity=100)
        state = np.zeros(4, dtype=np.float32)

        # Push 10 non-terminal and 1 terminal
        for _ in range(10):
            buffer.push(state, 0, 1.0, state, False)
        buffer.push(state, 0, 0.0, state, True)

        # Sample all 11
        _, _, _, _, dones = buffer.sample(11)

        # At least one done should be 1.0
        assert dones.sum().item() >= 1.0

    def test_repr(self):
        buffer = ReplayBuffer(capacity=500)
        assert "500" in repr(buffer)
        assert "0" in repr(buffer)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit tests for AgentForge components.
"""

import pytest
import numpy as np
import torch

from src.replay_buffer import ReplayBuffer
from src.model import DQN
from src.agent import DQNAgent


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


class TestDQN:
    """Tests for DQN Neural Network."""

    def test_default_architecture(self):
        model = DQN()
        assert model.state_dim == 4
        assert model.action_dim == 2
        assert model.hidden_size == 128
        assert model.num_hidden_layers == 2

    def test_forward_single_state(self):
        model = DQN(state_dim=4, action_dim=2)
        state = torch.randn(1, 4)
        output = model(state)
        assert output.shape == (1, 2)

    def test_forward_batch(self):
        """Batch of 64 states (matching proposal batch_size)."""
        model = DQN()
        batch = torch.randn(64, 4)
        output = model(batch)
        assert output.shape == (64, 2)

    def test_parameter_count(self):
        """Verify expected param count for 4→128→128→2."""
        model = DQN(hidden_size=128, num_hidden_layers=2)
        params = model.get_num_parameters()
        # 4*128+128 + 128*128+128 + 128*2+2 = 512+128+16384+128+256+2 = 17410
        assert params == 17410

    def test_variable_depth(self):
        """Ablation support: different depths must all produce valid output."""
        for depth in [1, 2, 3]:
            model = DQN(num_hidden_layers=depth)
            output = model(torch.randn(1, 4))
            assert output.shape == (1, 2)

    def test_deeper_network_has_more_params(self):
        m1 = DQN(num_hidden_layers=1)
        m2 = DQN(num_hidden_layers=2)
        m3 = DQN(num_hidden_layers=3)
        assert m1.get_num_parameters() < m2.get_num_parameters() < m3.get_num_parameters()

    def test_output_not_all_zeros(self):
        """Xavier init should produce non-trivial first outputs."""
        model = DQN()
        output = model(torch.randn(10, 4))
        assert not torch.all(output == 0)

    def test_gradient_flows(self):
        """Ensure gradients propagate through the full network."""
        model = DQN()
        state = torch.randn(1, 4, requires_grad=True)
        output = model(state)
        loss = output.sum()
        loss.backward()
        assert state.grad is not None


class TestDQNAgent:
    """Tests for DQN Agent."""

    def test_initialization(self):
        agent = DQNAgent()
        assert agent.epsilon == 1.0
        assert agent.total_steps == 0
        assert len(agent.replay_buffer) == 0

    def test_action_in_valid_range(self):
        agent = DQNAgent()
        state = np.random.randn(4).astype(np.float32)
        action = agent.select_action(state)
        assert action in [0, 1]

    def test_exploration_at_high_epsilon(self):
        """With ε=1.0, should take diverse random actions."""
        agent = DQNAgent(epsilon_start=1.0)
        state = np.zeros(4, dtype=np.float32)
        actions = [agent.select_action(state) for _ in range(100)]
        assert 0 in actions and 1 in actions

    def test_exploitation_at_zero_epsilon(self):
        """With ε=0.0, should be deterministic."""
        agent = DQNAgent(epsilon_start=0.0, epsilon_end=0.0)
        state = np.zeros(4, dtype=np.float32)
        actions = [agent.select_action(state) for _ in range(10)]
        assert len(set(actions)) == 1

    def test_epsilon_decay(self):
        agent = DQNAgent(epsilon_start=1.0, epsilon_decay=0.995)
        agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.995)

    def test_epsilon_floor(self):
        agent = DQNAgent(epsilon_start=0.02, epsilon_end=0.01, epsilon_decay=0.5)
        agent.decay_epsilon()
        assert agent.epsilon >= 0.01

    def test_store_transition(self):
        agent = DQNAgent()
        state = np.zeros(4, dtype=np.float32)
        agent.store_transition(state, 0, 1.0, state, False)
        assert len(agent.replay_buffer) == 1

    def test_optimize_returns_none_before_ready(self):
        agent = DQNAgent(batch_size=64)
        assert agent.optimize() is None

    def test_optimize_returns_loss_when_ready(self):
        agent = DQNAgent(batch_size=16)
        for _ in range(20):
            s = np.random.randn(4).astype(np.float32)
            agent.store_transition(s, 0, 1.0, s, False)
        loss = agent.optimize()
        assert loss is not None
        assert isinstance(loss, float)
        assert loss >= 0

    def test_save_and_load(self, tmp_path):
        agent = DQNAgent()
        for _ in range(100):
            s = np.random.randn(4).astype(np.float32)
            agent.store_transition(s, 0, 1.0, s, False)
        agent.optimize()
        agent.decay_epsilon()

        path = str(tmp_path / "test_agent.pt")
        agent.save(path)

        new_agent = DQNAgent()
        new_agent.load(path)
        assert new_agent.epsilon == agent.epsilon
        assert new_agent.total_steps == agent.total_steps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


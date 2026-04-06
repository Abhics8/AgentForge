"""
Environment wrapper for CartPole-v1.

Provides a clean interface around Gymnasium's CartPole-v1 environment
with optional video recording and seed management.

CartPole-v1 Specifications (from proposal):
  - State Space:  4-dimensional continuous
      [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
  - Action Space: 2 discrete actions (0 = push left, 1 = push right)
  - Reward:       +1 for every timestep the pole remains upright
  - Termination:  Pole angle > ±12°, cart position > ±2.4, or 500 steps
  - Solved:       Average reward >= 195 over 100 consecutive episodes
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Tuple


class CartPoleEnvironment:
    """
    Wrapper around Gymnasium CartPole-v1 environment.

    Provides:
      - Consistent seeding for reproducibility
      - Clean reset/step interface returning numpy arrays
      - Optional video recording via Gymnasium wrappers
      - Environment metadata (state dim, action dim)
    """

    def __init__(
        self,
        env_name: str = "CartPole-v1",
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        record_video: bool = False,
        video_dir: str = "results/videos",
        video_episode_trigger: Optional[callable] = None,
    ):
        """
        Initialize the CartPole environment.

        Args:
            env_name: Gymnasium environment ID
            seed: Random seed for reproducibility
            render_mode: 'human' for live rendering, 'rgb_array' for video recording
            record_video: If True, wrap environment with RecordVideo
            video_dir: Directory to save recorded videos
            video_episode_trigger: Function(episode_id) -> bool, controls which episodes to record
        """
        self.env_name = env_name
        self.seed = seed

        # Determine render mode
        if record_video and render_mode is None:
            render_mode = "rgb_array"

        # Create the base environment
        self.env = gym.make(env_name, render_mode=render_mode)

        # Wrap with video recorder if requested
        if record_video:
            if video_episode_trigger is None:
                video_episode_trigger = lambda ep: ep % 100 == 0

            self.env = gym.wrappers.RecordVideo(
                self.env,
                video_folder=video_dir,
                episode_trigger=video_episode_trigger,
                name_prefix="agentforge",
            )

        # Environment metadata
        self.state_dim = self.env.observation_space.shape[0]  # 4
        self.action_dim = self.env.action_space.n  # 2

    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial state."""
        state, info = self.env.reset(seed=self.seed)
        return np.array(state, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Take an action in the environment.

        Args:
            action: 0 (push left) or 1 (push right)

        Returns:
            next_state, reward, terminated, truncated, info
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        return (
            np.array(next_state, dtype=np.float32),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def sample_random_action(self) -> int:
        """Sample a random action from the action space."""
        return self.env.action_space.sample()

    def close(self):
        """Clean up environment resources."""
        self.env.close()

    def __repr__(self) -> str:
        return (
            f"CartPoleEnvironment("
            f"env={self.env_name}, "
            f"state_dim={self.state_dim}, "
            f"action_dim={self.action_dim})"
        )


def make_env(
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    record_video: bool = False,
    video_dir: str = "results/videos",
    video_episode_trigger: Optional[callable] = None,
) -> CartPoleEnvironment:
    """Factory function to create a CartPole environment."""
    return CartPoleEnvironment(
        seed=seed,
        render_mode=render_mode,
        record_video=record_video,
        video_dir=video_dir,
        video_episode_trigger=video_episode_trigger,
    )


if __name__ == "__main__":
    # Quick smoke test
    env = make_env(seed=42)
    print(f"Environment: {env}")
    print(f"State dimensions: {env.state_dim}")
    print(f"Action dimensions: {env.action_dim}")

    state = env.reset()
    print(f"Initial state: {state}")

    total_reward = 0
    done = False
    steps = 0

    while not done:
        action = env.sample_random_action()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    print(f"Random agent survived {steps} steps with reward {total_reward}")
    env.close()

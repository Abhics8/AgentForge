import numpy as np

class RandomAgent:
    """
    Uniform random baseline agent.
    Selects actions randomly from the action space.
    """
    def __init__(self, action_dim=2):
        self.action_dim = action_dim

    def select_action(self, state):
        return np.random.randint(self.action_dim)

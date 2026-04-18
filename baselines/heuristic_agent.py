import numpy as np

class HeuristicAgent:
    """
    Rule-based baseline agent.
    If the pole leans to the right (angle > 0), push the cart right.
    If the pole leans to the left (angle < 0), push the cart left.
    """
    def __init__(self):
        pass

    def select_action(self, state):
        # State array: [cart_pos, cart_vel, pole_angle, pole_velocity]
        pole_angle = state[2]
        if pole_angle > 0:
            return 1  # push right
        else:
            return 0  # push left

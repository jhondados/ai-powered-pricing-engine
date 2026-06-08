"""RL-based price optimization agent."""
import numpy as np
from typing import Tuple

class PricingRLAgent:
    """Q-learning agent for dynamic price optimization."""
    def __init__(self, price_points: list, learning_rate: float = 0.1, discount: float = 0.95,
                 epsilon: float = 0.1):
        self.price_points = price_points
        self.n_actions = len(price_points)
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.q_table = {}  # state -> [Q-values per price point]

    def state_to_key(self, demand_level: float, competitor_ratio: float,
                     inventory_level: float) -> str:
        d = int(demand_level * 5)  # discretize 0-1 into 5 buckets
        c = int(competitor_ratio * 5)
        i = int(inventory_level * 5)
        return f"{d}_{c}_{i}"

    def select_price(self, state_key: str) -> Tuple[float, int]:
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        if np.random.random() < self.epsilon:  # explore
            action = np.random.randint(self.n_actions)
        else:  # exploit
            action = int(np.argmax(self.q_table[state_key]))
        return self.price_points[action], action

    def update(self, state_key: str, action: int, reward: float, next_state_key: str):
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.n_actions)
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = new_q

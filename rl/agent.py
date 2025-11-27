from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


State = Tuple[int, ...]
Action = int


@dataclass
class QLearningAgent:
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 0.2
    min_epsilon: float = 0.05
    epsilon_decay: float = 0.995

    def __post_init__(self) -> None:
        self.q_table: Dict[Tuple[State, Action], float] = defaultdict(float)

    def select_action(self, state: State, actions: Iterable[Action], rng) -> Action:
        actions = list(actions)
        if not actions:
            raise ValueError("No available actions")

        if rng.random() < self.epsilon:
            return rng.choice(actions)

        values = [self.q_table[(state, action)] for action in actions]
        max_value = max(values)
        best_actions = [action for action, value in zip(actions, values) if value == max_value]
        # print(f"Best actions: {best_actions}")
        return rng.choice(best_actions)

    def update(self, state: State, action: Action, reward: float, next_state: State, next_actions: Iterable[Action]) -> None:
        next_actions = list(next_actions)
        next_max = 0.0
        if next_actions:
            next_max = max(self.q_table[(next_state, a)] for a in next_actions)

        old_value = self.q_table[(state, action)]
        td_target = reward + self.gamma * next_max
        self.q_table[(state, action)] = old_value + self.alpha * (td_target - old_value)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

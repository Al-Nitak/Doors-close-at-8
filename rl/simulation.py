from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, List

from .agent import QLearningAgent
from .environment import LudoEnvironment
from .types import StepInfo


@dataclass
class EpisodeStats:
    episode: int
    total_reward: float
    turns: int
    trace: List[StepInfo]


@dataclass
class EvaluationStats:
    run: int
    total_reward: float
    turns: int


@dataclass
class SimulationResult:
    training_history: List[EpisodeStats]
    evaluation: List[EvaluationStats]

    @property
    def avg_training_reward(self) -> float:
        return mean(stat.total_reward for stat in self.training_history)

    @property
    def avg_training_turns(self) -> float:
        return mean(stat.turns for stat in self.training_history)

    @property
    def avg_eval_reward(self) -> float:
        return mean(stat.total_reward for stat in self.evaluation)

    @property
    def avg_eval_turns(self) -> float:
        return mean(stat.turns for stat in self.evaluation)


def run_simulation(config: Dict) -> SimulationResult:
    env = LudoEnvironment(
        board_size=config.get("board_size", 32),
        num_players=config.get("num_players", 4),
        num_pawns=config.get("num_pawns", 4),
        dice_sides=config.get("dice_sides", 6),
        power_squares=config.get("power_squares", []),
        power_bonus=config.get("power_bonus", 1),
        shortcut_squares=config.get("shortcut_squares", {}),
        construction_zones=config.get("construction_zones", {}),
        max_turns=config.get("max_turns", 500),
        seed=config.get("seed"),
        allow_odd_split=config.get("allow_odd_split", False),
        enable_capture=config.get("enable_capture", True),
    )

    agent = QLearningAgent(
        alpha=config.get("alpha", 0.1),
        gamma=config.get("gamma", 0.95),
        epsilon=config.get("epsilon", 0.2),
        min_epsilon=config.get("min_epsilon", 0.05),
        epsilon_decay=config.get("epsilon_decay", 0.995),
    )

    episodes = config.get("episodes", 200)
    training_history: List[EpisodeStats] = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        turns = 0
        episode_trace: List[StepInfo] = []

        while not done:
            options = env.available_actions()
            if not options:
                break
            action_keys = [option.key for option in options]
            chosen_key = agent.select_action(state, action_keys, env.rng)
            option_lookup = {option.key: option for option in options}
            selected_option = option_lookup[chosen_key]
            next_state, reward, done, info = env.step(selected_option)
            total_reward += reward
            turns += 1
            episode_trace.append(info)

            next_options = [] if done else env.available_actions()
            next_keys = [option.key for option in next_options]
            agent.update(state, chosen_key, reward, next_state, next_keys)
            state = next_state

        agent.decay_epsilon()
        training_history.append(
            EpisodeStats(
                episode=episode,
                total_reward=total_reward,
                turns=turns,
                trace=episode_trace,
            )
        )

    evaluation_runs = config.get("evaluation_runs", 50)
    evaluation: List[EvaluationStats] = []

    # Use near-greedy policy for evaluation
    agent.epsilon = agent.min_epsilon

    for run in range(1, evaluation_runs + 1):
        eval_env = env.copy()
        state = eval_env.reset()
        done = False
        total_reward = 0.0
        turns = 0

        while not done:
            options = eval_env.available_actions()
            if not options:
                break
            action_keys = [option.key for option in options]
            chosen_key = agent.select_action(state, action_keys, eval_env.rng)
            option_lookup = {option.key: option for option in options}
            selected_option = option_lookup[chosen_key]
            next_state, reward, done, _ = eval_env.step(selected_option)
            total_reward += reward
            turns += 1
            state = next_state

        evaluation.append(EvaluationStats(run=run, total_reward=total_reward, turns=turns))

    return SimulationResult(training_history=training_history, evaluation=evaluation)

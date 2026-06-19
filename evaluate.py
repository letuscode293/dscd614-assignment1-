"""
evaluate.py
===========

Part E - Evaluation.

Evaluates a trained Q-Learning agent (greedy policy) over a number of episodes
and reports:

* Success Rate (%)
* Average Reward
* Number of Failures
* Number of Successful Runs

The trained Q-table is loaded from ``results/q_table.npy`` by default, or a
freshly trained agent can be passed in programmatically via :func:`evaluate`.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np

from agent import QLearningAgent
from environment import FrozenLakeEnv

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


@dataclass
class EvaluationResult:
    episodes: int
    successes: int
    failures: int
    average_reward: float

    @property
    def success_rate(self) -> float:
        return 100.0 * self.successes / self.episodes if self.episodes else 0.0

    def report(self) -> str:
        return (
            "Evaluation Results\n"
            "==================\n"
            f"Episodes Evaluated   : {self.episodes}\n"
            f"Success Rate (%)     : {self.success_rate:.2f}\n"
            f"Average Reward       : {self.average_reward:.4f}\n"
            f"Number of Failures   : {self.failures}\n"
            f"Number of Successes  : {self.successes}\n"
        )


def evaluate(
    env: FrozenLakeEnv,
    agent: QLearningAgent,
    episodes: int = 100,
    max_steps: int = 200,
) -> EvaluationResult:
    """Run ``episodes`` greedy episodes and collect outcome statistics."""
    successes = 0
    total_reward = 0.0

    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            action = agent.choose_action(state, greedy=True)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
            if done:
                if next_state in env.goal_states:
                    successes += 1
                break

    failures = episodes - successes
    avg_reward = total_reward / episodes if episodes else 0.0
    return EvaluationResult(episodes, successes, failures, avg_reward)


def load_agent(env: FrozenLakeEnv, q_table_path: str, seed: int | None = None) -> QLearningAgent:
    """Create an agent and load a previously saved Q-table."""
    agent = QLearningAgent(env.n_states, env.n_actions, seed=seed)
    agent.q_table = np.load(q_table_path)
    if agent.q_table.shape != (env.n_states, env.n_actions):
        raise ValueError("Loaded Q-table shape does not match the environment.")
    return agent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained Q-Learning agent.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--q-table", type=str, default=os.path.join(RESULTS_DIR, "q_table.npy"))
    parser.add_argument("--slippery", action="store_true", help="Evaluate with stochastic transitions")
    parser.add_argument("--slip-prob", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if not os.path.exists(args.q_table):
        raise SystemExit(
            f"Q-table not found at '{args.q_table}'. Run train.py first."
        )

    env = FrozenLakeEnv(is_slippery=args.slippery, slip_prob=args.slip_prob, seed=args.seed)
    agent = load_agent(env, args.q_table, seed=args.seed)

    result = evaluate(env, agent, episodes=args.episodes, max_steps=args.max_steps)
    print(result.report())


if __name__ == "__main__":
    main()

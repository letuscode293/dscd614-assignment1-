"""
train.py
========

Part C - Training.

Runs the Q-Learning training loop on the Frozen Lake environment and records
training statistics:

* Episode rewards
* Success rate (rolling)
* Number of successful episodes
* Epsilon value over time

The module exposes a reusable ``train`` function and a command-line entry point.
Running ``python train.py`` trains an agent, prints the learned policy, saves
the Q-table to ``results/q_table.npy`` and the statistics to
``results/training_stats.npz``, and (if matplotlib is available) generates the
training graphs via ``visualize.py``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np

from agent import QLearningAgent
from environment import FrozenLakeEnv
from policy import print_policy

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


@dataclass
class TrainingStats:
    """Container for per-episode training statistics."""

    episode_rewards: List[float] = field(default_factory=list)
    episode_successes: List[int] = field(default_factory=list)  # 1 if goal reached
    episode_steps: List[int] = field(default_factory=list)
    epsilon_history: List[float] = field(default_factory=list)

    @property
    def num_successful_episodes(self) -> int:
        return int(np.sum(self.episode_successes))

    def rolling_success_rate(self, window: int = 100) -> np.ndarray:
        successes = np.asarray(self.episode_successes, dtype=np.float64)
        if len(successes) == 0:
            return successes
        kernel = np.ones(window) / window
        return np.convolve(successes, kernel, mode="valid")


def train(
    env: FrozenLakeEnv,
    agent: QLearningAgent,
    episodes: int = 20000,
    max_steps: int = 200,
    verbose: bool = True,
    log_every: int = 2000,
) -> TrainingStats:
    """Train ``agent`` on ``env`` for a number of episodes.

    Returns the collected :class:`TrainingStats`.
    """
    stats = TrainingStats()

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        reached_goal = 0
        steps = 0

        for _ in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                # Goal tiles carry a positive reward; holes do not.
                reached_goal = 1 if next_state in env.goal_states else 0
                break

        agent.decay()

        stats.episode_rewards.append(total_reward)
        stats.episode_successes.append(reached_goal)
        stats.episode_steps.append(steps)
        stats.epsilon_history.append(agent.epsilon)

        if verbose and episode % log_every == 0:
            window = min(log_every, len(stats.episode_successes))
            recent = np.mean(stats.episode_successes[-window:]) * 100
            print(
                f"Episode {episode:>6}/{episodes} | "
                f"epsilon={agent.epsilon:.3f} | "
                f"recent success rate ({window} eps)={recent:5.1f}% | "
                f"total successes={stats.num_successful_episodes}"
            )

    return stats


def save_results(agent: QLearningAgent, stats: TrainingStats) -> None:
    """Persist the Q-table and training statistics to the results directory."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(os.path.join(RESULTS_DIR, "q_table.npy"), agent.q_table)
    np.savez(
        os.path.join(RESULTS_DIR, "training_stats.npz"),
        episode_rewards=np.asarray(stats.episode_rewards),
        episode_successes=np.asarray(stats.episode_successes),
        episode_steps=np.asarray(stats.episode_steps),
        epsilon_history=np.asarray(stats.epsilon_history),
    )
    print(f"Saved Q-table and training statistics to '{RESULTS_DIR}'.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Q-Learning agent on Frozen Lake.")
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--slippery", action="store_true", help="Enable stochastic transitions (Bonus A)")
    parser.add_argument("--slip-prob", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true", help="Skip generating graphs")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    env = FrozenLakeEnv(
        is_slippery=args.slippery,
        slip_prob=args.slip_prob,
        seed=args.seed,
    )
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        decay_epsilon=True,
        seed=args.seed,
    )

    print("Training configuration")
    print("======================")
    print(f"episodes      = {args.episodes}")
    print(f"alpha (lr)    = {args.alpha}")
    print(f"gamma         = {args.gamma}")
    print(f"epsilon       = {args.epsilon} -> {args.epsilon_min} (decay {args.epsilon_decay})")
    print(f"slippery      = {args.slippery} (slip_prob={args.slip_prob})")
    print()

    stats = train(env, agent, episodes=args.episodes, max_steps=args.max_steps)
    print()
    print(f"Total successful episodes during training: {stats.num_successful_episodes}")
    print()

    policy = agent.extract_policy()
    print_policy(env, policy)

    save_results(agent, stats)

    if not args.no_plot:
        try:
            from visualize import plot_training_stats

            plot_training_stats(stats, out_dir=RESULTS_DIR)
        except Exception as exc:  # pragma: no cover - plotting is optional
            print(f"(Skipped plotting: {exc})")


if __name__ == "__main__":
    main()

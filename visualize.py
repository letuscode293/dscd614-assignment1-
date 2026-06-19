"""
visualize.py
============

Bonus Option B - Visualisation of training performance.

Generates graphs from the training statistics:

* Episode reward (with moving average)
* Rolling success rate
* Epsilon decay over time

All figures are written to the ``results/`` directory as PNG files.
"""

from __future__ import annotations

import os

import numpy as np

# Use a non-interactive backend so the script works on headless machines.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        window = max(1, len(values))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training_stats(stats, out_dir: str, window: int = 100) -> None:
    """Create and save the standard set of training graphs."""
    os.makedirs(out_dir, exist_ok=True)

    rewards = np.asarray(stats.episode_rewards, dtype=np.float64)
    successes = np.asarray(stats.episode_successes, dtype=np.float64)
    epsilons = np.asarray(stats.epsilon_history, dtype=np.float64)
    episodes = np.arange(1, len(rewards) + 1)

    # 1. Episode reward + moving average.
    plt.figure(figsize=(9, 5))
    plt.plot(episodes, rewards, color="#cccccc", linewidth=0.6, label="Episode reward")
    if len(rewards) >= window:
        ma = _moving_average(rewards, window)
        plt.plot(np.arange(window, len(rewards) + 1), ma, color="#1f77b4",
                 linewidth=2, label=f"{window}-episode moving avg")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Episode Reward over Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "episode_rewards.png"), dpi=120)
    plt.close()

    # 2. Rolling success rate.
    plt.figure(figsize=(9, 5))
    if len(successes) >= window:
        rate = _moving_average(successes, window) * 100
        plt.plot(np.arange(window, len(successes) + 1), rate, color="#2ca02c", linewidth=2)
    else:
        plt.plot(episodes, successes * 100, color="#2ca02c", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Success rate (%)")
    plt.title(f"Rolling Success Rate ({window}-episode window)")
    plt.ylim(-5, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "success_rate.png"), dpi=120)
    plt.close()

    # 3. Epsilon decay.
    plt.figure(figsize=(9, 5))
    plt.plot(episodes, epsilons, color="#d62728", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Exploration Rate (Epsilon) over Training")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "epsilon_decay.png"), dpi=120)
    plt.close()

    print(f"Saved training graphs to '{out_dir}'.")


def plot_value_heatmap(env, q_table: np.ndarray, out_dir: str) -> None:
    """Save a heatmap of the state-value function V(s) = max_a Q(s, a)."""
    os.makedirs(out_dir, exist_ok=True)
    values = q_table.max(axis=1).reshape(env.n_rows, env.n_cols)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(values, cmap="viridis")
    plt.colorbar(im, label="V(s) = max_a Q(s, a)")
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            tile = env.grid[r][c]
            plt.text(c, r, tile, ha="center", va="center",
                     color="white", fontsize=10, fontweight="bold")
    plt.title("Learned State-Value Function")
    plt.xticks(range(env.n_cols))
    plt.yticks(range(env.n_rows))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "value_heatmap.png"), dpi=120)
    plt.close()
    print(f"Saved value heatmap to '{out_dir}'.")

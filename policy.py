"""
policy.py
=========

Utilities for Part D - Policy Extraction.

Given a trained Q-table (or an array of greedy actions) and the environment,
these helpers produce a human-readable grid showing the recommended action for
every non-terminal state, with holes (H) and the goal (G) clearly marked.
"""

from __future__ import annotations

import numpy as np

from environment import ACTION_ARROWS, FrozenLakeEnv


def policy_grid(env: FrozenLakeEnv, policy: np.ndarray) -> str:
    """Return a string visualising the policy on the grid.

    Each non-terminal Frozen/Start tile shows the recommended action arrow.
    Holes are shown as ``H`` and the goal as ``G``.
    """
    lines = []
    for r in range(env.n_rows):
        cells = []
        for c in range(env.n_cols):
            state = env.coords_to_state(r, c)
            tile = env.tile_at(state)
            if tile == "H":
                cells.append("H")
            elif tile == "G":
                cells.append("G")
            else:
                cells.append(ACTION_ARROWS[int(policy[state])])
        lines.append(" ".join(cells))
    return "\n".join(lines)


def print_policy(env: FrozenLakeEnv, policy: np.ndarray, title: str = "Learned Policy") -> None:
    """Pretty-print the policy grid with a title and legend."""
    print(title)
    print("-" * len(title))
    print(policy_grid(env, policy))
    print()
    print("Legend:  \u2190 Left   \u2193 Down   \u2192 Right   \u2191 Up   H Hole   G Goal")
    print()

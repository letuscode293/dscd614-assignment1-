"""
environment.py
==============

A custom implementation of the Frozen Lake grid-world environment, written
entirely from first principles (no Gymnasium / OpenAI Gym / RLlib / etc.).

The agent must navigate an 8x8 grid from the Start (S) to the Goal (G) while
avoiding Holes (H) and staying on Frozen (F) tiles.

Tile legend
-----------
S : Start state
F : Frozen (safe) state
H : Hole (terminal state, reward 0)
G : Goal (terminal state, reward +1)

Actions
-------
0 = Left, 1 = Down, 2 = Right, 3 = Up

State representation
--------------------
States are represented as single integer indices in [0, n_rows * n_cols).
The index for grid cell (row, col) is `row * n_cols + col`. Helper methods
``state_to_coords`` and ``coords_to_state`` convert between the two views.
"""

from __future__ import annotations

import random
from typing import List, Tuple


# The standard 8x8 Frozen Lake map used in this assignment.
DEFAULT_MAP: List[str] = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]

# Action constants for readability.
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

# Mapping from action -> human readable arrow used for rendering policies.
ACTION_ARROWS = {LEFT: "\u2190", DOWN: "\u2193", RIGHT: "\u2192", UP: "\u2191"}


class FrozenLakeEnv:
    """A from-scratch Frozen Lake environment.

    Parameters
    ----------
    grid_map:
        List of equal-length strings describing the grid. Defaults to the
        standard 8x8 assignment map.
    is_slippery:
        If True, transitions become stochastic (Bonus Option A). With
        probability ``slip_prob`` the intended action is replaced by one of
        the two perpendicular actions (each with half of ``slip_prob``).
    slip_prob:
        Total probability that the agent slips when ``is_slippery`` is True.
    step_reward:
        Reward returned for a normal, non-terminal step.
    hole_reward:
        Reward returned when the agent falls into a hole.
    goal_reward:
        Reward returned when the agent reaches the goal.
    seed:
        Optional seed for reproducible stochastic behaviour.
    """

    def __init__(
        self,
        grid_map: List[str] | None = None,
        is_slippery: bool = False,
        slip_prob: float = 0.2,
        step_reward: float = 0.0,
        hole_reward: float = 0.0,
        goal_reward: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.grid = list(grid_map) if grid_map is not None else list(DEFAULT_MAP)
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])

        # Validate that the grid is rectangular.
        for row in self.grid:
            if len(row) != self.n_cols:
                raise ValueError("All rows in the grid map must have equal length.")

        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4

        self.is_slippery = is_slippery
        self.slip_prob = slip_prob
        self.step_reward = step_reward
        self.hole_reward = hole_reward
        self.goal_reward = goal_reward

        self._rng = random.Random(seed)

        # Locate the start state.
        self.start_state = self._find_tile("S")

        # Cache terminal states (holes + goal).
        self.hole_states = {i for i, t in enumerate(self._flat_grid()) if t == "H"}
        self.goal_states = {i for i, t in enumerate(self._flat_grid()) if t == "G"}

        self.state = self.start_state
        self.done = False

    # ------------------------------------------------------------------
    # Coordinate / state helpers
    # ------------------------------------------------------------------
    def _flat_grid(self) -> str:
        return "".join(self.grid)

    def _find_tile(self, symbol: str) -> int:
        for i, tile in enumerate(self._flat_grid()):
            if tile == symbol:
                return i
        raise ValueError(f"Tile '{symbol}' not found in grid map.")

    def coords_to_state(self, row: int, col: int) -> int:
        return row * self.n_cols + col

    def state_to_coords(self, state: int) -> Tuple[int, int]:
        return divmod(state, self.n_cols)

    def tile_at(self, state: int) -> str:
        return self._flat_grid()[state]

    # ------------------------------------------------------------------
    # Core Gym-like API
    # ------------------------------------------------------------------
    def reset(self) -> int:
        """Reset the agent to the start state and return the initial state."""
        self.state = self.start_state
        self.done = False
        return self.state

    def get_state(self) -> int:
        """Return the current integer state index."""
        return self.state

    def is_terminal(self, state: int | None = None) -> bool:
        """Return True if ``state`` (or current state) is a hole or the goal."""
        s = self.state if state is None else state
        return s in self.hole_states or s in self.goal_states

    def _move(self, state: int, action: int) -> int:
        """Return the next state given a deterministic action (with walls)."""
        row, col = self.state_to_coords(state)
        if action == LEFT:
            col = max(col - 1, 0)
        elif action == DOWN:
            row = min(row + 1, self.n_rows - 1)
        elif action == RIGHT:
            col = min(col + 1, self.n_cols - 1)
        elif action == UP:
            row = max(row - 1, 0)
        else:
            raise ValueError(f"Invalid action: {action}")
        return self.coords_to_state(row, col)

    def _apply_slip(self, action: int) -> int:
        """Possibly replace the intended action with a perpendicular one."""
        if not self.is_slippery:
            return action
        if self._rng.random() >= self.slip_prob:
            return action
        # Perpendicular actions: for LEFT/RIGHT the slips are UP/DOWN, etc.
        if action in (LEFT, RIGHT):
            return self._rng.choice([UP, DOWN])
        return self._rng.choice([LEFT, RIGHT])

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Take an action.

        Returns
        -------
        (next_state, reward, done, info)
        """
        if self.done:
            raise RuntimeError("step() called on a terminated episode. Call reset().")

        effective_action = self._apply_slip(action)
        next_state = self._move(self.state, effective_action)

        tile = self.tile_at(next_state)
        if tile == "H":
            reward = self.hole_reward
            self.done = True
        elif tile == "G":
            reward = self.goal_reward
            self.done = True
        else:
            reward = self.step_reward
            self.done = False

        self.state = next_state
        info = {"intended_action": action, "effective_action": effective_action}
        return next_state, reward, self.done, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self, mode: str = "human") -> str:
        """Render the grid with the agent's position marked by ``*``."""
        agent_row, agent_col = self.state_to_coords(self.state)
        lines = []
        for r in range(self.n_rows):
            row_chars = []
            for c in range(self.n_cols):
                if r == agent_row and c == agent_col:
                    row_chars.append("*")
                else:
                    row_chars.append(self.grid[r][c])
            lines.append(" ".join(row_chars))
        out = "\n".join(lines)
        if mode == "human":
            print(out)
            print()
        return out


if __name__ == "__main__":
    # Quick manual sanity check.
    env = FrozenLakeEnv()
    s = env.reset()
    print(f"Start state: {s}  coords: {env.state_to_coords(s)}")
    env.render()
    for a in [RIGHT, DOWN, DOWN]:
        ns, r, done, info = env.step(a)
        print(f"action={a} -> state={ns} reward={r} done={done}")
        env.render()
        if done:
            break

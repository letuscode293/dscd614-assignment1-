"""
agent.py
========

A tabular Q-Learning agent implemented from first principles.

The agent maintains a Q-table of shape ``(n_states, n_actions)`` and learns
through the temporal-difference update:

    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

Exploration is handled with an epsilon-greedy policy with optional epsilon
decay over the course of training.
"""

from __future__ import annotations

import numpy as np


class QLearningAgent:
    """Tabular Q-Learning agent.

    Parameters
    ----------
    n_states, n_actions:
        Size of the state and action spaces.
    alpha:
        Learning rate (step size).
    gamma:
        Discount factor.
    epsilon:
        Initial exploration probability.
    epsilon_min:
        Lower bound for epsilon.
    epsilon_decay:
        Multiplicative decay applied to epsilon after every episode
        (epsilon <- max(epsilon_min, epsilon * epsilon_decay)).
    decay_epsilon:
        If False the agent uses a fixed epsilon (pure epsilon-greedy). This is
        used for the Bonus Option C comparison of exploration strategies.
    seed:
        Optional seed for reproducible action selection.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995,
        decay_epsilon: bool = True,
        seed: int | None = None,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.decay_epsilon = decay_epsilon

        self._rng = np.random.default_rng(seed)

        # Q-table initialised to zeros.
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def choose_action(self, state: int, greedy: bool = False) -> int:
        """Select an action using an epsilon-greedy policy.

        When ``greedy`` is True (used at evaluation time) the agent always
        exploits the current best action.
        """
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        return self._argmax(state)

    def _argmax(self, state: int) -> int:
        """Greedy action with random tie-breaking among equal Q-values."""
        q_values = self.q_table[state]
        best = np.flatnonzero(q_values == q_values.max())
        return int(self._rng.choice(best))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """Apply the Q-Learning temporal-difference update.

        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]

        If the transition is terminal there is no bootstrap term, so the
        target reduces to just the reward.
        """
        best_next = 0.0 if done else float(self.q_table[next_state].max())
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def decay(self) -> None:
        """Decay epsilon after an episode (no-op for pure epsilon-greedy)."""
        if self.decay_epsilon:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Policy extraction
    # ------------------------------------------------------------------
    def extract_policy(self) -> np.ndarray:
        """Return the greedy action for every state as an int array."""
        return np.argmax(self.q_table, axis=1)

    def state_value(self) -> np.ndarray:
        """Return the value function V(s) = max_a Q(s, a)."""
        return np.max(self.q_table, axis=1)

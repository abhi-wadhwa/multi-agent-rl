"""Base class for multi-agent environments.

Follows an OpenAI Gym-style interface adapted for multiple agents:
- reset() returns list of observations
- step(actions) takes list of actions, returns (obs, rewards, done, info)
- render() produces a visual representation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class MultiAgentEnv(ABC):
    """Abstract base for multi-agent environments.

    All environments share this interface so that training algorithms
    can work with any environment interchangeably.
    """

    n_agents: int
    obs_dim: int
    act_dim: int
    max_steps: int

    @abstractmethod
    def reset(self) -> list[np.ndarray]:
        """Reset the environment and return initial observations.

        Returns:
            List of numpy arrays, one observation per agent.
        """
        ...

    @abstractmethod
    def step(
        self, actions: list[int]
    ) -> tuple[list[np.ndarray], list[float], bool, dict[str, Any]]:
        """Execute one environment step.

        Args:
            actions: List of integer actions, one per agent.

        Returns:
            observations: Updated per-agent observations.
            rewards: Per-agent scalar rewards.
            done: Whether the episode has ended.
            info: Additional diagnostic information.
        """
        ...

    @abstractmethod
    def render(self) -> np.ndarray:
        """Render the current state as an RGB image array.

        Returns:
            RGB image array of shape (H, W, 3), dtype uint8.
        """
        ...

    def get_global_state(self) -> np.ndarray:
        """Return the global state (concatenation of all observations).

        Default implementation concatenates individual observations.
        Override for environments with richer global state.
        """
        obs = self.get_observations()
        return np.concatenate(obs)

    @abstractmethod
    def get_observations(self) -> list[np.ndarray]:
        """Return current per-agent observations without stepping."""
        ...

    def close(self) -> None:
        """Clean up resources."""
        pass

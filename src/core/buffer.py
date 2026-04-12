"""Experience replay buffers for PPO and multi-agent PPO."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    """Stores rollout experience for a single PPO agent.

    Collects (obs, action, reward, done, log_prob, value) tuples from
    environment interactions, then computes GAE advantages for policy updates.
    """

    gamma: float = 0.99
    gae_lambda: float = 0.95

    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Store a single transition."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_returns_and_advantages(
        self, last_value: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and discounted returns.

        Args:
            last_value: Bootstrap value for the final state (0 if terminal).

        Returns:
            returns: Discounted returns tensor, shape (T,).
            advantages: GAE advantages tensor, shape (T,).
        """
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)

        gae = 0.0
        next_value = last_value

        for t in reversed(range(T)):
            mask = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + self.gamma * next_value * mask - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]
            next_value = self.values[t]

        return torch.tensor(returns), torch.tensor(advantages)

    def get_batches(
        self, last_value: float = 0.0
    ) -> dict[str, torch.Tensor]:
        """Return all data as tensors with computed advantages.

        Returns:
            Dictionary with keys: observations, actions, log_probs,
            returns, advantages.
        """
        returns, advantages = self.compute_returns_and_advantages(last_value)
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "observations": torch.tensor(np.array(self.observations), dtype=torch.float32),
            "actions": torch.tensor(self.actions, dtype=torch.long),
            "log_probs": torch.tensor(self.log_probs, dtype=torch.float32),
            "returns": returns,
            "advantages": advantages,
        }

    def clear(self) -> None:
        """Reset all stored data."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def __len__(self) -> int:
        return len(self.rewards)


@dataclass
class MultiAgentRolloutBuffer:
    """Stores rollout experience for multiple agents simultaneously.

    Each agent has its own trajectory storage. Also stores global states
    for the centralized critic in MAPPO.
    """

    n_agents: int = 1
    gamma: float = 0.99
    gae_lambda: float = 0.95

    def __post_init__(self) -> None:
        self.agent_observations: list[list[np.ndarray]] = [[] for _ in range(self.n_agents)]
        self.agent_actions: list[list[int]] = [[] for _ in range(self.n_agents)]
        self.agent_rewards: list[list[float]] = [[] for _ in range(self.n_agents)]
        self.agent_log_probs: list[list[float]] = [[] for _ in range(self.n_agents)]
        self.global_states: list[np.ndarray] = []
        self.global_values: list[float] = []
        self.dones: list[bool] = []
        self.agent_messages: list[list[np.ndarray]] = [[] for _ in range(self.n_agents)]

    def add(
        self,
        observations: list[np.ndarray],
        actions: list[int],
        rewards: list[float],
        done: bool,
        log_probs: list[float],
        global_state: np.ndarray,
        global_value: float,
        messages: list[np.ndarray] | None = None,
    ) -> None:
        """Store a single multi-agent transition.

        Args:
            observations: Per-agent local observations.
            actions: Per-agent actions.
            rewards: Per-agent rewards.
            done: Whether episode ended.
            log_probs: Per-agent action log probabilities.
            global_state: Concatenated global state for centralized critic.
            global_value: Centralized value estimate.
            messages: Optional per-agent message vectors.
        """
        for i in range(self.n_agents):
            self.agent_observations[i].append(observations[i])
            self.agent_actions[i].append(actions[i])
            self.agent_rewards[i].append(rewards[i])
            self.agent_log_probs[i].append(log_probs[i])
            if messages is not None:
                self.agent_messages[i].append(messages[i])

        self.global_states.append(global_state)
        self.global_values.append(global_value)
        self.dones.append(done)

    def compute_returns_and_advantages(
        self, last_value: float = 0.0, agent_idx: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages using centralized value and per-agent rewards.

        Args:
            last_value: Bootstrap value from centralized critic.
            agent_idx: Which agent's rewards to use.

        Returns:
            returns: Discounted returns, shape (T,).
            advantages: GAE advantages, shape (T,).
        """
        rewards = self.agent_rewards[agent_idx]
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)

        gae = 0.0
        next_value = last_value

        for t in reversed(range(T)):
            mask = 1.0 - float(self.dones[t])
            delta = rewards[t] + self.gamma * next_value * mask - self.global_values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.global_values[t]
            next_value = self.global_values[t]

        return torch.tensor(returns), torch.tensor(advantages)

    def get_agent_batches(
        self, agent_idx: int, last_value: float = 0.0
    ) -> dict[str, torch.Tensor]:
        """Return agent-specific data with centralized advantages.

        Args:
            agent_idx: Index of the agent.
            last_value: Bootstrap value for the final state.

        Returns:
            Dictionary with observations, actions, log_probs, returns,
            advantages, global_states.
        """
        returns, advantages = self.compute_returns_and_advantages(last_value, agent_idx)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        result = {
            "observations": torch.tensor(
                np.array(self.agent_observations[agent_idx]), dtype=torch.float32
            ),
            "actions": torch.tensor(self.agent_actions[agent_idx], dtype=torch.long),
            "log_probs": torch.tensor(self.agent_log_probs[agent_idx], dtype=torch.float32),
            "returns": returns,
            "advantages": advantages,
            "global_states": torch.tensor(np.array(self.global_states), dtype=torch.float32),
        }

        if self.agent_messages[agent_idx]:
            result["messages"] = torch.tensor(
                np.array(self.agent_messages[agent_idx]), dtype=torch.float32
            )

        return result

    def clear(self) -> None:
        """Reset all stored data."""
        for i in range(self.n_agents):
            self.agent_observations[i].clear()
            self.agent_actions[i].clear()
            self.agent_rewards[i].clear()
            self.agent_log_probs[i].clear()
            self.agent_messages[i].clear()
        self.global_states.clear()
        self.global_values.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.dones)

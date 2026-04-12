"""Proximal Policy Optimization (PPO) for independent multi-agent learning.

Each agent runs its own PPO instance and treats other agents as part of
the environment. This is the simplest MARL approach: Independent PPO (IPPO).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.core.buffer import RolloutBuffer
from src.core.networks import Actor, Critic


@dataclass
class PPOConfig:
    """Hyperparameters for PPO."""

    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 64
    hidden_dim: int = 64


class PPOAgent:
    """Independent PPO agent with actor-critic architecture.

    Implements the clipped surrogate objective from Schulman et al. (2017)
    with Generalized Advantage Estimation (GAE).

    In multi-agent settings, each PPOAgent operates independently, treating
    other agents as part of the environment (non-stationarity).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        agent_id: int = 0,
        config: PPOConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.config = config or PPOConfig()
        self.agent_id = agent_id
        self.device = torch.device(device)

        self.actor = Actor(obs_dim, act_dim, self.config.hidden_dim).to(self.device)
        self.critic = Critic(obs_dim, self.config.hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.lr_critic
        )

        self.buffer = RolloutBuffer(
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> tuple[int, float, float]:
        """Select action given observation.

        Args:
            obs: Environment observation.
            deterministic: If True, take the greedy action.

        Returns:
            action: Chosen action (int).
            log_prob: Log probability of the action.
            value: Value estimate of the current state.
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action, log_prob = self.actor.get_action(obs_t, deterministic=deterministic)
            value = self.critic(obs_t)

        return action.item(), log_prob.item(), value.item()

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Store a transition in the rollout buffer."""
        self.buffer.add(obs, action, reward, done, log_prob, value)

    def update(self) -> dict[str, float]:
        """Run PPO update using collected rollout data.

        Performs multiple epochs of minibatch updates with clipped
        surrogate objective.

        Returns:
            Dictionary of loss metrics: policy_loss, value_loss, entropy.
        """
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # Compute bootstrap value for last state
        last_obs = self.buffer.observations[-1]
        last_done = self.buffer.dones[-1]
        if last_done:
            last_value = 0.0
        else:
            obs_t = torch.tensor(last_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                last_value = self.critic(obs_t).item()

        batch = self.buffer.get_batches(last_value)
        obs = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        old_log_probs = batch["log_probs"].to(self.device)
        returns = batch["returns"].to(self.device)
        advantages = batch["advantages"].to(self.device)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.config.n_epochs):
            # Generate random minibatch indices
            dataset_size = len(obs)
            indices = np.random.permutation(dataset_size)

            for start in range(0, dataset_size, self.config.batch_size):
                end = min(start + self.config.batch_size, dataset_size)
                mb_idx = indices[start:end]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                # Policy loss with clipped objective
                new_log_probs, entropy = self.actor.evaluate_actions(mb_obs, mb_actions)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                # Value loss
                values = self.critic(mb_obs)
                value_loss = nn.functional.mse_loss(values, mb_returns)

                # Combined actor update
                actor_loss = policy_loss + self.config.entropy_coef * entropy_loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()

                # Critic update
                critic_loss = self.config.value_coef * value_loss
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }


class IndependentPPO:
    """Manages multiple independent PPO agents.

    Each agent has its own actor and critic. Agents are trained independently,
    treating other agents as part of the non-stationary environment.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        config: PPOConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.n_agents = n_agents
        self.agents = [
            PPOAgent(obs_dim, act_dim, agent_id=i, config=config, device=device)
            for i in range(n_agents)
        ]

    def select_actions(
        self, observations: list[np.ndarray], deterministic: bool = False
    ) -> tuple[list[int], list[float], list[float]]:
        """Select actions for all agents.

        Args:
            observations: List of per-agent observations.
            deterministic: If True, take greedy actions.

        Returns:
            actions: List of per-agent actions.
            log_probs: List of per-agent log probabilities.
            values: List of per-agent value estimates.
        """
        actions, log_probs, values = [], [], []
        for i, agent in enumerate(self.agents):
            a, lp, v = agent.select_action(observations[i], deterministic)
            actions.append(a)
            log_probs.append(lp)
            values.append(v)
        return actions, log_probs, values

    def store_transitions(
        self,
        observations: list[np.ndarray],
        actions: list[int],
        rewards: list[float],
        done: bool,
        log_probs: list[float],
        values: list[float],
    ) -> None:
        """Store transitions for all agents."""
        for i, agent in enumerate(self.agents):
            agent.store_transition(
                observations[i], actions[i], rewards[i], done, log_probs[i], values[i]
            )

    def update_all(self) -> list[dict[str, float]]:
        """Update all agents and return their metrics."""
        return [agent.update() for agent in self.agents]

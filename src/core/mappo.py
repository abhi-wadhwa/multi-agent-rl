"""Multi-Agent PPO (MAPPO) with centralized training and decentralized execution.

Key idea: Each agent has its own actor (policy) that sees only local observations,
but shares a centralized critic that sees the global state (all agents' observations).
This resolves the non-stationarity problem of independent learning while maintaining
scalable decentralized execution.

Reference: Yu et al., "The Surprising Effectiveness of PPO in Cooperative
Multi-Agent Games" (2022).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.core.buffer import MultiAgentRolloutBuffer
from src.core.communication import CommunicationChannel
from src.core.networks import Actor, CentralizedCritic, CommActor


@dataclass
class MAPPOConfig:
    """Hyperparameters for MAPPO."""

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
    critic_hidden_dim: int = 128
    use_communication: bool = False
    msg_dim: int = 8


class MAPPOTrainer:
    """Multi-Agent PPO trainer with centralized critic.

    Architecture:
    - N actors: pi_i(a_i | o_i) -- each agent's policy sees only local obs
    - 1 centralized critic: V(s) where s = [o_1, ..., o_N] is global state
    - Optional communication: agents exchange differentiable messages

    Training is centralized (critic sees everything), but execution is
    decentralized (actors only use local observations + messages).
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        config: MAPPOConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.config = config or MAPPOConfig()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device(device)

        global_state_dim = obs_dim * n_agents

        # Create actors -- one per agent
        if self.config.use_communication:
            self.comm_channel = CommunicationChannel(n_agents, self.config.msg_dim)
            self.actors: list[nn.Module] = [
                CommActor(
                    obs_dim,
                    act_dim,
                    self.config.msg_dim,
                    n_agents - 1,
                    self.config.hidden_dim,
                ).to(self.device)
                for _ in range(n_agents)
            ]
        else:
            self.comm_channel = None
            self.actors = [
                Actor(obs_dim, act_dim, self.config.hidden_dim).to(self.device)
                for _ in range(n_agents)
            ]

        # Single centralized critic shared by all agents
        self.critic = CentralizedCritic(
            global_state_dim, self.config.critic_hidden_dim
        ).to(self.device)

        # Optimizers
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=self.config.lr_actor)
            for actor in self.actors
        ]
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.lr_critic
        )

        # Shared buffer
        self.buffer = MultiAgentRolloutBuffer(
            n_agents=n_agents,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

    def select_actions(
        self, observations: list[np.ndarray], deterministic: bool = False
    ) -> tuple[list[int], list[float], float]:
        """Select actions for all agents using decentralized policies.

        Args:
            observations: List of per-agent observations.
            deterministic: If True, take greedy actions.

        Returns:
            actions: Per-agent actions.
            log_probs: Per-agent log probabilities.
            global_value: Centralized critic value estimate.
        """
        actions = []
        log_probs = []
        messages_out = []

        for i in range(self.n_agents):
            obs_t = torch.tensor(
                observations[i], dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            with torch.no_grad():
                if self.config.use_communication and self.comm_channel is not None:
                    incoming = self.comm_channel.receive(i).unsqueeze(0).to(self.device)
                    actor = self.actors[i]
                    assert isinstance(actor, CommActor)
                    action, log_prob, msg = actor.get_action(
                        obs_t, incoming, deterministic=deterministic
                    )
                    messages_out.append(msg.squeeze(0))
                else:
                    actor = self.actors[i]
                    assert isinstance(actor, Actor)
                    action, log_prob = actor.get_action(obs_t, deterministic=deterministic)

            actions.append(action.item())
            log_probs.append(log_prob.item())

        # Update communication channel
        if self.config.use_communication and self.comm_channel is not None and messages_out:
            self.comm_channel.broadcast(messages_out)

        # Centralized critic
        global_state = np.concatenate(observations)
        global_state_t = torch.tensor(
            global_state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            global_value = self.critic(global_state_t).item()

        return actions, log_probs, global_value

    def store_transition(
        self,
        observations: list[np.ndarray],
        actions: list[int],
        rewards: list[float],
        done: bool,
        log_probs: list[float],
        global_value: float,
    ) -> None:
        """Store a multi-agent transition."""
        global_state = np.concatenate(observations)
        messages = None
        if self.config.use_communication and self.comm_channel is not None:
            messages = [
                self.comm_channel.messages[i].numpy() for i in range(self.n_agents)
            ]
        self.buffer.add(
            observations, actions, rewards, done, log_probs,
            global_state, global_value, messages
        )

    def update(self) -> dict[str, float]:
        """Run MAPPO update: update actors with clipped PPO, update shared critic.

        Returns:
            Dictionary of aggregated loss metrics.
        """
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # Bootstrap value
        last_state = self.buffer.global_states[-1]
        last_done = self.buffer.dones[-1]
        if last_done:
            last_value = 0.0
        else:
            state_t = torch.tensor(
                last_state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                last_value = self.critic(state_t).item()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.config.n_epochs):
            for agent_idx in range(self.n_agents):
                batch = self.buffer.get_agent_batches(agent_idx, last_value)
                obs = batch["observations"].to(self.device)
                act = batch["actions"].to(self.device)
                old_lp = batch["log_probs"].to(self.device)
                returns = batch["returns"].to(self.device)
                advantages = batch["advantages"].to(self.device)
                global_states = batch["global_states"].to(self.device)

                dataset_size = len(obs)
                indices = np.random.permutation(dataset_size)

                for start in range(0, dataset_size, self.config.batch_size):
                    end = min(start + self.config.batch_size, dataset_size)
                    mb_idx = indices[start:end]

                    mb_obs = obs[mb_idx]
                    mb_act = act[mb_idx]
                    mb_old_lp = old_lp[mb_idx]
                    mb_returns = returns[mb_idx]
                    mb_advantages = advantages[mb_idx]
                    mb_global = global_states[mb_idx]

                    # Actor update
                    actor = self.actors[agent_idx]
                    if self.config.use_communication and "messages" in batch:
                        mb_msgs = batch["messages"][mb_idx].to(self.device)
                        # For communication actors, we need incoming messages
                        # Use zeros as a simplified approach during update
                        n_other = self.n_agents - 1
                        incoming = torch.zeros(
                            len(mb_idx), n_other * self.config.msg_dim, device=self.device
                        )
                        assert isinstance(actor, CommActor)
                        dist, _ = actor(mb_obs, incoming)
                        new_lp = dist.log_prob(mb_act)
                        entropy = dist.entropy()
                    else:
                        assert isinstance(actor, Actor)
                        new_lp, entropy = actor.evaluate_actions(mb_obs, mb_act)

                    ratio = torch.exp(new_lp - mb_old_lp)
                    surr1 = ratio * mb_advantages
                    surr2 = (
                        torch.clamp(
                            ratio,
                            1.0 - self.config.clip_eps,
                            1.0 + self.config.clip_eps,
                        )
                        * mb_advantages
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()
                    entropy_loss = -entropy.mean()

                    actor_loss = policy_loss + self.config.entropy_coef * entropy_loss
                    self.actor_optimizers[agent_idx].zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(
                        actor.parameters(), self.config.max_grad_norm
                    )
                    self.actor_optimizers[agent_idx].step()

                    # Critic update (shared)
                    values = self.critic(mb_global)
                    value_loss = nn.functional.mse_loss(values, mb_returns)

                    critic_loss = self.config.value_coef * value_loss
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.critic.parameters(), self.config.max_grad_norm
                    )
                    self.critic_optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.mean().item()
                    n_updates += 1

        self.buffer.clear()
        if self.comm_channel is not None:
            self.comm_channel.reset()

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    def reset_communication(self) -> None:
        """Reset communication channel at episode start."""
        if self.comm_channel is not None:
            self.comm_channel.reset()

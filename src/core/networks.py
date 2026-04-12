"""Actor-Critic neural networks for single-agent and multi-agent PPO."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


class Actor(nn.Module):
    """Policy network that maps observations to action distributions.

    Uses a simple MLP with tanh activations. Outputs a categorical
    distribution over discrete actions.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> Categorical:
        """Return a categorical distribution over actions."""
        logits = self.net(obs)
        return Categorical(logits=logits)

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and return (action, log_prob)."""
        dist = self.forward(obs)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log_prob and entropy for given obs-action pairs."""
        dist = self.forward(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy


class Critic(nn.Module):
    """Value network that maps observations to scalar state values.

    Standard single-agent critic: V(o_i).
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return state value estimate, shape (...,)."""
        return self.net(obs).squeeze(-1)


class CentralizedCritic(nn.Module):
    """Centralized value function for MAPPO: V(s) where s is global state.

    Takes the concatenation of all agents' observations (the global state)
    and outputs a scalar value estimate. Used during training only.
    """

    def __init__(self, global_state_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """Return value estimate from global state, shape (...,)."""
        return self.net(global_state).squeeze(-1)


class CommActor(nn.Module):
    """Actor that can receive messages from other agents.

    Concatenates local observation with incoming messages before
    computing the policy. Also produces an outgoing message vector.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        msg_dim: int,
        n_other_agents: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        input_dim = obs_dim + msg_dim * n_other_agents
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )
        self.msg_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, msg_dim),
            nn.Tanh(),
        )
        self.msg_dim = msg_dim
        self.n_other_agents = n_other_agents

    def forward(
        self, obs: torch.Tensor, incoming_msgs: torch.Tensor
    ) -> tuple[Categorical, torch.Tensor]:
        """Compute action distribution and outgoing message.

        Args:
            obs: Local observation, shape (batch, obs_dim).
            incoming_msgs: Messages from other agents, shape (batch, n_other * msg_dim).

        Returns:
            dist: Categorical distribution over actions.
            message: Outgoing message vector, shape (batch, msg_dim).
        """
        x = torch.cat([obs, incoming_msgs], dim=-1)
        logits = self.policy_net(x)
        message = self.msg_net(x)
        return Categorical(logits=logits), message

    def get_action(
        self,
        obs: torch.Tensor,
        incoming_msgs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and return (action, log_prob, outgoing_message)."""
        dist, message = self.forward(obs, incoming_msgs)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, message

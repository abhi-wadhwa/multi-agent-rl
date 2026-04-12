"""Differentiable communication channel for multi-agent message passing.

Agents produce continuous message vectors that are broadcast to other agents.
Messages are concatenated with local observations before being fed into
the policy network, allowing agents to learn communication protocols.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class CommunicationChannel:
    """Manages message passing between agents.

    Each agent produces a fixed-size message vector at each timestep.
    Messages from all other agents are concatenated and provided as
    additional input to each agent's policy.

    This implements a fully-connected broadcast topology: every agent
    receives messages from every other agent.
    """

    def __init__(self, n_agents: int, msg_dim: int = 8) -> None:
        """Initialize communication channel.

        Args:
            n_agents: Number of agents.
            msg_dim: Dimensionality of each message vector.
        """
        self.n_agents = n_agents
        self.msg_dim = msg_dim
        self.messages: torch.Tensor = torch.zeros(n_agents, msg_dim)
        self.message_history: list[torch.Tensor] = []

    def reset(self) -> None:
        """Reset all messages to zero (start of episode)."""
        self.messages = torch.zeros(self.n_agents, self.msg_dim)
        self.message_history = []

    def send(self, agent_id: int, message: torch.Tensor) -> None:
        """Update the message from a specific agent.

        Args:
            agent_id: Index of the sending agent.
            message: Message vector, shape (msg_dim,) or (1, msg_dim).
        """
        if message.dim() == 2:
            message = message.squeeze(0)
        self.messages[agent_id] = message.detach()

    def receive(self, agent_id: int) -> torch.Tensor:
        """Get concatenated messages from all other agents.

        Args:
            agent_id: Index of the receiving agent.

        Returns:
            Concatenated message tensor, shape ((n_agents-1) * msg_dim,).
        """
        other_msgs = []
        for i in range(self.n_agents):
            if i != agent_id:
                other_msgs.append(self.messages[i])
        return torch.cat(other_msgs, dim=-1)

    def receive_numpy(self, agent_id: int) -> np.ndarray:
        """Get concatenated messages as numpy array."""
        return self.receive(agent_id).numpy()

    def broadcast(self, messages: list[torch.Tensor]) -> None:
        """Update all messages simultaneously.

        Args:
            messages: List of message tensors, one per agent.
        """
        for i, msg in enumerate(messages):
            self.send(i, msg)
        self.message_history.append(self.messages.clone())

    def get_all_incoming(self) -> list[torch.Tensor]:
        """Get incoming messages for all agents.

        Returns:
            List of concatenated incoming message tensors, one per agent.
        """
        return [self.receive(i) for i in range(self.n_agents)]

    def get_history(self) -> list[torch.Tensor]:
        """Return full message history for visualization."""
        return self.message_history

    @property
    def incoming_msg_dim(self) -> int:
        """Total dimensionality of incoming messages for one agent."""
        return (self.n_agents - 1) * self.msg_dim


class GatedCommunication(nn.Module):
    """Learnable gating mechanism for received messages.

    Instead of directly concatenating all incoming messages, this module
    applies a learned gating function that selectively attends to
    different messages based on the agent's current observation.
    """

    def __init__(self, obs_dim: int, total_msg_dim: int, output_dim: int) -> None:
        """Initialize gated communication module.

        Args:
            obs_dim: Dimension of the agent's observation.
            total_msg_dim: Total dimension of all incoming messages.
            output_dim: Desired output dimension after gating.
        """
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(obs_dim + total_msg_dim, total_msg_dim),
            nn.Sigmoid(),
        )
        self.transform = nn.Linear(total_msg_dim, output_dim)

    def forward(self, obs: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """Apply gated attention to incoming messages.

        Args:
            obs: Agent's local observation, shape (batch, obs_dim).
            messages: Concatenated incoming messages, shape (batch, total_msg_dim).

        Returns:
            Processed message representation, shape (batch, output_dim).
        """
        combined = torch.cat([obs, messages], dim=-1)
        gate_values = self.gate(combined)
        gated_msgs = gate_values * messages
        return self.transform(gated_msgs)

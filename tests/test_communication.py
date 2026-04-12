"""Tests for communication channel and gated communication."""

import numpy as np
import pytest
import torch

from src.core.communication import CommunicationChannel, GatedCommunication
from src.core.networks import CommActor


class TestCommunicationChannel:
    """Tests for the basic message passing channel."""

    def test_initialization(self):
        ch = CommunicationChannel(n_agents=3, msg_dim=4)
        assert ch.messages.shape == (3, 4)
        assert ch.incoming_msg_dim == 2 * 4  # (n_agents - 1) * msg_dim

    def test_send_and_receive(self):
        ch = CommunicationChannel(n_agents=3, msg_dim=4)
        msg = torch.ones(4) * 0.5
        ch.send(0, msg)
        assert torch.allclose(ch.messages[0], msg)

        # Agent 1 should receive messages from agents 0 and 2
        received = ch.receive(1)
        assert received.shape == (8,)  # 2 * 4
        assert received[:4].sum() > 0  # agent 0's message
        assert received[4:].sum() == 0  # agent 2's message (still zero)

    def test_send_2d_message(self):
        ch = CommunicationChannel(n_agents=2, msg_dim=4)
        msg = torch.ones(1, 4) * 0.3
        ch.send(0, msg)
        assert ch.messages[0].shape == (4,)

    def test_broadcast(self):
        ch = CommunicationChannel(n_agents=3, msg_dim=2)
        messages = [
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.5, 0.5]),
        ]
        ch.broadcast(messages)
        assert torch.allclose(ch.messages[0], messages[0])
        assert torch.allclose(ch.messages[1], messages[1])
        assert torch.allclose(ch.messages[2], messages[2])

    def test_get_all_incoming(self):
        ch = CommunicationChannel(n_agents=3, msg_dim=2)
        messages = [
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.5, 0.5]),
        ]
        ch.broadcast(messages)

        all_incoming = ch.get_all_incoming()
        assert len(all_incoming) == 3
        # Agent 0 receives from 1 and 2
        assert all_incoming[0].shape == (4,)  # 2 * 2

    def test_reset(self):
        ch = CommunicationChannel(n_agents=2, msg_dim=4)
        ch.send(0, torch.ones(4))
        ch.reset()
        assert ch.messages.sum() == 0
        assert len(ch.message_history) == 0

    def test_receive_numpy(self):
        ch = CommunicationChannel(n_agents=2, msg_dim=3)
        ch.send(0, torch.ones(3))
        result = ch.receive_numpy(1)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_history_tracking(self):
        ch = CommunicationChannel(n_agents=2, msg_dim=2)
        for step in range(5):
            ch.broadcast([torch.ones(2) * step, torch.zeros(2)])
        history = ch.get_history()
        assert len(history) == 5


class TestGatedCommunication:
    """Tests for the learnable gated communication module."""

    def test_forward_shape(self):
        gate = GatedCommunication(obs_dim=4, total_msg_dim=8, output_dim=6)
        obs = torch.randn(16, 4)
        msgs = torch.randn(16, 8)
        output = gate(obs, msgs)
        assert output.shape == (16, 6)

    def test_gradient_flow(self):
        gate = GatedCommunication(obs_dim=4, total_msg_dim=8, output_dim=4)
        obs = torch.randn(4, 4, requires_grad=True)
        msgs = torch.randn(4, 8, requires_grad=True)
        output = gate(obs, msgs)
        loss = output.sum()
        loss.backward()
        assert obs.grad is not None
        assert msgs.grad is not None

    def test_gate_bounded(self):
        """Gate values should be in [0, 1] due to sigmoid."""
        gate = GatedCommunication(obs_dim=4, total_msg_dim=8, output_dim=4)
        obs = torch.randn(8, 4)
        msgs = torch.randn(8, 8)
        combined = torch.cat([obs, msgs], dim=-1)
        gate_values = gate.gate(combined)
        assert (gate_values >= 0).all()
        assert (gate_values <= 1).all()


class TestCommActor:
    """Tests for the communication-enabled actor network."""

    def test_forward(self):
        actor = CommActor(obs_dim=4, act_dim=3, msg_dim=2, n_other_agents=2)
        obs = torch.randn(8, 4)
        msgs = torch.randn(8, 4)  # 2 agents * 2 msg_dim
        dist, message = actor(obs, msgs)
        assert hasattr(dist, "sample")
        assert message.shape == (8, 2)

    def test_get_action(self):
        actor = CommActor(obs_dim=4, act_dim=5, msg_dim=3, n_other_agents=1)
        obs = torch.randn(1, 4)
        msgs = torch.randn(1, 3)
        action, log_prob, message = actor.get_action(obs, msgs)
        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert message.shape == (1, 3)
        assert 0 <= action.item() < 5

    def test_message_bounded(self):
        """Messages should be bounded by tanh in [-1, 1]."""
        actor = CommActor(obs_dim=4, act_dim=3, msg_dim=4, n_other_agents=2)
        obs = torch.randn(16, 4)
        msgs = torch.randn(16, 8)
        _, message = actor(obs, msgs)
        assert (message >= -1).all()
        assert (message <= 1).all()

    def test_deterministic_action(self):
        actor = CommActor(obs_dim=4, act_dim=3, msg_dim=2, n_other_agents=1)
        obs = torch.randn(1, 4)
        msgs = torch.randn(1, 2)
        actions = set()
        for _ in range(10):
            action, _, _ = actor.get_action(obs, msgs, deterministic=True)
            actions.add(action.item())
        assert len(actions) == 1

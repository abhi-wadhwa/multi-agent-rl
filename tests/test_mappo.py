"""Tests for Multi-Agent PPO with centralized critic."""

import numpy as np
import pytest
import torch

from src.core.buffer import MultiAgentRolloutBuffer
from src.core.mappo import MAPPOConfig, MAPPOTrainer
from src.core.networks import CentralizedCritic


class TestCentralizedCritic:
    """Tests for the centralized value network."""

    def test_input_dimension(self):
        """Critic should accept concatenated observations from all agents."""
        n_agents = 3
        obs_dim = 4
        global_dim = n_agents * obs_dim
        critic = CentralizedCritic(global_dim)

        global_state = torch.randn(1, global_dim)
        value = critic(global_state)
        assert value.shape == (1,)

    def test_batch_forward(self):
        critic = CentralizedCritic(global_state_dim=12, hidden_dim=64)
        batch = torch.randn(32, 12)
        values = critic(batch)
        assert values.shape == (32,)

    def test_gradient_flow(self):
        critic = CentralizedCritic(global_state_dim=8)
        state = torch.randn(4, 8, requires_grad=True)
        values = critic(state)
        loss = values.sum()
        loss.backward()
        assert state.grad is not None


class TestMultiAgentRolloutBuffer:
    """Tests for the multi-agent rollout buffer."""

    def test_add_and_length(self):
        buf = MultiAgentRolloutBuffer(n_agents=3)
        obs = [np.zeros(4, dtype=np.float32) for _ in range(3)]
        global_state = np.zeros(12, dtype=np.float32)
        buf.add(obs, [0, 1, 2], [1.0, 0.5, 0.5], False, [-0.5, -0.3, -0.4], global_state, 0.5)
        assert len(buf) == 1

    def test_get_agent_batches(self):
        buf = MultiAgentRolloutBuffer(n_agents=2)
        for _ in range(10):
            obs = [np.random.randn(4).astype(np.float32) for _ in range(2)]
            global_state = np.concatenate(obs)
            buf.add(obs, [0, 1], [1.0, 0.5], False, [-0.5, -0.3], global_state, 0.3)
        buf.dones[-1] = True

        batch = buf.get_agent_batches(agent_idx=0, last_value=0.0)
        assert batch["observations"].shape == (10, 4)
        assert batch["global_states"].shape == (10, 8)
        assert batch["actions"].shape == (10,)
        assert batch["returns"].shape == (10,)

    def test_clear(self):
        buf = MultiAgentRolloutBuffer(n_agents=2)
        obs = [np.zeros(4, dtype=np.float32) for _ in range(2)]
        buf.add(obs, [0, 1], [1.0, 0.5], False, [-0.5, -0.3], np.zeros(8), 0.5)
        buf.clear()
        assert len(buf) == 0


class TestMAPPOTrainer:
    """Tests for the MAPPO training loop."""

    def test_initialization(self):
        trainer = MAPPOTrainer(n_agents=3, obs_dim=4, act_dim=5)
        assert len(trainer.actors) == 3
        assert trainer.critic is not None

    def test_critic_input_dim(self):
        """Centralized critic should accept n_agents * obs_dim input."""
        n_agents = 3
        obs_dim = 8
        trainer = MAPPOTrainer(n_agents=n_agents, obs_dim=obs_dim, act_dim=5)

        global_state = torch.randn(1, n_agents * obs_dim)
        value = trainer.critic(global_state)
        assert value.shape == (1,)

    def test_select_actions(self):
        trainer = MAPPOTrainer(n_agents=2, obs_dim=4, act_dim=3)
        obs = [np.random.randn(4).astype(np.float32) for _ in range(2)]
        actions, log_probs, global_value = trainer.select_actions(obs)
        assert len(actions) == 2
        assert len(log_probs) == 2
        assert all(0 <= a < 3 for a in actions)
        assert isinstance(global_value, float)

    def test_update_returns_metrics(self):
        config = MAPPOConfig(n_epochs=2, batch_size=8)
        trainer = MAPPOTrainer(n_agents=2, obs_dim=4, act_dim=3, config=config)

        for _ in range(15):
            obs = [np.random.randn(4).astype(np.float32) for _ in range(2)]
            actions, log_probs, gv = trainer.select_actions(obs)
            trainer.store_transition(obs, actions, [1.0, 0.5], False, log_probs, gv)
        # Mark last as done
        trainer.buffer.dones[-1] = True

        metrics = trainer.update()
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics

    def test_update_empty_buffer(self):
        trainer = MAPPOTrainer(n_agents=2, obs_dim=4, act_dim=3)
        metrics = trainer.update()
        assert metrics["policy_loss"] == 0.0

    def test_with_communication(self):
        config = MAPPOConfig(use_communication=True, msg_dim=4)
        trainer = MAPPOTrainer(n_agents=2, obs_dim=4, act_dim=3, config=config)
        assert trainer.comm_channel is not None

        trainer.reset_communication()
        obs = [np.random.randn(4).astype(np.float32) for _ in range(2)]
        actions, log_probs, gv = trainer.select_actions(obs)
        assert len(actions) == 2

    def test_training_on_environment(self):
        """MAPPO should run without errors on an actual environment."""
        from src.environments.predator_prey import PredatorPreyEnv

        env = PredatorPreyEnv(grid_size=5, max_steps=20)
        config = MAPPOConfig(n_epochs=2, batch_size=16)
        trainer = MAPPOTrainer(
            env.n_agents, env.obs_dim, env.act_dim, config=config
        )

        obs = env.reset()
        trainer.reset_communication()
        total_reward = 0.0

        for step in range(20):
            actions, log_probs, gv = trainer.select_actions(obs)
            next_obs, rewards, done, info = env.step(actions)
            trainer.store_transition(obs, actions, rewards, done, log_probs, gv)
            total_reward += sum(rewards)
            obs = next_obs
            if done:
                break

        metrics = trainer.update()
        assert isinstance(metrics["policy_loss"], float)

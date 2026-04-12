"""Tests for PPO agent and independent multi-agent PPO."""

import numpy as np
import pytest
import torch

from src.core.buffer import RolloutBuffer
from src.core.networks import Actor, Critic
from src.core.ppo import IndependentPPO, PPOAgent, PPOConfig


class TestActor:
    """Tests for the Actor network."""

    def test_forward_returns_distribution(self):
        actor = Actor(obs_dim=4, act_dim=3)
        obs = torch.randn(1, 4)
        dist = actor(obs)
        assert hasattr(dist, "sample")
        assert hasattr(dist, "log_prob")

    def test_get_action_shape(self):
        actor = Actor(obs_dim=8, act_dim=5)
        obs = torch.randn(1, 8)
        action, log_prob = actor.get_action(obs)
        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert 0 <= action.item() < 5

    def test_deterministic_action(self):
        actor = Actor(obs_dim=4, act_dim=3)
        obs = torch.randn(1, 4)
        actions = set()
        for _ in range(10):
            action, _ = actor.get_action(obs, deterministic=True)
            actions.add(action.item())
        # Deterministic should always give same action
        assert len(actions) == 1

    def test_evaluate_actions(self):
        actor = Actor(obs_dim=4, act_dim=3)
        obs = torch.randn(5, 4)
        actions = torch.randint(0, 3, (5,))
        log_probs, entropy = actor.evaluate_actions(obs, actions)
        assert log_probs.shape == (5,)
        assert entropy.shape == (5,)
        assert (entropy >= 0).all()

    def test_batch_forward(self):
        actor = Actor(obs_dim=4, act_dim=3)
        obs = torch.randn(16, 4)
        dist = actor(obs)
        samples = dist.sample()
        assert samples.shape == (16,)


class TestCritic:
    """Tests for the Critic network."""

    def test_forward_shape(self):
        critic = Critic(obs_dim=4)
        obs = torch.randn(1, 4)
        value = critic(obs)
        assert value.shape == (1,)

    def test_batch_forward(self):
        critic = Critic(obs_dim=8)
        obs = torch.randn(32, 8)
        values = critic(obs)
        assert values.shape == (32,)


class TestRolloutBuffer:
    """Tests for single-agent rollout buffer."""

    def test_add_and_length(self):
        buf = RolloutBuffer()
        obs = np.zeros(4, dtype=np.float32)
        buf.add(obs, 0, 1.0, False, -0.5, 0.3)
        assert len(buf) == 1

    def test_compute_returns(self):
        buf = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
        for i in range(10):
            obs = np.random.randn(4).astype(np.float32)
            buf.add(obs, 0, 1.0, i == 9, -0.5, 0.5)

        returns, advantages = buf.compute_returns_and_advantages(last_value=0.0)
        assert returns.shape == (10,)
        assert advantages.shape == (10,)
        # Returns should generally be positive for positive rewards
        assert returns.mean() > 0

    def test_get_batches(self):
        buf = RolloutBuffer()
        for _ in range(20):
            obs = np.random.randn(4).astype(np.float32)
            buf.add(obs, np.random.randint(0, 3), 1.0, False, -0.5, 0.5)
        buf.dones[-1] = True

        batch = buf.get_batches(last_value=0.0)
        assert batch["observations"].shape == (20, 4)
        assert batch["actions"].shape == (20,)
        assert batch["log_probs"].shape == (20,)
        assert batch["returns"].shape == (20,)
        assert batch["advantages"].shape == (20,)

    def test_clear(self):
        buf = RolloutBuffer()
        obs = np.zeros(4, dtype=np.float32)
        buf.add(obs, 0, 1.0, False, -0.5, 0.3)
        buf.clear()
        assert len(buf) == 0


class TestPPOAgent:
    """Tests for the PPO agent."""

    def test_select_action(self):
        agent = PPOAgent(obs_dim=4, act_dim=3)
        obs = np.random.randn(4).astype(np.float32)
        action, log_prob, value = agent.select_action(obs)
        assert isinstance(action, int)
        assert 0 <= action < 3
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_update_returns_metrics(self):
        agent = PPOAgent(obs_dim=4, act_dim=3, config=PPOConfig(n_epochs=2))
        for _ in range(20):
            obs = np.random.randn(4).astype(np.float32)
            action, log_prob, value = agent.select_action(obs)
            agent.store_transition(obs, action, 1.0, False, log_prob, value)
        agent.buffer.dones[-1] = True

        metrics = agent.update()
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert len(agent.buffer) == 0  # cleared after update

    def test_update_empty_buffer(self):
        agent = PPOAgent(obs_dim=4, act_dim=3)
        metrics = agent.update()
        assert metrics["policy_loss"] == 0.0

    def test_convergence_simple_env(self):
        """PPO should learn to prefer higher-reward actions on a trivial problem."""
        config = PPOConfig(lr_actor=1e-3, lr_critic=3e-3, n_epochs=4)
        agent = PPOAgent(obs_dim=1, act_dim=2, config=config)

        # Simple bandit: action 1 always gives reward 1, action 0 gives 0
        rewards_collected = []
        for episode in range(80):
            obs = np.array([0.0], dtype=np.float32)
            ep_reward = 0.0
            for step in range(10):
                action, log_prob, value = agent.select_action(obs)
                reward = 1.0 if action == 1 else 0.0
                ep_reward += reward
                done = step == 9
                agent.store_transition(obs, action, reward, done, log_prob, value)
            agent.update()
            rewards_collected.append(ep_reward)

        # Last 20 episodes should have higher average than first 20
        early_avg = np.mean(rewards_collected[:20])
        late_avg = np.mean(rewards_collected[-20:])
        assert late_avg > early_avg, f"PPO did not improve: early={early_avg:.2f}, late={late_avg:.2f}"


class TestIndependentPPO:
    """Tests for multi-agent independent PPO."""

    def test_select_actions(self):
        ippo = IndependentPPO(n_agents=3, obs_dim=4, act_dim=5)
        obs = [np.random.randn(4).astype(np.float32) for _ in range(3)]
        actions, log_probs, values = ippo.select_actions(obs)
        assert len(actions) == 3
        assert len(log_probs) == 3
        assert len(values) == 3
        assert all(0 <= a < 5 for a in actions)

    def test_update_all(self):
        ippo = IndependentPPO(n_agents=2, obs_dim=4, act_dim=3)
        for _ in range(15):
            obs = [np.random.randn(4).astype(np.float32) for _ in range(2)]
            actions, log_probs, values = ippo.select_actions(obs)
            ippo.store_transitions(obs, actions, [1.0, 0.5], False, log_probs, values)
        # Mark last step as done
        for agent in ippo.agents:
            agent.buffer.dones[-1] = True

        metrics_list = ippo.update_all()
        assert len(metrics_list) == 2
        assert all("policy_loss" in m for m in metrics_list)

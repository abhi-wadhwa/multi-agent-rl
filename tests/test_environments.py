"""Tests for multi-agent environments."""

import numpy as np
import pytest

from src.environments.predator_prey import PredatorPreyEnv
from src.environments.coin_game import CoinGameEnv
from src.environments.simple_spread import SimpleSpreadEnv


class TestPredatorPreyEnv:
    """Tests for Predator-Prey environment."""

    def test_reset_returns_correct_obs(self):
        env = PredatorPreyEnv(grid_size=8, n_predators=3)
        obs = env.reset()
        assert len(obs) == 3
        assert all(o.shape == (env.obs_dim,) for o in obs)
        assert all(o.dtype == np.float32 for o in obs)

    def test_step_returns_correct_shapes(self):
        env = PredatorPreyEnv(grid_size=8, n_predators=3)
        env.reset()
        actions = [0, 1, 2]  # stay, up, down
        obs, rewards, done, info = env.step(actions)
        assert len(obs) == 3
        assert len(rewards) == 3
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_actions_move_agents(self):
        env = PredatorPreyEnv(grid_size=10)
        env.reset()
        # Place predator 0 in the middle
        env.predator_positions[0] = [5, 5]
        original_pos = list(env.predator_positions[0])

        # Action 1 = up (row decreases)
        env.step([1, 0, 0])
        assert env.predator_positions[0][0] == original_pos[0] - 1

    def test_boundary_clipping(self):
        env = PredatorPreyEnv(grid_size=10)
        env.reset()
        env.predator_positions[0] = [0, 0]
        # Try to move up from top-left corner
        env.step([1, 0, 0])
        assert env.predator_positions[0][0] >= 0
        assert env.predator_positions[0][1] >= 0

    def test_capture_gives_reward(self):
        env = PredatorPreyEnv(grid_size=10)
        env.reset()
        # Place predator on prey position
        env.predator_positions[0] = list(env.prey_position)
        _, rewards, done, info = env.step([0, 0, 0])
        assert info["captured"]
        assert rewards[0] > 0
        assert done

    def test_max_steps_terminates(self):
        env = PredatorPreyEnv(grid_size=10, max_steps=5)
        env.reset()
        # Move prey far away
        env.prey_position = [9, 9]
        for p in env.predator_positions:
            p[0], p[1] = 0, 0

        done = False
        for _ in range(10):
            _, _, done, _ = env.step([0, 0, 0])
            if done:
                break
        assert done

    def test_render_shape(self):
        env = PredatorPreyEnv(grid_size=8)
        env.reset()
        img = env.render()
        assert img.ndim == 3
        assert img.shape[2] == 3
        assert img.dtype == np.uint8

    def test_n_agents_property(self):
        env = PredatorPreyEnv(n_predators=4)
        assert env.n_agents == 4


class TestCoinGameEnv:
    """Tests for Coin Game environment."""

    def test_reset(self):
        env = CoinGameEnv(grid_size=5)
        obs = env.reset()
        assert len(obs) == 2
        assert all(o.shape == (env.obs_dim,) for o in obs)

    def test_step_shapes(self):
        env = CoinGameEnv()
        env.reset()
        obs, rewards, done, info = env.step([0, 1])
        assert len(obs) == 2
        assert len(rewards) == 2
        assert isinstance(done, bool)

    def test_coin_collection_own(self):
        env = CoinGameEnv(grid_size=5)
        env.reset()
        # Place agent 0 next to the coin, make coin color 0
        env.coin_color = 0
        env.coin_position = [2, 2]
        env.agent_positions[0] = [2, 2]
        env.agent_positions[1] = [4, 4]

        _, rewards, _, info = env.step([0, 0])  # stay
        if info["coin_collected"]:
            # Agent 0 picked up own coin
            assert rewards[0] > 0
            # Other agent should not be penalized for own-coin pickup
            assert rewards[1] >= -0.02  # only step penalty

    def test_coin_collection_other(self):
        env = CoinGameEnv(grid_size=5)
        env.reset()
        # Place agent 0 on a coin that belongs to agent 1
        env.coin_color = 1
        env.coin_position = [2, 2]
        env.agent_positions[0] = [2, 2]
        env.agent_positions[1] = [4, 4]

        _, rewards, _, info = env.step([0, 0])
        if info["coin_collected"]:
            # Agent 0 picked up agent 1's coin
            assert rewards[0] > 0  # collector gets +1
            assert rewards[1] < 0  # victim gets -2

    def test_max_steps(self):
        env = CoinGameEnv(max_steps=3)
        env.reset()
        for _ in range(5):
            _, _, done, _ = env.step([0, 0])
            if done:
                break
        assert done

    def test_cooperation_rate(self):
        env = CoinGameEnv()
        env.reset()
        # Simulate: 2 own pickups, 1 other pickup
        env.own_coins_collected = [1, 1]
        env.other_coins_collected = [1, 0]
        rate = env._cooperation_rate()
        assert abs(rate - 2.0 / 3.0) < 1e-6

    def test_render(self):
        env = CoinGameEnv()
        env.reset()
        img = env.render()
        assert img.ndim == 3
        assert img.shape[2] == 3


class TestSimpleSpreadEnv:
    """Tests for Simple Spread environment."""

    def test_reset(self):
        env = SimpleSpreadEnv(grid_size=8, n_agents=3, n_landmarks=3)
        obs = env.reset()
        assert len(obs) == 3
        expected_dim = 2 + 3 * 2 + 2 * 2  # own + landmarks + others
        assert all(o.shape == (expected_dim,) for o in obs)

    def test_step_shapes(self):
        env = SimpleSpreadEnv()
        env.reset()
        obs, rewards, done, info = env.step([0, 1, 2])
        assert len(obs) == env.n_agents
        assert len(rewards) == env.n_agents
        assert isinstance(info["n_covered"], int)

    def test_coverage_reward(self):
        env = SimpleSpreadEnv(grid_size=10, n_agents=3, n_landmarks=3)
        env.reset()
        # Place agents exactly on landmarks
        for i in range(3):
            env.agent_positions[i] = list(env.landmark_positions[i])

        _, rewards, _, info = env.step([0, 0, 0])  # stay
        assert info["n_covered"] >= 2  # at least 2 should still be covered
        # Reward should include coverage bonus
        assert rewards[0] > 0

    def test_collision_penalty(self):
        env = SimpleSpreadEnv(grid_size=10)
        env.reset()
        # Place all agents on same cell
        env.agent_positions[0] = [5, 5]
        env.agent_positions[1] = [5, 5]
        env.agent_positions[2] = [5, 5]

        _, _, _, info = env.step([0, 0, 0])
        assert info["collisions"] > 0

    def test_render(self):
        env = SimpleSpreadEnv()
        env.reset()
        img = env.render()
        assert img.ndim == 3
        assert img.shape[2] == 3

    def test_observations_normalized(self):
        env = SimpleSpreadEnv(grid_size=10)
        obs = env.reset()
        for o in obs:
            assert o.min() >= 0.0
            assert o.max() <= 1.0

    def test_team_reward(self):
        """All agents should receive the same team reward."""
        env = SimpleSpreadEnv()
        env.reset()
        _, rewards, _, _ = env.step([1, 2, 3])
        assert rewards[0] == rewards[1] == rewards[2]

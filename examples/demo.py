"""Demo: Train MAPPO on Predator-Prey and show results.

Quick demonstration of the multi-agent RL framework. Trains 3 predator
agents using MAPPO with centralized critic on a small grid for a few
hundred episodes, then evaluates learned behavior.
"""

from __future__ import annotations

import numpy as np

from src.core.mappo import MAPPOConfig, MAPPOTrainer
from src.core.ppo import IndependentPPO, PPOConfig
from src.environments.predator_prey import PredatorPreyEnv
from src.environments.coin_game import CoinGameEnv
from src.environments.simple_spread import SimpleSpreadEnv


def train_predator_prey_mappo(n_episodes: int = 200) -> None:
    """Train MAPPO on Predator-Prey."""
    print("=" * 60)
    print("MAPPO on Predator-Prey (3 predators, 1 prey)")
    print("=" * 60)

    env = PredatorPreyEnv(grid_size=7, n_predators=3, max_steps=50)
    config = MAPPOConfig(
        lr_actor=3e-4,
        lr_critic=1e-3,
        n_epochs=4,
        clip_eps=0.2,
    )
    trainer = MAPPOTrainer(env.n_agents, env.obs_dim, env.act_dim, config=config)

    rewards_history = []
    capture_count = 0

    for ep in range(n_episodes):
        obs = env.reset()
        trainer.reset_communication()
        ep_reward = 0.0

        for step in range(env.max_steps):
            actions, log_probs, gv = trainer.select_actions(obs)
            next_obs, rewards, done, info = env.step(actions)
            trainer.store_transition(obs, actions, rewards, done, log_probs, gv)
            ep_reward += sum(rewards)
            obs = next_obs
            if done:
                if info.get("captured", False):
                    capture_count += 1
                break

        trainer.update()
        rewards_history.append(ep_reward)

        if (ep + 1) % 50 == 0:
            avg = np.mean(rewards_history[-50:])
            print(
                f"  Episode {ep + 1:4d} | Avg Reward: {avg:7.2f} | "
                f"Captures: {capture_count}"
            )

    print(f"\nFinal capture rate: {capture_count}/{n_episodes}")
    print()


def train_coin_game_ippo(n_episodes: int = 200) -> None:
    """Train Independent PPO on Coin Game."""
    print("=" * 60)
    print("Independent PPO on Coin Game (cooperation vs. defection)")
    print("=" * 60)

    env = CoinGameEnv(grid_size=5, max_steps=30)
    config = PPOConfig(lr_actor=3e-4, lr_critic=1e-3, n_epochs=4)
    trainer = IndependentPPO(env.n_agents, env.obs_dim, env.act_dim, config=config)

    rewards_history = []

    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward = 0.0

        for step in range(env.max_steps):
            actions, log_probs, values = trainer.select_actions(obs)
            next_obs, rewards, done, info = env.step(actions)
            trainer.store_transitions(obs, actions, rewards, done, log_probs, values)
            ep_reward += sum(rewards)
            obs = next_obs
            if done:
                break

        trainer.update_all()
        rewards_history.append(ep_reward)

        if (ep + 1) % 50 == 0:
            avg = np.mean(rewards_history[-50:])
            coop = info.get("cooperation_rate", 0.0)
            print(
                f"  Episode {ep + 1:4d} | Avg Reward: {avg:7.2f} | "
                f"Cooperation: {coop:.1%}"
            )

    print()


def train_spread_mappo(n_episodes: int = 200) -> None:
    """Train MAPPO on Simple Spread with communication."""
    print("=" * 60)
    print("MAPPO + Communication on Simple Spread (3 agents, 3 landmarks)")
    print("=" * 60)

    env = SimpleSpreadEnv(grid_size=7, n_agents=3, n_landmarks=3, max_steps=30)
    config = MAPPOConfig(
        lr_actor=3e-4,
        lr_critic=1e-3,
        use_communication=True,
        msg_dim=4,
        n_epochs=4,
    )
    trainer = MAPPOTrainer(env.n_agents, env.obs_dim, env.act_dim, config=config)

    rewards_history = []

    for ep in range(n_episodes):
        obs = env.reset()
        trainer.reset_communication()
        ep_reward = 0.0

        for step in range(env.max_steps):
            actions, log_probs, gv = trainer.select_actions(obs)
            next_obs, rewards, done, info = env.step(actions)
            trainer.store_transition(obs, actions, rewards, done, log_probs, gv)
            ep_reward += sum(rewards)
            obs = next_obs
            if done:
                break

        trainer.update()
        rewards_history.append(ep_reward)

        if (ep + 1) % 50 == 0:
            avg = np.mean(rewards_history[-50:])
            cov = info.get("coverage_ratio", 0.0)
            print(
                f"  Episode {ep + 1:4d} | Avg Reward: {avg:7.2f} | "
                f"Coverage: {cov:.1%}"
            )

    print()


if __name__ == "__main__":
    print("Multi-Agent RL Framework Demo")
    print("=" * 60)
    print()

    train_predator_prey_mappo(n_episodes=150)
    train_coin_game_ippo(n_episodes=150)
    train_spread_mappo(n_episodes=150)

    print("=" * 60)
    print("Demo complete! Run `streamlit run src/viz/app.py` for interactive UI.")

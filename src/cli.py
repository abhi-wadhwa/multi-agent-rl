"""Command-line interface for training and evaluating multi-agent RL."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from tqdm import tqdm

from src.core.mappo import MAPPOConfig, MAPPOTrainer
from src.core.ppo import IndependentPPO, PPOConfig
from src.environments import ENV_REGISTRY


def train(args: argparse.Namespace) -> None:
    """Train agents on the specified environment."""
    if args.env not in ENV_REGISTRY:
        print(f"Unknown environment: {args.env}")
        print(f"Available: {list(ENV_REGISTRY.keys())}")
        sys.exit(1)

    env = ENV_REGISTRY[args.env]()
    print(f"Environment: {args.env}")
    print(f"  Agents: {env.n_agents}, Obs: {env.obs_dim}, Act: {env.act_dim}")
    print(f"Algorithm: {args.algo}")
    print(f"Episodes: {args.episodes}")
    print()

    if args.algo == "mappo":
        config = MAPPOConfig(
            lr_actor=args.lr,
            lr_critic=args.lr * 3,
            use_communication=args.comm,
            n_epochs=args.epochs,
        )
        trainer = MAPPOTrainer(
            env.n_agents, env.obs_dim, env.act_dim, config=config, device=args.device
        )
    elif args.algo == "ippo":
        config_ippo = PPOConfig(
            lr_actor=args.lr,
            lr_critic=args.lr * 3,
            n_epochs=args.epochs,
        )
        trainer = IndependentPPO(
            env.n_agents, env.obs_dim, env.act_dim, config=config_ippo, device=args.device
        )
    else:
        print(f"Unknown algorithm: {args.algo}. Use 'mappo' or 'ippo'.")
        sys.exit(1)

    # Training loop
    all_rewards = []
    pbar = tqdm(range(args.episodes), desc="Training")

    for ep in pbar:
        obs = env.reset()
        episode_rewards = [0.0] * env.n_agents
        episode_length = 0

        if args.algo == "mappo":
            trainer.reset_communication()

        for step in range(env.max_steps):
            if args.algo == "mappo":
                actions, log_probs, global_value = trainer.select_actions(obs)
            else:
                actions, log_probs, values = trainer.select_actions(obs)

            next_obs, rewards, done, info = env.step(actions)
            episode_length += 1

            for i in range(env.n_agents):
                episode_rewards[i] += rewards[i]

            if args.algo == "mappo":
                trainer.store_transition(
                    obs, actions, rewards, done, log_probs, global_value
                )
            else:
                trainer.store_transitions(
                    obs, actions, rewards, done, log_probs, values
                )

            obs = next_obs
            if done:
                break

        # Update
        if args.algo == "mappo":
            metrics = trainer.update()
        else:
            metrics_list = trainer.update_all()
            metrics = {
                "policy_loss": np.mean([m["policy_loss"] for m in metrics_list]),
                "value_loss": np.mean([m["value_loss"] for m in metrics_list]),
                "entropy": np.mean([m["entropy"] for m in metrics_list]),
            }

        team_reward = sum(episode_rewards)
        all_rewards.append(team_reward)

        # Running average
        window = min(20, len(all_rewards))
        avg_reward = np.mean(all_rewards[-window:])

        pbar.set_postfix(
            reward=f"{team_reward:.2f}",
            avg=f"{avg_reward:.2f}",
            length=episode_length,
            ploss=f"{metrics['policy_loss']:.4f}",
        )

    print(f"\nTraining complete.")
    print(f"Final avg reward (last 20): {np.mean(all_rewards[-20:]):.2f}")
    print(f"Best episode reward: {max(all_rewards):.2f}")


def evaluate(args: argparse.Namespace) -> None:
    """Run evaluation episodes with trained agents (random policy for demo)."""
    if args.env not in ENV_REGISTRY:
        print(f"Unknown environment: {args.env}")
        sys.exit(1)

    env = ENV_REGISTRY[args.env]()
    print(f"Running {args.episodes} evaluation episodes on {args.env}...")

    all_rewards = []
    for ep in range(args.episodes):
        obs = env.reset()
        episode_rewards = [0.0] * env.n_agents

        for step in range(env.max_steps):
            actions = [np.random.randint(0, env.act_dim) for _ in range(env.n_agents)]
            obs, rewards, done, info = env.step(actions)
            for i in range(env.n_agents):
                episode_rewards[i] += rewards[i]
            if done:
                break

        team_reward = sum(episode_rewards)
        all_rewards.append(team_reward)
        print(f"  Episode {ep + 1}: team_reward={team_reward:.2f}, info={info}")

    print(f"\nMean team reward: {np.mean(all_rewards):.2f} +/- {np.std(all_rewards):.2f}")


def main() -> None:
    """Entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Reinforcement Learning CLI"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train agents")
    train_parser.add_argument(
        "--env",
        type=str,
        default="predator_prey",
        choices=list(ENV_REGISTRY.keys()),
        help="Environment name",
    )
    train_parser.add_argument(
        "--algo",
        type=str,
        default="mappo",
        choices=["mappo", "ippo"],
        help="Algorithm: mappo or ippo",
    )
    train_parser.add_argument(
        "--episodes", type=int, default=200, help="Number of training episodes"
    )
    train_parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=4, help="PPO update epochs per episode"
    )
    train_parser.add_argument(
        "--comm", action="store_true", help="Enable communication"
    )
    train_parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu or cuda)"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Evaluate agents")
    eval_parser.add_argument(
        "--env",
        type=str,
        default="predator_prey",
        choices=list(ENV_REGISTRY.keys()),
    )
    eval_parser.add_argument("--episodes", type=int, default=10)

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

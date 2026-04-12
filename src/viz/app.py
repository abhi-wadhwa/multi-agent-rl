"""Streamlit dashboard for multi-agent RL training and visualization.

Features:
- Live environment rendering with animated agents
- Training curves (per-agent and team rewards)
- Episode replay browser
- Communication message visualization
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.core.mappo import MAPPOConfig, MAPPOTrainer
from src.core.ppo import IndependentPPO, PPOConfig
from src.environments import ENV_REGISTRY


@dataclass
class TrainingLog:
    """Stores training metrics across episodes."""

    episode_rewards: list[list[float]] = field(default_factory=list)
    team_rewards: list[float] = field(default_factory=list)
    policy_losses: list[float] = field(default_factory=list)
    value_losses: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    extra_metrics: list[dict] = field(default_factory=list)


def create_reward_plot(log: TrainingLog, n_agents: int) -> go.Figure:
    """Create per-agent and team reward training curves."""
    fig = go.Figure()
    episodes = list(range(1, len(log.team_rewards) + 1))

    # Team reward
    fig.add_trace(
        go.Scatter(
            x=episodes,
            y=log.team_rewards,
            name="Team Reward",
            line=dict(color="black", width=3),
        )
    )

    # Per-agent rewards
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i in range(n_agents):
        agent_rewards = [r[i] if i < len(r) else 0.0 for r in log.episode_rewards]
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=agent_rewards,
                name=f"Agent {i}",
                line=dict(color=colors[i % len(colors)], dash="dot"),
                opacity=0.7,
            )
        )

    fig.update_layout(
        title="Training Rewards",
        xaxis_title="Episode",
        yaxis_title="Reward",
        template="plotly_white",
        height=400,
    )
    return fig


def create_loss_plot(log: TrainingLog) -> go.Figure:
    """Create policy/value loss training curves."""
    fig = go.Figure()
    episodes = list(range(1, len(log.policy_losses) + 1))

    fig.add_trace(
        go.Scatter(
            x=episodes,
            y=log.policy_losses,
            name="Policy Loss",
            line=dict(color="#1f77b4"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=episodes,
            y=log.value_losses,
            name="Value Loss",
            line=dict(color="#ff7f0e"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=episodes,
            y=log.entropies,
            name="Entropy",
            line=dict(color="#2ca02c"),
        )
    )

    fig.update_layout(
        title="Training Losses",
        xaxis_title="Episode",
        yaxis_title="Loss",
        template="plotly_white",
        height=350,
    )
    return fig


def create_comm_plot(messages_history: list[np.ndarray], n_agents: int) -> go.Figure:
    """Visualize communication messages over time as a heatmap."""
    if not messages_history:
        fig = go.Figure()
        fig.update_layout(title="No communication data")
        return fig

    # Stack messages: (timesteps, n_agents, msg_dim)
    msg_array = np.array(messages_history)
    if msg_array.ndim == 2:
        msg_array = msg_array.reshape(len(messages_history), n_agents, -1)

    # Show mean absolute message value per agent over time
    mean_msgs = np.mean(np.abs(msg_array), axis=-1)  # (timesteps, n_agents)

    fig = go.Figure(
        data=go.Heatmap(
            z=mean_msgs.T,
            x=list(range(len(messages_history))),
            y=[f"Agent {i}" for i in range(n_agents)],
            colorscale="Viridis",
        )
    )
    fig.update_layout(
        title="Communication Activity (mean |message|)",
        xaxis_title="Timestep",
        yaxis_title="Agent",
        template="plotly_white",
        height=300,
    )
    return fig


def run_episode(
    env,
    trainer,
    algo: str,
    record: bool = False,
    deterministic: bool = False,
) -> tuple[list[float], int, list[np.ndarray], dict, list[np.ndarray]]:
    """Run a single episode and return metrics.

    Returns:
        agent_rewards: Cumulative reward per agent.
        length: Episode length.
        frames: List of rendered frames (if record=True).
        last_info: Final info dict.
        comm_msgs: Communication messages over time.
    """
    obs = env.reset()
    agent_rewards = [0.0] * env.n_agents
    frames: list[np.ndarray] = []
    comm_msgs: list[np.ndarray] = []
    length = 0
    last_info: dict = {}

    if algo == "mappo":
        trainer.reset_communication()

    for step in range(env.max_steps):
        if record:
            frames.append(env.render())

        if algo == "mappo":
            actions, log_probs, global_value = trainer.select_actions(obs, deterministic)
        else:
            actions, log_probs, values = trainer.select_actions(obs, deterministic)

        next_obs, rewards, done, info = env.step(actions)
        last_info = info
        length += 1

        for i in range(env.n_agents):
            agent_rewards[i] += rewards[i]

        if not deterministic:
            if algo == "mappo":
                trainer.store_transition(obs, actions, rewards, done, log_probs, global_value)
            else:
                trainer.store_transitions(obs, actions, rewards, done, log_probs, values)

        # Record communication
        if algo == "mappo" and hasattr(trainer, "comm_channel") and trainer.comm_channel:
            comm_msgs.append(trainer.comm_channel.messages.numpy().copy())

        obs = next_obs
        if done:
            break

    if record:
        frames.append(env.render())

    return agent_rewards, length, frames, last_info, comm_msgs


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(page_title="Multi-Agent RL", layout="wide")
    st.title("Multi-Agent Reinforcement Learning")
    st.markdown("Interactive training and visualization for cooperative/competitive agents.")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    env_name = st.sidebar.selectbox(
        "Environment",
        list(ENV_REGISTRY.keys()),
        format_func=lambda x: x.replace("_", " ").title(),
    )

    algo = st.sidebar.selectbox("Algorithm", ["mappo", "ippo"])
    n_episodes = st.sidebar.slider("Training Episodes", 10, 500, 100, 10)
    lr = st.sidebar.select_slider(
        "Learning Rate",
        options=[1e-4, 3e-4, 1e-3, 3e-3],
        value=3e-4,
    )
    use_comm = st.sidebar.checkbox("Enable Communication", value=False)

    # Create environment
    env_cls = ENV_REGISTRY[env_name]
    env = env_cls()

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Agents:** {env.n_agents}")
    st.sidebar.markdown(f"**Obs dim:** {env.obs_dim}")
    st.sidebar.markdown(f"**Act dim:** {env.act_dim}")

    # Tabs
    tab_train, tab_replay, tab_comm = st.tabs(
        ["Training", "Episode Replay", "Communication"]
    )

    with tab_train:
        st.subheader("Train Agents")

        col1, col2 = st.columns([1, 1])

        with col1:
            start_training = st.button("Start Training", type="primary")

        if start_training:
            # Initialize trainer
            if algo == "mappo":
                config = MAPPOConfig(
                    lr_actor=lr,
                    lr_critic=lr * 3,
                    use_communication=use_comm,
                )
                trainer = MAPPOTrainer(
                    env.n_agents, env.obs_dim, env.act_dim, config=config
                )
            else:
                config_ippo = PPOConfig(lr_actor=lr, lr_critic=lr * 3)
                trainer = IndependentPPO(
                    env.n_agents, env.obs_dim, env.act_dim, config=config_ippo
                )

            log = TrainingLog()
            progress = st.progress(0)
            status = st.empty()
            chart_placeholder = st.empty()
            loss_placeholder = st.empty()

            for ep in range(n_episodes):
                agent_rewards, length, _, info, _ = run_episode(env, trainer, algo)

                # Update trainer
                if algo == "mappo":
                    metrics = trainer.update()
                else:
                    metrics_list = trainer.update_all()
                    metrics = {
                        "policy_loss": np.mean([m["policy_loss"] for m in metrics_list]),
                        "value_loss": np.mean([m["value_loss"] for m in metrics_list]),
                        "entropy": np.mean([m["entropy"] for m in metrics_list]),
                    }

                log.episode_rewards.append(agent_rewards)
                log.team_rewards.append(sum(agent_rewards))
                log.policy_losses.append(metrics["policy_loss"])
                log.value_losses.append(metrics["value_loss"])
                log.entropies.append(metrics["entropy"])
                log.episode_lengths.append(length)
                log.extra_metrics.append(info)

                progress.progress((ep + 1) / n_episodes)
                status.text(
                    f"Episode {ep + 1}/{n_episodes} | "
                    f"Team Reward: {sum(agent_rewards):.2f} | "
                    f"Length: {length}"
                )

                if (ep + 1) % 5 == 0 or ep == n_episodes - 1:
                    chart_placeholder.plotly_chart(
                        create_reward_plot(log, env.n_agents), use_container_width=True
                    )
                    loss_placeholder.plotly_chart(
                        create_loss_plot(log), use_container_width=True
                    )

            st.success("Training complete!")

            # Store trainer for replay
            st.session_state["trainer"] = trainer
            st.session_state["algo"] = algo
            st.session_state["env_name"] = env_name
            st.session_state["log"] = log

    with tab_replay:
        st.subheader("Episode Replay")

        if "trainer" not in st.session_state:
            st.info("Train agents first to watch episode replays.")
        else:
            if st.button("Generate Replay"):
                trainer = st.session_state["trainer"]
                algo = st.session_state["algo"]
                env_replay = ENV_REGISTRY[st.session_state["env_name"]]()

                agent_rewards, length, frames, info, comm_msgs = run_episode(
                    env_replay, trainer, algo, record=True, deterministic=True
                )

                st.markdown(f"**Episode length:** {length}")
                st.markdown(f"**Agent rewards:** {[f'{r:.2f}' for r in agent_rewards]}")

                if frames:
                    st.markdown("**Replay:**")
                    frame_idx = st.slider("Frame", 0, len(frames) - 1, 0)
                    img = Image.fromarray(frames[frame_idx])
                    st.image(img, caption=f"Step {frame_idx}", width=400)

                    # Auto-play
                    if st.button("Play Animation"):
                        placeholder = st.empty()
                        for idx, frame in enumerate(frames):
                            img = Image.fromarray(frame)
                            placeholder.image(img, caption=f"Step {idx}", width=400)
                            time.sleep(0.15)

                st.session_state["replay_comm"] = comm_msgs

    with tab_comm:
        st.subheader("Communication Visualization")

        if "replay_comm" in st.session_state and st.session_state["replay_comm"]:
            comm_msgs = st.session_state["replay_comm"]
            fig = create_comm_plot(comm_msgs, env.n_agents)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Raw Messages")
            step_idx = st.slider(
                "Timestep", 0, len(comm_msgs) - 1, 0, key="comm_step"
            )
            msg = comm_msgs[step_idx]
            for i in range(env.n_agents):
                vals = msg[i] if msg.ndim > 1 else msg
                st.text(f"Agent {i}: [{', '.join(f'{v:.3f}' for v in vals)}]")
        else:
            st.info(
                "Enable communication and run a replay to see message visualizations."
            )


if __name__ == "__main__":
    main()

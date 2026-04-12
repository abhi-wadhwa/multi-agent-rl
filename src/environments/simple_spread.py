"""Simple Spread environment: cooperative landmark coverage.

N agents must spread out to cover N landmarks. Agents are rewarded based
on how well the landmarks are covered (minimum distance from any agent to
each landmark). Agents must learn to coordinate without explicit assignment.

Inspired by the Simple Spread scenario from MPE (Multi-Agent Particle
Environment, Lowe et al. 2017).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.environments.env_base import MultiAgentEnv


class SimpleSpreadEnv(MultiAgentEnv):
    """Cooperative landmark coverage in continuous-ish grid world.

    Agents move on a discrete grid and must cover fixed landmark positions.
    The team reward is the negative sum of minimum distances from each
    landmark to its nearest agent.

    Actions: 0=stay, 1=up, 2=down, 3=left, 4=right.
    """

    def __init__(
        self,
        grid_size: int = 10,
        n_agents: int = 3,
        n_landmarks: int = 3,
        max_steps: int = 50,
    ) -> None:
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks
        self.max_steps = max_steps

        # Observation: own_pos(2) + all_landmark_positions(n_landmarks*2)
        #              + all_other_agent_positions((n_agents-1)*2)
        self.obs_dim = 2 + n_landmarks * 2 + (n_agents - 1) * 2
        self.act_dim = 5

        self.agent_positions: list[list[int]] = []
        self.landmark_positions: list[list[int]] = []
        self.current_step = 0

    def reset(self) -> list[np.ndarray]:
        """Place agents and landmarks randomly."""
        self.current_step = 0

        self.agent_positions = [
            [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)]
            for _ in range(self.n_agents)
        ]

        # Place landmarks ensuring they are distinct
        occupied: set[tuple[int, int]] = set()
        self.landmark_positions = []
        for _ in range(self.n_landmarks):
            while True:
                pos = [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)]
                if tuple(pos) not in occupied:
                    self.landmark_positions.append(pos)
                    occupied.add(tuple(pos))
                    break

        return self.get_observations()

    def step(
        self, actions: list[int]
    ) -> tuple[list[np.ndarray], list[float], bool, dict[str, Any]]:
        """Move agents and compute coverage reward."""
        self.current_step += 1
        action_deltas = {0: [0, 0], 1: [-1, 0], 2: [1, 0], 3: [0, -1], 4: [0, 1]}

        # Move agents
        for i, action in enumerate(actions):
            delta = action_deltas[action]
            new_pos = [
                np.clip(self.agent_positions[i][0] + delta[0], 0, self.grid_size - 1),
                np.clip(self.agent_positions[i][1] + delta[1], 0, self.grid_size - 1),
            ]
            self.agent_positions[i] = [int(new_pos[0]), int(new_pos[1])]

        # Compute coverage reward
        # For each landmark, find the minimum distance to any agent
        total_min_dist = 0.0
        n_covered = 0
        for lm in self.landmark_positions:
            min_dist = float("inf")
            for ag in self.agent_positions:
                dist = abs(ag[0] - lm[0]) + abs(ag[1] - lm[1])
                min_dist = min(min_dist, dist)
            total_min_dist += min_dist
            if min_dist == 0:
                n_covered += 1

        # Shared team reward: penalize total uncovered distance
        reward = -total_min_dist / (self.grid_size * self.n_landmarks)
        # Bonus for covering landmarks
        reward += n_covered * 0.5

        # Collision penalty
        collision_count = 0
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if (
                    self.agent_positions[i][0] == self.agent_positions[j][0]
                    and self.agent_positions[i][1] == self.agent_positions[j][1]
                ):
                    collision_count += 1
        reward -= collision_count * 0.1

        # All agents get the same team reward
        rewards = [reward] * self.n_agents

        done = self.current_step >= self.max_steps

        info: dict[str, Any] = {
            "step": self.current_step,
            "n_covered": n_covered,
            "total_min_dist": total_min_dist,
            "collisions": collision_count,
            "coverage_ratio": n_covered / self.n_landmarks,
        }

        return self.get_observations(), rewards, done, info

    def get_observations(self) -> list[np.ndarray]:
        """Per-agent observations: own position, landmarks, other agents."""
        observations = []
        gs = max(self.grid_size - 1, 1)

        for i in range(self.n_agents):
            parts = []
            # Own position (normalized)
            parts.append(np.array(self.agent_positions[i], dtype=np.float32) / gs)

            # All landmark positions (normalized)
            for lm in self.landmark_positions:
                parts.append(np.array(lm, dtype=np.float32) / gs)

            # Other agent positions (normalized)
            for j in range(self.n_agents):
                if j != i:
                    parts.append(np.array(self.agent_positions[j], dtype=np.float32) / gs)

            observations.append(np.concatenate(parts))

        return observations

    def render(self) -> np.ndarray:
        """Render grid with agents and landmarks."""
        cell_size = 40
        img_size = self.grid_size * cell_size
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240

        # Grid lines
        for k in range(self.grid_size + 1):
            y = min(k * cell_size, img_size - 1)
            img[y, :, :] = 200
            img[:, y, :] = 200

        # Landmarks (green circles)
        for lm in self.landmark_positions:
            self._fill_diamond(img, lm[0], lm[1], cell_size, (50, 180, 50))

        # Agents (blue shades)
        agent_colors = [
            (50, 50, 220),
            (220, 100, 50),
            (180, 50, 180),
            (50, 180, 180),
        ]
        for i, pos in enumerate(self.agent_positions):
            color = agent_colors[i % len(agent_colors)]
            self._fill_cell(img, pos[0], pos[1], cell_size, color)

        return img

    @staticmethod
    def _fill_cell(
        img: np.ndarray, row: int, col: int, cell_size: int, color: tuple[int, int, int]
    ) -> None:
        r0 = row * cell_size + 3
        c0 = col * cell_size + 3
        r1 = (row + 1) * cell_size - 3
        c1 = (col + 1) * cell_size - 3
        h, w = img.shape[:2]
        img[max(0, r0) : min(h, r1), max(0, c0) : min(w, c1)] = color

    @staticmethod
    def _fill_diamond(
        img: np.ndarray, row: int, col: int, cell_size: int, color: tuple[int, int, int]
    ) -> None:
        """Draw a diamond shape to distinguish landmarks from agents."""
        cy = row * cell_size + cell_size // 2
        cx = col * cell_size + cell_size // 2
        radius = cell_size // 3
        h, w = img.shape[:2]
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if abs(dy) + abs(dx) <= radius:
                    py, px = cy + dy, cx + dx
                    if 0 <= py < h and 0 <= px < w:
                        img[py, px] = color

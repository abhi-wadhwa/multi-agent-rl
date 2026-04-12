"""Coin Game environment: a cooperation vs. defection dilemma.

Two agents move on a grid. Colored coins (red or blue) spawn one at a time.
Any agent can pick up any coin and gets +1 reward. However, if an agent
picks up the OTHER agent's colored coin, that other agent gets -2.

This creates a social dilemma: agents can cooperate (only pick up own coins)
or defect (greedily pick up all coins, hurting the other agent).

Reference: Lerer & Peysakhovich, "Maintaining cooperation in complex
social dilemmas using deep reinforcement learning" (2017).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.environments.env_base import MultiAgentEnv


class CoinGameEnv(MultiAgentEnv):
    """Two-player Coin Game with cooperation/defection dynamics.

    Agents: Red (0) and Blue (1).
    Observations: Full grid encoded as channels (own pos, other pos, coin pos, coin color).
    Actions: 0=stay, 1=up, 2=down, 3=left, 4=right.
    """

    def __init__(self, grid_size: int = 5, max_steps: int = 50) -> None:
        self.grid_size = grid_size
        self.n_agents = 2
        self.max_steps = max_steps

        # Observation: own_pos(2) + other_pos(2) + coin_pos(2) + coin_color(1) + grid_size(1)
        self.obs_dim = 8
        self.act_dim = 5

        self.agent_positions: list[list[int]] = [[0, 0], [0, 0]]
        self.coin_position: list[int] = [0, 0]
        self.coin_color: int = 0  # 0=red, 1=blue
        self.current_step = 0

        # Metrics for tracking cooperation
        self.own_coins_collected: list[int] = [0, 0]
        self.other_coins_collected: list[int] = [0, 0]

    def reset(self) -> list[np.ndarray]:
        """Reset with random positions."""
        self.current_step = 0
        self.own_coins_collected = [0, 0]
        self.other_coins_collected = [0, 0]

        occupied: set[tuple[int, int]] = set()
        for i in range(self.n_agents):
            while True:
                pos = [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)]
                if tuple(pos) not in occupied:
                    self.agent_positions[i] = pos
                    occupied.add(tuple(pos))
                    break

        self._spawn_coin(occupied)
        return self.get_observations()

    def _spawn_coin(self, occupied: set[tuple[int, int]] | None = None) -> None:
        """Spawn a new coin at a random unoccupied position."""
        if occupied is None:
            occupied = {tuple(p) for p in self.agent_positions}

        while True:
            pos = [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)]
            if tuple(pos) not in occupied:
                self.coin_position = pos
                break

        self.coin_color = np.random.randint(0, 2)

    def step(
        self, actions: list[int]
    ) -> tuple[list[np.ndarray], list[float], bool, dict[str, Any]]:
        """Execute actions and resolve coin collection.

        Reward structure:
        - Picking up any coin: +1 for the collector
        - Picking up other's coin: -2 for the other agent
        """
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

        # Check coin collection
        rewards = [0.0, 0.0]
        coin_collected = False

        for i in range(self.n_agents):
            if (
                self.agent_positions[i][0] == self.coin_position[0]
                and self.agent_positions[i][1] == self.coin_position[1]
            ):
                coin_collected = True
                rewards[i] += 1.0  # collector always gets +1

                if self.coin_color == i:
                    # Picked up own coin - cooperative
                    self.own_coins_collected[i] += 1
                else:
                    # Picked up other's coin - defection
                    other = 1 - i
                    rewards[other] -= 2.0
                    self.other_coins_collected[i] += 1
                break  # only one agent can collect per step

        if coin_collected:
            self._spawn_coin()

        # Small step penalty to encourage faster collection
        for i in range(self.n_agents):
            rewards[i] -= 0.01

        done = self.current_step >= self.max_steps

        info: dict[str, Any] = {
            "coin_collected": coin_collected,
            "coin_color": self.coin_color,
            "step": self.current_step,
            "own_coins": list(self.own_coins_collected),
            "other_coins": list(self.other_coins_collected),
            "cooperation_rate": self._cooperation_rate(),
        }

        return self.get_observations(), rewards, done, info

    def _cooperation_rate(self) -> float:
        """Fraction of coins that were picked up by the matching agent."""
        total_own = sum(self.own_coins_collected)
        total_other = sum(self.other_coins_collected)
        total = total_own + total_other
        if total == 0:
            return 0.0
        return total_own / total

    def get_observations(self) -> list[np.ndarray]:
        """Return per-agent observations.

        Each agent sees: own normalized position (2), other's normalized position (2),
        coin normalized position (2), coin color relative to self (1), grid size (1).
        """
        observations = []
        gs = max(self.grid_size - 1, 1)

        for i in range(self.n_agents):
            other = 1 - i
            obs = np.array(
                [
                    self.agent_positions[i][0] / gs,
                    self.agent_positions[i][1] / gs,
                    self.agent_positions[other][0] / gs,
                    self.agent_positions[other][1] / gs,
                    self.coin_position[0] / gs,
                    self.coin_position[1] / gs,
                    1.0 if self.coin_color == i else -1.0,  # +1 if own coin, -1 if other's
                    self.grid_size / 10.0,
                ],
                dtype=np.float32,
            )
            observations.append(obs)

        return observations

    def render(self) -> np.ndarray:
        """Render as RGB image."""
        cell_size = 60
        img_size = self.grid_size * cell_size
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240

        # Grid lines
        for k in range(self.grid_size + 1):
            y = min(k * cell_size, img_size - 1)
            img[y, :, :] = 200
            img[:, y, :] = 200

        # Coin (circle approximation)
        coin_color = (220, 50, 50) if self.coin_color == 0 else (50, 50, 220)
        cr, cc = self.coin_position
        self._fill_circle(img, cr, cc, cell_size, coin_color, radius_frac=0.3)

        # Agent 0 (Red)
        self._fill_cell(img, *self.agent_positions[0], cell_size, (200, 80, 80))
        # Agent 1 (Blue)
        self._fill_cell(img, *self.agent_positions[1], cell_size, (80, 80, 200))

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
    def _fill_circle(
        img: np.ndarray,
        row: int,
        col: int,
        cell_size: int,
        color: tuple[int, int, int],
        radius_frac: float = 0.3,
    ) -> None:
        cy = row * cell_size + cell_size // 2
        cx = col * cell_size + cell_size // 2
        radius = int(cell_size * radius_frac)
        h, w = img.shape[:2]
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy * dy + dx * dx <= radius * radius:
                    py, px = cy + dy, cx + dx
                    if 0 <= py < h and 0 <= px < w:
                        img[py, px] = color

"""Predator-Prey grid world environment.

3 predators cooperatively chase 1 prey on a grid. The prey moves randomly.
Predators are rewarded when they collectively surround or catch the prey.
This environment tests cooperative behavior and emergent hunting strategies.

Observations: Local view around each agent (partial observability).
Actions: 0=stay, 1=up, 2=down, 3=left, 4=right.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.environments.env_base import MultiAgentEnv


class PredatorPreyEnv(MultiAgentEnv):
    """Cooperative predator-prey on a discrete grid.

    Predators must coordinate to capture the prey. Capture occurs when
    a predator occupies the same cell as the prey. Team reward is given
    on capture; a small step penalty encourages efficiency.
    """

    def __init__(
        self,
        grid_size: int = 10,
        n_predators: int = 3,
        view_range: int = 3,
        max_steps: int = 100,
    ) -> None:
        self.grid_size = grid_size
        self.n_predators = n_predators
        self.n_agents = n_predators
        self.view_range = view_range
        self.max_steps = max_steps

        # Observation: flattened local view (2*view_range+1)^2 * channels + own position
        view_side = 2 * view_range + 1
        # Channels: empty(0), predator(1), prey(2), wall(3) encoded as one-hot -> 4 channels
        self.obs_dim = view_side * view_side * 4 + 2  # +2 for normalized own position
        self.act_dim = 5  # stay, up, down, left, right

        self.predator_positions: list[list[int]] = []
        self.prey_position: list[int] = [0, 0]
        self.current_step = 0

    def reset(self) -> list[np.ndarray]:
        """Place predators and prey randomly on the grid."""
        self.current_step = 0
        occupied: set[tuple[int, int]] = set()

        self.predator_positions = []
        for _ in range(self.n_predators):
            while True:
                pos = [
                    np.random.randint(0, self.grid_size),
                    np.random.randint(0, self.grid_size),
                ]
                if tuple(pos) not in occupied:
                    self.predator_positions.append(pos)
                    occupied.add(tuple(pos))
                    break

        while True:
            pos = [
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size),
            ]
            if tuple(pos) not in occupied:
                self.prey_position = pos
                break

        return self.get_observations()

    def step(
        self, actions: list[int]
    ) -> tuple[list[np.ndarray], list[float], bool, dict[str, Any]]:
        """Execute actions for all predators, then move prey randomly.

        Args:
            actions: One action per predator (0-4).

        Returns:
            observations, rewards, done, info.
        """
        self.current_step += 1
        action_deltas = {0: [0, 0], 1: [-1, 0], 2: [1, 0], 3: [0, -1], 4: [0, 1]}

        # Move predators
        for i, action in enumerate(actions):
            delta = action_deltas[action]
            new_pos = [
                np.clip(self.predator_positions[i][0] + delta[0], 0, self.grid_size - 1),
                np.clip(self.predator_positions[i][1] + delta[1], 0, self.grid_size - 1),
            ]
            self.predator_positions[i] = [int(new_pos[0]), int(new_pos[1])]

        # Check capture before prey moves
        captured = any(
            p[0] == self.prey_position[0] and p[1] == self.prey_position[1]
            for p in self.predator_positions
        )

        if not captured:
            # Move prey randomly (simple evasion)
            prey_action = np.random.randint(0, 5)
            delta = action_deltas[prey_action]
            new_prey = [
                np.clip(self.prey_position[0] + delta[0], 0, self.grid_size - 1),
                np.clip(self.prey_position[1] + delta[1], 0, self.grid_size - 1),
            ]
            self.prey_position = [int(new_prey[0]), int(new_prey[1])]

        # Compute rewards
        rewards = []
        for i in range(self.n_predators):
            if captured:
                rewards.append(10.0)
            else:
                # Proximity reward: small bonus for being close to prey
                dist = abs(self.predator_positions[i][0] - self.prey_position[0]) + abs(
                    self.predator_positions[i][1] - self.prey_position[1]
                )
                proximity_reward = -0.1 * dist / self.grid_size
                rewards.append(-0.1 + proximity_reward)  # step penalty + proximity

        done = captured or self.current_step >= self.max_steps

        info: dict[str, Any] = {
            "captured": captured,
            "step": self.current_step,
            "predator_positions": [list(p) for p in self.predator_positions],
            "prey_position": list(self.prey_position),
        }

        return self.get_observations(), rewards, done, info

    def get_observations(self) -> list[np.ndarray]:
        """Get local partial observations for each predator.

        Each predator sees a (2*view_range+1) x (2*view_range+1) window
        around itself, encoded as one-hot channels.
        """
        observations = []
        for i in range(self.n_predators):
            obs = self._get_local_view(i)
            observations.append(obs)
        return observations

    def _get_local_view(self, agent_idx: int) -> np.ndarray:
        """Compute local observation for one agent."""
        view_side = 2 * self.view_range + 1
        # 4 channels: empty, predator, prey, wall
        grid = np.zeros((view_side, view_side, 4), dtype=np.float32)

        ax, ay = self.predator_positions[agent_idx]

        for dx in range(-self.view_range, self.view_range + 1):
            for dy in range(-self.view_range, self.view_range + 1):
                gx, gy = ax + dx, ay + dy
                lx, ly = dx + self.view_range, dy + self.view_range

                if gx < 0 or gx >= self.grid_size or gy < 0 or gy >= self.grid_size:
                    grid[lx, ly, 3] = 1.0  # wall
                else:
                    # Check for predators
                    is_predator = False
                    for j, pos in enumerate(self.predator_positions):
                        if j != agent_idx and pos[0] == gx and pos[1] == gy:
                            grid[lx, ly, 1] = 1.0
                            is_predator = True
                            break
                    # Check for prey
                    if self.prey_position[0] == gx and self.prey_position[1] == gy:
                        grid[lx, ly, 2] = 1.0
                    elif not is_predator:
                        grid[lx, ly, 0] = 1.0  # empty

        flat = grid.flatten()
        # Add normalized own position
        norm_pos = np.array(
            [ax / max(self.grid_size - 1, 1), ay / max(self.grid_size - 1, 1)],
            dtype=np.float32,
        )
        return np.concatenate([flat, norm_pos])

    def render(self) -> np.ndarray:
        """Render the grid as an RGB image."""
        cell_size = 40
        img_size = self.grid_size * cell_size
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240  # light gray bg

        # Draw grid lines
        for i in range(self.grid_size + 1):
            y = i * cell_size
            img[min(y, img_size - 1), :, :] = 200
            img[:, min(y, img_size - 1), :] = 200

        # Draw prey (red)
        px, py = self.prey_position
        self._fill_cell(img, px, py, cell_size, color=(220, 50, 50))

        # Draw predators (blue shades)
        colors = [(50, 50, 220), (50, 150, 220), (50, 220, 150)]
        for i, pos in enumerate(self.predator_positions):
            color = colors[i % len(colors)]
            self._fill_cell(img, pos[0], pos[1], cell_size, color=color)

        return img

    @staticmethod
    def _fill_cell(
        img: np.ndarray, row: int, col: int, cell_size: int, color: tuple[int, int, int]
    ) -> None:
        """Fill a grid cell with a color, leaving a border."""
        r0 = row * cell_size + 2
        c0 = col * cell_size + 2
        r1 = (row + 1) * cell_size - 2
        c1 = (col + 1) * cell_size - 2
        h, w = img.shape[:2]
        r0, r1 = max(0, r0), min(h, r1)
        c0, c1 = max(0, c0), min(w, c1)
        img[r0:r1, c0:c1] = color

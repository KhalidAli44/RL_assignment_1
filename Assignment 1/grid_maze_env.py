import random
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GridMazeEnv(gym.Env):
    """Custom 5x5 Grid Maze Environment"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, grid_size=5, render_mode=None):
        super(GridMazeEnv, self).__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode

        # --- Rendering setup ---
        pygame.init()
        self.cell_size = 100  # pixels per cell
        self.window_size = (self.grid_size * self.cell_size, self.grid_size * self.cell_size)
        self.window_surface = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Grid Maze")

        # Used when rendering in human mode
        self.clock = pygame.time.Clock()

        # --- Observation space ---
        # [agent_x, agent_y, goal_x, goal_y, bad1_x, bad1_y, bad2_x, bad2_y]
        self.observation_space = spaces.Box(
            low=0, high=grid_size - 1, shape=(8,), dtype=np.int32
        )

        # --- Action space ---
        # 0: Right, 1: Up, 2: Left, 3: Down
        self.action_space = spaces.Discrete(4)

        # Initialize positions
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random positions for agent, goal, and bad cells (no overlaps)
        positions = random.sample(range(self.grid_size ** 2), 4)
        coords = [(p // self.grid_size, p % self.grid_size) for p in positions]
        self.agent_pos, self.goal_pos, self.bad1_pos, self.bad2_pos = coords
        # self.agent_pos = (4,0)
        # self.goal_pos = (0,4)
        # self.bad1_pos = (2,3)
        # self.bad2_pos = (3,2)

        obs = np.array(
            [
                *self.agent_pos,
                *self.goal_pos,
                *self.bad1_pos,
                *self.bad2_pos,
            ],
            dtype=np.int32,
        )

        self.done = False
        return obs, {}

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start again.")

        # Stochastic movement
        move_probs = [0, 0, 0, 0]
        move_probs[action] = 0.7
        perpendiculars = [((action + 1) % 4), ((action + 3) % 4)]
        for a in perpendiculars:
            move_probs[a] = 0.15

        chosen_action = np.random.choice([0, 1, 2, 3], p=move_probs)
        dx, dy = self._action_to_delta(chosen_action)

        # Move agent within grid bounds
        new_x = np.clip(self.agent_pos[0] + dx, 0, self.grid_size - 1)
        new_y = np.clip(self.agent_pos[1] + dy, 0, self.grid_size - 1)
        self.agent_pos = (new_x, new_y)

        # Compute reward
        reward = self._get_reward()
        self.done = reward == 10 or reward == -10  # stop if goal or bad cell reached

        obs = np.array(
            [
                *self.agent_pos,
                *self.goal_pos,
                *self.bad1_pos,
                *self.bad2_pos,
            ],
            dtype=np.int32,
        )
        return obs, reward, self.done, False, {}

    def _action_to_delta(self, action):
        if action == 0:  # right
            return (0, 1)
        elif action == 1:  # up
            return (-1, 0)
        elif action == 2:  # left
            return (0, -1)
        elif action == 3:  # down
            return (1, 0)

    def _get_reward(self):
        if self.agent_pos == self.goal_pos:
            return 10
        elif self.agent_pos in [self.bad1_pos, self.bad2_pos]:
            return -10
        else:
            return -1  # small negative reward to encourage faster goal reach

    def render(self):
        # Colors
        WHITE = (255, 255, 255)
        GREY = (180, 180, 180)
        GREEN = (0, 200, 0)
        RED = (200, 0, 0)
        BLUE = (0, 0, 255)

        self.window_surface.fill(WHITE)

        # Draw grid lines
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(
                    y * self.cell_size,
                    x * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.window_surface, GREY, rect, 1)

        # Draw goal cell
        gx, gy = self.goal_pos
        pygame.draw.rect(
            self.window_surface,
            GREEN,
            (gy * self.cell_size, gx * self.cell_size, self.cell_size, self.cell_size),
        )

        # Draw bad cells
        for bx, by in [self.bad1_pos, self.bad2_pos]:
            pygame.draw.rect(
                self.window_surface,
                RED,
                (by * self.cell_size, bx * self.cell_size, self.cell_size, self.cell_size),
            )

        # Draw agent
        ax, ay = self.agent_pos
        pygame.draw.circle(
            self.window_surface,
            BLUE,
            (
                ay * self.cell_size + self.cell_size // 2,
                ax * self.cell_size + self.cell_size // 2,
            ),
            self.cell_size // 3,
        )

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

        if self.render_mode == "rgb_array":
            frame = pygame.surfarray.array3d(self.window_surface)
            frame = np.transpose(frame, (1, 0, 2))  # convert to (H, W, C)
            return frame

    def close(self):
        pygame.quit()
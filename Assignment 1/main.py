import gymnasium as gym
import numpy as np
from grid_maze_env import GridMazeEnv
from policy_iteration import policy_iteration

GRID_SIZE = 5
env = gym.wrappers.RecordVideo(
    GridMazeEnv(grid_size=GRID_SIZE, render_mode="rgb_array"),
    video_folder="videos/",
    name_prefix="policy_iteration_agent",
    episode_trigger=lambda episode_id: True
)
# ---- MUST RESET HERE to start video recording ----
obs, info = env.reset()

# ---- POLICY ITERATION ----
print("Running Policy Iteration...")
optimal_policy, V = policy_iteration(env.env, gamma=0.9, theta=1e-4)
print("Policy Iteration complete!\nValue vector (flat):")
print(V)

state_to_index = lambda pos: pos[0] * GRID_SIZE + pos[1]

done = False
total_reward = 0.0

while not done:
    row, col = env.unwrapped.agent_pos
    s_idx = state_to_index((row, col))
    action = int(np.argmax(optimal_policy[s_idx]))

    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

env.close()
print(f"✅ Episode finished with total reward = {total_reward}")
print("🎥 Video saved in videos/")
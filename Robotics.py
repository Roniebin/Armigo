import gymnasium as gym

# env = gym.make('FetchPickAndPlace-v2', render_mode="human")
# env = gym.make('FetchReach-v2', render_mode="human")
# env = gym.make('AdroitHandDoorSparse-v1',
#                render_mode="human", max_episode_steps=400)
env = gym.make('PointMaze_UMaze-v3',
               render_mode="human")

observation, info = env.reset(seed=42)

for _ in range(50000):
    action = env.action_space.sample()  # random action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset(seed=42)

env.close()

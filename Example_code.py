import gymnasium as gym
env = gym.make("FetchPickAndPlace-v2", render_mode="human")

observation, info = env.reset(seed=42)
total_reward = 0
last_reward = 0
for _ in range(1000):

   if last_reward >0:
      action = last_action
   else:
      action = env.action_space.sample()

   observation, reward, terminated, truncated, info = env.step(action)

   total_reward += last_reward
   last_reward = reward
   last_action = action

   if terminated or truncated:
      observation, info = env.reset()
      total_reward = 0
      last_reward = 0
env.close()

print(f"총 보상: {total_reward}")

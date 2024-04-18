import gym as gym
import os
import panda_gym
import numpy as np
from gym import spaces
from stable_baselines3 import DDPG
from stable_baselines3.her import HerReplayBuffer
from sb3_contrib import TQC

class CustomEnv(gym.Env):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.action_space = env.action_space
        # 첫 6개 차원은 -10.0에서 10.0, 마지막 차원은 0에서 1 범위
        low = np.array([-10.0] * 6 + [0.0], dtype=np.float32)
        high = np.array([10.0] * 6 + [1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            'achieved_goal': gym.spaces.Box(-10.0, 10.0, (3,), dtype=np.float32),
            'desired_goal': gym.spaces.Box(-10.0, 10.0, (3,), dtype=np.float32),
            'observation': gym.spaces.Box(low, high, dtype=np.float32)
        })

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.modify_observation(obs), reward, done, info

    def reset(self):
        return self.modify_observation(self.env.reset())

    def modify_observation(self, observation):
        # Modify the observation as needed
        return observation
# 환경 생성과 래퍼 적용
env = gym.make('PandaReach-v1')
env = gym.wrappers.TimeLimit(env, max_episode_steps=1000) 
env = CustomEnv(env)

# 모델 로드
model = TQC.load("tqc-PandaReach-v1", env, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs={'n_sampled_goal': 4, 'goal_selection_strategy': 'future', 'max_episode_length': 100})
observation, info,_ = env.reset()

for _ in range(1000):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated = env.step(action)
    if terminated or truncated:
        observation, info,_ = env.reset()

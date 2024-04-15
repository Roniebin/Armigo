import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

# 환경 초기화
env = gym.make('FetchReachDense-v2',render_mode="human")
obs_dim = env.observation_space['observation'].shape[0]
act_dim = env.action_space.shape[0]
class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dimension) * mu

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale

# Actor 네트워크
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Tanh()
        )
        self.act_limit = act_limit

    def forward(self, obs):
        return self.act_limit * self.net(obs)

# Critic 네트워크
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))

# DDPG 클래스
class DDPG:
    def __init__(self, obs_dim, act_dim, act_limit,noise_scale=0.1,epsilon=1.0,epsilon_decay=0.995):
        self.actor = Actor(obs_dim, act_dim, act_limit)
        self.critic = Critic(obs_dim, act_dim)
        self.target_actor = Actor(obs_dim, act_dim, act_limit)
        self.target_critic = Critic(obs_dim, act_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.noise_scale = noise_scale
        self.act_limit = act_limit
        self.act_dim =act_dim
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def select_action(self, obs):
        if(np.random.rand()< self.epsilon):
            action = np.random.uniform(-self.act_limit,self.act_limit,size= act_dim)
        else:
            obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = self.actor(obs).detach().numpy()  # 신경망을 통해 행동 결정
            noise = self.noise_scale * np.random.randn(self.act_dim)  # 노이즈 생성
            action += noise  # 행동에 노이즈 추가
            action = np.clip(action, -self.act_limit, self.act_limit)  # 행동 범위 제한
            action = action.squeeze()  # 배치 차원 제거
        return action

    
    @staticmethod
    def compute_reward(achieved_goal, goal):
       # 여기서 'self' 매개변수가 제거됨
        if not isinstance(achieved_goal, np.ndarray):
             achieved_goal = achieved_goal.numpy()
        if not isinstance(goal, np.ndarray):
            goal = goal.numpy()
        distance = np.linalg.norm(achieved_goal - goal, axis=-1)
        print(f"goal : {goal}, cur : {achieved_goal}, distance: {distance}")

        return -distance  # 거리가 작을수록 보상이 높아짐
    
    def update(self, obs, act, reward, next_obs, done,achieved_goal, desired_goal):
    # 텐서 변환 및 차원 조정
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        act = torch.as_tensor(act, dtype=torch.float32).unsqueeze(0)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0)

        # 예시: 'next_obs'의 특정 요소(예를 들어 인덱스 0)가 목표에 근접할수록 보상 증가
        # 이 부분은 환경의 구체적 목표에 맞게 수정 필요
        achieved_goal = torch.as_tensor(achieved_goal, dtype=torch.float32)
        goal = torch.as_tensor(desired_goal, dtype=torch.float32)
        reward = DDPG.compute_reward(achieved_goal, goal)
        with torch.no_grad():
            next_act = self.target_actor(next_obs)
            target_q = reward + 0.99 * self.target_critic(next_obs, next_act) * (1 - done)

        # Critic 네트워크 업데이트
        critic_loss = ((self.critic(obs, act) - target_q)**2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor 네트워크 업데이트
        actor_loss = -self.critic(obs, self.actor(obs)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return reward

# 메인 학습 루프
def main():
    ddpg = DDPG(obs_dim, act_dim, act_limit=1)
    episodes = 50
    rewards = []
    max_steps_per_episode = 100  # 한 에피소드 당 최대 스텝 수
    for ep in range(episodes):
        observation, info = env.reset()
        obs = torch.as_tensor(observation['observation'], dtype=torch.float32)
        desired_goal = torch.as_tensor(observation['desired_goal'], dtype=torch.float32)
        done = False
        total_reward = 0
        steps = 0  # 스텝 카운터 초기화
        while steps<max_steps_per_episode:
            act = ddpg.select_action(obs)
            # act = ddpg.actor(obs).detach().numpy()
            next_observation, reward, done, info,_ = env.step(act)
            next_obs = next_observation['observation']
            next_obs = torch.as_tensor(next_obs, dtype=torch.float32)  # 다음 관측값을 Tensor로 변환
    
            achieved_goal = torch.as_tensor(next_observation['achieved_goal'], dtype=torch.float32)
            custom_reward = ddpg.update(obs, act, reward, next_obs, done, achieved_goal, desired_goal)
            obs = next_obs  # 다음 상태를 현재 상태로 업데이트
            env.render()
            steps+=1
            total_reward += reward
        rewards.append(total_reward)
        print(f"Episode {ep + 1}: Total Reward = {total_reward}")
    # 보상 시각화
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward over Episodes')
    plt.show()

if __name__ == "__main__":
    main()

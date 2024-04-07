import gym 
import numpy as np
import matplotlib.pyplot as plt
# 환경 구축
env = gym.make('FrozenLake-v1')
# Q 테이블 생성
q_table = np.zeros((env.observation_space.n,env.action_space.n))

Episodes = 2000
Epsilon = 0.9
Gamma = 0.95
Learning_rate = 0.81
MAX_STEPS = 100 
rewards = []
#실행 구문
for _ in range(Episodes):
    # 상태 초기화
    state = env.reset()
    done = False
    while not done:
      if np.random.uniform(0,1) < Epsilon:
          action = env.action_space.sample()
      else: 
          action = np.argmax(q_table[state,:])
      next_state,reward,done,_ = env.step(action)
      q_table[state,action] = (1-Learning_rate) * (q_table[state,action]) + Gamma*(reward + Learning_rate*np.max(q_table[next_state,:]))
      #q_table[state,action] = q_table[state,action] + Learning_rate * (reward + Gamma*np.max(q_table[next_state,:]) - q_table[state,action])
      state = next_state

    rewards.append(reward)
    Epsilon -= 0.001

print(q_table)
print(f"Avg reward :{sum(rewards)/ len(rewards)}")

state = env.reset()
done = False
while not done:
    env.render()
    action = np.argmax(q_table[state])
    state, _, done, _ = env.step(action)

avg_rewards = []
for i in range(0,len(rewards),100):
    avg_rewards.append(sum(rewards[i:i+100]) / len(rewards[i:i+100]))
  
plt.plot(avg_rewards)
plt.ylabel('average reward')
plt.xlabel('episodes (100\'s)')
plt.show()
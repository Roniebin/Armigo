---
library_name: stable-baselines3
tags:
- PandaPickAndPlace-v3
- deep-reinforcement-learning
- reinforcement-learning
- stable-baselines3
model-index:
- name: TQC
  results:
  - task:
      type: reinforcement-learning
      name: reinforcement-learning
    dataset:
      name: PandaPickAndPlace-v3
      type: PandaPickAndPlace-v3
    metrics:
    - type: mean_reward
      value: -6.10 +/- 1.45
      name: mean_reward
      verified: false
---

# **TQC** Agent playing **PandaPickAndPlace-v3**
This is a trained model of a **TQC** agent playing **PandaPickAndPlace-v3**
using the [stable-baselines3 library](https://github.com/DLR-RM/stable-baselines3)
and the [RL Zoo](https://github.com/DLR-RM/rl-baselines3-zoo).

The RL Zoo is a training framework for Stable Baselines3
reinforcement learning agents,
with hyperparameter optimization and pre-trained agents included.

## Usage (with SB3 RL Zoo)

RL Zoo: https://github.com/DLR-RM/rl-baselines3-zoo<br/>
SB3: https://github.com/DLR-RM/stable-baselines3<br/>
SB3 Contrib: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib

Install the RL Zoo (with SB3 and SB3-Contrib):
```bash
pip install rl_zoo3
```

```
# Download model and save it into the logs/ folder
python -m rl_zoo3.load_from_hub --algo tqc --env PandaPickAndPlace-v3 -orga chencliu -f logs/
python -m rl_zoo3.enjoy --algo tqc --env PandaPickAndPlace-v3  -f logs/
```

If you installed the RL Zoo3 via pip (`pip install rl_zoo3`), from anywhere you can do:
```
python -m rl_zoo3.load_from_hub --algo tqc --env PandaPickAndPlace-v3 -orga chencliu -f logs/
python -m rl_zoo3.enjoy --algo tqc --env PandaPickAndPlace-v3  -f logs/
```

## Training (with the RL Zoo)
```
python -m rl_zoo3.train --algo tqc --env PandaPickAndPlace-v3 -f logs/
# Upload the model and generate video (when possible)
python -m rl_zoo3.push_to_hub --algo tqc --env PandaPickAndPlace-v3 -f logs/ -orga chencliu
```

## Hyperparameters
```python
OrderedDict([('batch_size', 2048),
             ('buffer_size', 1000000),
             ('ent_coef', 'auto'),
             ('gamma', 0.95),
             ('learning_rate', 0.001),
             ('learning_starts', 100),
             ('n_timesteps', 5000000.0),
             ('normalize', True),
             ('policy', 'MultiInputPolicy'),
             ('policy_kwargs', 'dict(net_arch=[512, 512, 512], n_critics=2)'),
             ('replay_buffer_class', 'HerReplayBuffer'),
             ('replay_buffer_kwargs',
              "dict( goal_selection_strategy='future', n_sampled_goal=4 )"),
             ('tau', 0.05),
             ('normalize_kwargs', {'norm_obs': True, 'norm_reward': False})])
```

# Environment Arguments
```python
{'render_mode': 'rgb_array'}
```

Panda Gym environments: [arxiv.org/abs/2106.13687](https://arxiv.org/abs/2106.13687)
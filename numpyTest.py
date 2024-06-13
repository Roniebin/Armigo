import numpy as np
# a = [4.2, 5.2, 4.4, 5.4, 1.5, 3.8, 45.4, 25.8, 6.9]
# b = [0.2, -0.2, 0.01, 0.2, 0, 0.01, 0.2, 0.2, 0.01]

# a_ndarray = np.array(a, np.float32)
# b_ndarray = np.array(b, np.float32)


# def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
#     assert a.shape == b.shape
#     return np.linalg.norm(a - b, axis=-1)


# num_objects = 3
# rt = 0
# for i in range(num_objects):
#     d = distance(a_ndarray[i*3:(i+1)*3], b_ndarray[i*3:(i+1)*3])
#     rt += -np.log(d + 1)
#     print(rt)

# d = distance(a_ndarray, b_ndarray)
# log_reward = -np.log(d + 1)
# print(log_reward)

d = 5
reward = -np.log(d)
print(reward)
d = 4
reward = -np.log(d)
print(reward)
d = 3
reward = -np.log(d)
print(reward)
d = 2
reward = -np.log(d)
print(reward)

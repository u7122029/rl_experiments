import numpy as np

# Assume S = 4, A = 3 for example
S, A = 4, 3

pi = np.random.rand(S, A)
pi = pi / pi.sum(axis=1, keepdims=True)

p = np.random.rand(S, S, A)
p = p / p.sum(axis=1, keepdims=True)

# Reshape pi to (S, 1, A) to align axes for broadcasting
pi_reshaped = pi[:, np.newaxis, :]  # (S, 1, A)
print(pi_reshaped.shape)

# Element-wise multiply and sum over actions
T_pi = (p * pi_reshaped).sum(axis=2)  # Result shape: (S, S)

print(T_pi)

import numpy as np
import matplotlib.pyplot as plt

from lib_tp1_ex1 import *

# Implement MDP

nx = 3
na = 2

P = np.empty((nx, na, nx))
P[:, 0, :] = [
    [0.45, 0.00, 0.55],
    [0.00, 0.00, 1.00],
    [0.60, 0.00, 0.40]
]

P[:, 1, :] = [
    [0.00, 0.00, 1.00],
    [0.50, 0.40, 0.10],
    [0.00, 0.90, 0.10]
]

R = np.array([
    [-0.40, 0.00],
    [ 2.00, 0.00],
    [ -1.00, -0.50]
])

# Run value iteration
VI = ValueIteration(P, R, eps=0.01, history=True)

print("Value iteration :")
print("  iterations =", VI.N)
print("  V          =", VI.V)
print("  pi         =", VI.pi)
print("")


pi_star = np.array([1, 0, 1])
Veval = EvaluatePolicy(P, R, pi_star)


# Evaluate policy
print("Evaluation of policy :", pi_star)
print("  V          =", Veval)
print("")

D_inf = np.max(np.abs(VI.V_hist - Veval), axis=1)
plt.plot(np.arange(VI.N), D_inf)
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
plt.xlabel("Iterations")
plt.ylabel("Infinite distance to optimal value")
plt.tight_layout()
plt.savefig("../imgs/VI_convergence.pdf")
plt.show()

# Policy iteration
PI = PolicyIteration(P, R)

print("Policy iteration :")
print("  iterations =", PI.N)
print("  pi         =", PI.pi)

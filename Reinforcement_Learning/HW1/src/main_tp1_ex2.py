from gridworld import GridWorld1
import gridrender as gui
import matplotlib.pyplot as plt
import numpy as np
import time

from lib_tp1_ex2 import *

env = GridWorld1

################################################################################
# Here I define the policy that goes right if possible, otherwise up
################################################################################

policy_actions = np.zeros(env.n_states, dtype=int)
for i, s in enumerate(env.state_actions):
    policy_actions[i] = 0 if 0 in s else 3
# gui.render_policy(env, policy_actions, savefile="../imgs/policy_right_or_up.ps")
gui.render_policy(env, policy_actions)

################################################################################
# Given values of v_pi and q_pi for this policy
################################################################################
v_q4 = [ 0.87691855,  0.92820033,  0.98817903,
         0.00000000,  0.67106071, -0.99447514,
         0.00000000, -0.82847001, -0.87691855,
        -0.93358351, -0.99447514]

q_q4 = [[0.87691855, 0.65706417],
        [0.92820033, 0.84364237],
        [0.98817903, -0.75639924, 0.89361129],
        [0.00000000],
        [-0.62503460, 0.67106071],
        [-0.99447514, -0.70433689, 0.75620264],
        [0.00000000],
        [-0.82847001, 0.49505225],
        [-0.87691855, -0.79703229],
        [-0.93358351, -0.84424050, -0.93896668],
        [-0.89268904, -0.99447514]]

N = 20000
EP = EvaluatePolicy(env, policy_actions, N, 100, history=True)

# Format Q in the right form for the function render_q
fmt_Q = [EP.Q[i, s].tolist() for i, s in enumerate(env.state_actions)]

# gui.render_q(env, fmt_Q, savefile="../imgs/q_policy.ps")
# gui.render_q(env, q_q4, savefile="../imgs/q_policy_ref.ps")
gui.render_q(env, fmt_Q)
gui.render_q(env, q_q4)

# Calculate V from Q
V = EP.Q_hist[:, np.arange(env.n_states), policy_actions]
J = (np.sum(V, axis=1) - np.sum(v_q4)) / env.n_states

# Plot J_k - J_pi
ids = np.logical_not(np.isnan(J))
X = np.arange(len(J))[ids]
Y = J[ids]
plt.plot(X, Y)
plt.plot([0, len(J)], [0, 0])
plt.tight_layout()
plt.xlabel("Number of iterations k")
plt.ylabel("J_k - J_pi")
# plt.savefig("../imgs/j_" + str(N) + ".pdf")
plt.show()

################################################################################
# Q5
################################################################################
v_opt = [0.87691855, 0.92820033, 0.98817903,
         0.00000000, 0.82369294, 0.92820033,
         0.00000000, 0.77818504, 0.82369294,
         0.87691855, 0.82847001]

N = 5000
for eps in [0.03, 0.1, 0.2, 0.4, 0.8]:
        PO = PolicyOptimization(env, N, eps, history=True)

        V_hist = np.max(PO.Q_hist, axis=2)
        D_hist = np.max(np.abs(V_hist - v_opt), axis=1)

        # Euclidian norm
        # D_hist = V_hist - v_opt
        # D_hist = np.sum(D_hist * D_hist, axis=1)

        plt.plot(np.arange(N), D_hist, label="eps = " + str(eps))

plt.xlabel("Number of iterations k")
plt.ylabel("V_k - V*")
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig("../imgs/comp_eps_" + str(N) + ".pdf")
plt.show()


N = 200
for eps in [0.03, 0.1, 0.2, 0.4, 0.8]:
        PO = PolicyOptimization(env, N, eps, rewards_history=True)
        plt.plot(np.arange(N), np.cumsum(PO.rewards), label="eps = " + str(eps))

plt.xlabel("Number of iterations k")
plt.ylabel("Cumulated reward")
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig("../imgs/cum_rewards_" + str(N) + ".pdf")
plt.show()


################################################################################
# Final policy
################################################################################
N = 20000
PO = PolicyOptimization(env, N, 0.2, rewards_history=True)
pi = np.argmax(PO.Q, axis=1)
V =  np.max(PO.Q, axis=1)

gui.render_policy(env, pi)

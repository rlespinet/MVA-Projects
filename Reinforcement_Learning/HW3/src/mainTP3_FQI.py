import numpy as np
import lqg1d
from fqi import FQI
import matplotlib.pyplot as plt
import collect_episodes
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from utils import collect_episodes, estimate_performance


env = lqg1d.LQG1D(initial_state_type='random')
discount = 0.9
horizon = 50

actions = discrete_actions = np.linspace(-8, 8, 20)


#################################################################
# Show the optimal Q-function
#################################################################
def make_grid(x, y):
    m = np.meshgrid(x, y, copy=False, indexing='ij')
    return np.vstack(m).reshape(2, -1).T

states = discrete_states = np.linspace(-10, 10, 20)
SA = make_grid(states, actions)
S, A = SA[:, 0], SA[:, 1]

K, cov = env.computeOptimalK(discount), 0.001
print('Optimal K: {} Covariance S: {}'.format(K, cov))

Q_fun_ = np.vectorize(lambda s, a: env.computeQFunction(s, a, K, cov, discount, 1))
Q_fun = lambda X: Q_fun_(X[:, 0], X[:, 1])

Q_opt = Q_fun(SA)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(S, A, Q_opt)

plt.show()

#################################################################
# Collect the samples using the behavioural policy
#################################################################
# You should use discrete actions
beh_policy =

dataset = collect_episodes(env, n_episodes=100,
                                            policy=beh_policy, horizon=horizon)

# define FQI
# to evaluate the policy you can use estimate_performance

# plot obtained Q-function against the true one


J = estimate_performance(env, policy=fqi, horizon=100, n_episodes=500, gamma=discount)
print('Policy performance: {}'.format(J))
plt.show()

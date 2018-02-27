import numpy as np
import lqg1d
import matplotlib.pyplot as plt
import utils

class ConstantStep(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, gt):
        return self.learning_rate * gt

#####################################################
# Define the environment and the policy
#####################################################
env = lqg1d.LQG1D(initial_state_type='random')

policy =

#####################################################
# Experiments parameters
#####################################################
# We will collect N trajectories per iteration
N = 100
# Each trajectory will have at most T time steps
T = 100
# Number of policy parameters updates
n_itr = 100
# Set the discount factor for the problem
discount = 0.9
# Learning rate for the gradient update
learning_rate = 0.1


#####################################################
# define the update rule (stepper)
stepper =  # e.g., constant, adam or anything you want

# fill the following part of the code with
#  - REINFORCE estimate i.e. gradient estimate
#  - update of policy parameters using the steppers
#  - average performance per iteration
#  - distance between optimal mean parameter and the one at it k
mean_parameters = []
avg_return = []
for _ in range(n_itr):

    paths = utils.collect_episodes(env, policy=policy, horizon=T, n_episodes=N)


# plot the average return obtained by simulating the policy
# at each iteration of the algorithm (this is a rought estimate
# of the performance
plt.figure()
plt.plot(avg_return)

# plot the distance mean parameter
# of iteration k
plt.figure()
plt.plot(mean_parameters)
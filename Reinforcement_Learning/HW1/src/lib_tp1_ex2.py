import numpy as np
import math
from pdb import set_trace

class EvaluatePolicy:

    def __init__(self, env, policy, n = 1000, Tmax = 100, discount = 0.95, history=False):
        # Q and N are |X| * |A| arrays
        Q = np.zeros((env.n_states, len(env.action_names)))
        N = np.zeros((env.n_states, len(env.action_names)))

        if history:
            self.Q_hist = np.empty((n, env.n_states, len(env.action_names)))

        # set_trace()
        for e in range(n):
            current = init_state = env.reset()
            action = init_action = np.random.choice(env.state_actions[current])

            # Cumulated reward
            R = 0.0
            cum_discount = 1.0
            for t in range(Tmax):
                next, reward, term = env.step(current, action)
                R += reward * cum_discount
                cum_discount *= discount
                if term:
                    break
                current = next
                action = policy[current]


            Q[init_state, init_action] += R
            N[init_state, init_action] += 1
            # print(Q / N)

            if history:
                # This takes care of division by 0, but in
                # fact we could just do Q / N, and 0 / 0 values
                # would be put to NaN which is what we want
                new_Q = np.where(N != 0, Q, np.nan) / N
                self.Q_hist[e, :, :] = new_Q

        self.Q = np.where(N != 0, Q, np.nan) / N



class PolicyOptimization:

    def __init__(self, env, n, epsilon = 0.1, Tmax = 1000, gamma = 0.95, history=False, rewards_history=False):

        # Initialize the action that are not possible to - infinity so
        # that they are not selected by the argmax later
        possible_action = np.zeros((env.n_states, len(env.action_names)), dtype=bool)
        for i, s in enumerate(env.state_actions):
            possible_action[i, s] = True

        Q = np.zeros((env.n_states, len(env.action_names)))
        Q[~possible_action] = -math.inf

        if history:
            Q_hist = np.empty((n, env.n_states, len(env.action_names)))

        if rewards_history:
            rewards = np.zeros(n)

        N = np.ones((env.n_states, len(env.action_names)))

        for e in range(n):

            current = env.reset()

            # Precompute the bernoulli law to choose between
            # exploration and exploitation
            p = np.random.binomial(1, 1.0 - epsilon, Tmax)

            for t in range(Tmax):

                # Take action
                if p[t]:
                    # Eploitation
                    action = np.argmax(Q[current, :])
                else:
                    # Exploration
                    action = np.random.choice(env.state_actions[current])

                next, reward, term = env.step(current, action)
                td = reward + gamma * np.max(Q[next, :]) - Q[current, action]
                Q[current, action] += 1.0 / N[current, action] * td

                N[current, action] += 1

                if rewards_history:
                    rewards[e] += reward

                if term:
                    break;

                current = next

            if history:
                Q_hist[e, :, :] = Q


        if rewards_history:
            self.rewards = rewards

        if history:
            self.Q_hist = Q_hist

        self.Q = Q

import numpy as np


def EvaluatePolicy(P, R, pi, discount = 0.95):
    nx, na = R.shape

    R_pi = R[np.arange(len(pi)), pi]
    P_pi = P[np.arange(len(pi)), pi, :]

    return np.linalg.solve(np.identity(nx) - discount * P_pi, R_pi)

class ValueIteration:

    def __init__(self, P, R, eps = 0.01, discount = 0.95, max_iter=500, history=False):
        nx, na = R.shape
        V0 = np.zeros(nx)
        V_hist = []
        for i in range(max_iter):
            V1 = np.max(R + discount * P.dot(V0), axis=1)

            if history:
                V_hist.append(V1)

            if (np.max(V1 - V0) < eps):
                break
            V0 = V1

        self.N = i + 1
        self.V_hist = np.array(V_hist)
        self.V = V1
        self.pi = np.argmax(R + discount * P.dot(V1), axis=1)

class PolicyIteration:

    def __init__(self, P, R, discount = 0.95, max_iter=500):
        nx, na = R.shape
        pi0 = np.zeros(nx, dtype=int)
        for i in range(max_iter):
            V = EvaluatePolicy(P, R, pi0, discount)
            pi1 = np.argmax(R + discount * P.dot(V), axis=1)
            if (np.array_equal(pi0, pi1)):
                break
            pi0 = pi1

        self.pi = pi1
        self.N = i + 1

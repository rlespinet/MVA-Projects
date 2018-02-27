import numpy as np
import matplotlib.pyplot as plt


mu0 = np.array([1.5, 0])
mu1 = np.array([-1.5, 0])


def rot(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

sigma0 = rot(np.pi / 4).dot(np.array([[4, 0], [0, 1.5]])).dot(rot(-np.pi / 4))
sigma1 = rot(np.pi / 12).dot(np.array([[0.07, 0], [0, 4.5]])).dot(rot(-np.pi / 12))


data0 = np.random.multivariate_normal(mu0, sigma0, 5*400)
data1 = np.random.multivariate_normal(mu1, sigma1, 5*200)

plt.plot(data0[:, 0], data0[:, 1], "xb")
plt.plot(data1[:, 0], data1[:, 1], "or")

plt.show()

N0 = data0.shape[0]
N1 = data1.shape[0]

p = 0.01

data0 = np.c_[data0, np.random.binomial(1, p, N0)]
data1 = np.c_[data1, np.random.binomial(1, 1 - p, N1)]

data = np.vstack((data0, data1))

np.savetxt("data/classificationE.train", data, "%8f", "\t")

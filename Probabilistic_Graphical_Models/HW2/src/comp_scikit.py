import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.axes as axis
import time
import clustering
from sklearn import mixture

train = np.loadtxt("data/EMGaussian.data")
test = np.loadtxt("data/EMGaussian.test")

K = 4
d = 2

method = clustering.full_EM()
gmix = mixture.GaussianMixture(n_components=K, covariance_type='full', init_params='kmeans', tol=1e-5)

n_iter = 0
start = time.perf_counter()
for i in range(500):
    gmix.fit(train)
    n_iter += gmix.n_iter_
end = time.perf_counter()
print(end - start)
print("iters :", n_iter)

n_iter = 0
start = time.perf_counter()
for i in range(500):
    method.train(train, K, 1e-5)
    n_iter += method.n_iter
end = time.perf_counter()
print(end - start)
print("iters :", n_iter)


print(gmix.lower_bound_)
print(method.E / 500)

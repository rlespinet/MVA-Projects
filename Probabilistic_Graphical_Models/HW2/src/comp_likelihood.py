import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axis

import clustering

train = np.loadtxt("data/EMGaussian.data")
test = np.loadtxt("data/EMGaussian.test")

K = 4
d = 2

kmeans = clustering.Kmean()
kmeanspp = clustering.Kmeanspp()
spherical = clustering.spherical_EM()
full = clustering.full_EM()

R = 500

kmeansD_train = np.empty(R)
kmeansppD_train = np.empty(R)
sphericalL_train = np.empty(R)
fullL_train = np.empty(R)

kmeansD_test = np.empty(R)
kmeansppD_test = np.empty(R)
sphericalL_test = np.empty(R)
fullL_test = np.empty(R)

for i in range(R):
    kmeansD_train[i] = kmeans.train(train, K, 1e-3)
    kmeansppD_train[i] = kmeanspp.train(train, K, 1e-3)
    sphericalL_train[i] = spherical.train(train, K, 1e-3)
    fullL_train[i] = full.train(train, K, 1e-3)

    kmeansD_test[i] = kmeans.compute_distortion(test)
    kmeansppD_test[i] = kmeanspp.compute_distortion(test)
    sphericalL_test[i] = spherical.compute_likelihood(test)
    fullL_test[i] = full.compute_likelihood(test)


print("Kmeans distortion (train)    # mean :", np.mean(kmeansD_train))
print("                             # std  :", np.std(kmeansD_train))
print("Kmeans++ distortion (train)  # mean :", np.mean(kmeansppD_train))
print("                             # std  :", np.std(kmeansppD_train))
print("Spherical likelihood (train) # mean :", np.mean(sphericalL_train))
print("                             # std  :", np.std(sphericalL_train))
print("Full likelihood (train)      # mean :", np.mean(fullL_train))
print("                             # std  :", np.std(fullL_train))

print("Kmeans distortion (test)     # mean :", np.mean(kmeansD_test))
print("                             # std  :", np.std(kmeansD_test))
print("Kmeans++ distortion (test)   # mean :", np.mean(kmeansppD_test))
print("                             # std  :", np.std(kmeansppD_test))
print("Spherical likelihood (test)  # mean :", np.mean(sphericalL_test))
print("                             # std  :", np.std(sphericalL_test))
print("Full likelihood (test)       # mean :", np.mean(fullL_test))
print("                             # std  :", np.std(fullL_test))

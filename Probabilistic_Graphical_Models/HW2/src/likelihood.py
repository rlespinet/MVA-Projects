import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axis

import clustering

train = np.loadtxt("data/EMGaussian.data")
test = np.loadtxt("data/EMGaussian.test")

K = 4
d = 2

method = clustering.full_EM()
E = method.train(train, K, 1e-6, all_E = True)

plt.plot(E)

print("n_iter", method.n_iter)
print("likelihood", method.E)

plt.xlabel("Number of iterations")
plt.ylabel("Incomplete log likelihood")

plt.tight_layout()

plt.savefig('../imgs/tmp.pdf')
plt.show()

# method = clustering.Kmean()
# J = method.train(train, K, 1e-6, all_J = True)

# plt.plot(J)

# print("n_iter", method.n_iter)
# print("distortion", method.J)


# plt.xlabel("Number of iterations")
# plt.ylabel("Distortion")

# plt.tight_layout()

# plt.savefig('../imgs/tmp.pdf')
# plt.show()

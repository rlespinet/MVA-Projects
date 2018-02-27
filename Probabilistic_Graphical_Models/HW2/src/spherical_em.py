import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axis
from matplotlib.patches import Circle
from scipy.stats import chi2

import clustering

train = np.loadtxt("data/EMGaussian.data")
test = np.loadtxt("data/EMGaussian.test")

K = 4
d = 2

S = chi2.isf(0.1, d)

# E = 0
# while E > -2720:
#     method = clustering.spherical_EM()
#     method.train(train, K, 1e-6)
#     E = method.E

method = clustering.spherical_EM()
method.train(train, K, 1e-6)

print("n_iter", method.n_iter)
print("likelihood", method.E)

plt.axis('equal')

xmin, ymin = np.min(train, axis=0)
xmax, ymax = np.max(train, axis=0)

h = 0.55 * max(xmax - xmin, ymax - ymin)
midx = 0.5 * (xmin + xmax)
midy = 0.5 * (ymin + ymax)

plt.xlim(midx - h, midx + h)
plt.ylim(midy - h, midy + h)

clusters = method.predict(train)

for i in range(K):
    a = plt.scatter(*train[clusters == i].T, 5)
    circle = Circle(method.mu[i], np.sqrt(S * method.var[i]))
    plt.gca().add_artist(circle)
    circle.set_alpha(0.3)
    circle.set_facecolor(*a.get_facecolor())


plt.scatter(*method.mu.T, 30, color="black")
plt.tight_layout()
plt.savefig('../imgs/sphericalEM4.pdf')
plt.show()

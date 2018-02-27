import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axis
from matplotlib.patches import Ellipse
from scipy.stats import chi2

import clustering

train = np.loadtxt("data/EMGaussian.data")
test = np.loadtxt("data/EMGaussian.test")

K = 4
d = 2

S = chi2.isf(0.1, d)

method = clustering.full_EM()
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
    a = plt.scatter(*train[clusters == i].T, 4)
    (l, v) = np.linalg.eigh(method.var[i])
    angle = np.arctan2(v[0, 1], v[0, 0])
    ellipse = Ellipse(method.mu[i], *np.sqrt(2 * S * l), angle * 180.0 / np.pi)
    plt.gca().add_artist(ellipse)
    ellipse.set_alpha(0.5)
    ellipse.set_facecolor(*a.get_facecolor())

plt.scatter(*method.mu.T)
plt.tight_layout()
plt.savefig('../imgs/fullEM4.pdf')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axis

import clustering
import voronoi

train = np.loadtxt("data/EMGaussian.data")
test = np.loadtxt("data/EMGaussian.test")

K = 4

method = clustering.Kmean()
J = method.train(train, K, 0.001)

print("n_iter", method.n_iter)
print("distortion", J)

# compute Voronoi tesselation
regions, vertices = voronoi.polygons_2d(method.mu, radius=50)

# colorize
for region in regions:
    polygon = vertices[region]
    plt.fill(*zip(*polygon), alpha=0.3)

plt.axis('equal')

xmin, ymin = np.min(train, axis=0)
xmax, ymax = np.max(train, axis=0)

h = 0.55 * max(xmax - xmin, ymax - ymin)
midx = 0.5 * (xmin + xmax)
midy = 0.5 * (ymin + ymax)

plt.xlim(midx - h, midx + h)
plt.ylim(midy - h, midy + h)

clusters = method.predict(train)

# Reset color cycle
plt.gca().set_prop_cycle(None)
for i in range(K):
    plt.scatter(*train[clusters == i].T, 5)

plt.scatter(*method.mu.T, 30, color="black")

plt.tight_layout()
plt.savefig('../imgs/kmeans4.pdf')
plt.show()

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.axes as axis
from matplotlib.patches import Circle
from scipy.stats import chi2

import clustering

train = np.loadtxt("data/EMGaussian.data")
test = np.loadtxt("data/EMGaussian.test")

K = 4
d = 2

R = 5000

iters=np.zeros(R)
likelihood=np.zeros(R)

name = "full"
method = clustering.full_EM()

for i in range(R):
    method.train(train, K, 1e-4)
    iters[i] = method.n_iter
    likelihood[i] = method.E


def plot_histogram(values, xlabel, ylabel, nbins = 25, savefile = None, color='blue', annotate = False):

    plt.figure(1)
    weights = np.ones_like(iters)/len(values)
    counts, bins, patches = plt.hist(values, nbins , weights=weights,
                                     alpha=0.75, facecolor=color, edgecolor='black', linewidth=0.8)

    if (annotate):
        bin_centers = 0.5 * np.diff(bins) + bins[:-1]
        for count, x in zip(counts[counts > 0.01], bin_centers[counts > 0.01]):
            frequency = '%0.0f%%' % (100 * count)
            plt.gca().annotate(frequency, xy=(x, count), xycoords=('data'),
                               color="k", xytext=(0, 4), textcoords='offset points',
                               bbox=dict(boxstyle="round", fc="w", alpha=0.8),
                               va='bottom', ha='center', size=8)

    for b, c in zip(bins, counts):
        print(b, "\t", c)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    if (savefile != None):
        plt.savefig(savefile)

    plt.show()


print("Number of iterations")
print("****************************************")
plot_histogram(iters, "Number of iterations", "Frequency", 20, "../imgs/" + name + "EM4_iterations_hist.pdf", 'blue')

print("Incomplete likelihood")
print("****************************************")
plot_histogram(likelihood, "Incomplete likelihood", "Frequency", 25, "../imgs/" + name + "EM4_likelihood_hist.pdf", 'red')

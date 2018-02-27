import numpy as np
import math
import random

# For debugging purposes
# np.random.seed(5)

class Kmean:
    def __init__(self):
        pass

    def train_init(self, X, K):
        (N, d) = X.shape;
        ids = np.random.choice(N, K, replace=False)
        return X[ids]

    def compute_distortion(self, X):
        sqdiff = X[:, np.newaxis, :] - self.mu[np.newaxis, :, :]
        sqdiff = np.sum(sqdiff * sqdiff, axis=2)
        return sqdiff.min(axis=1).sum()


    def train(self, X, K, e = 0.1, all_J = False):
        (N, d) = X.shape;
        J = []

        # Step 0
        mu0 = self.train_init(X, K)

        maxIt = 100
        for i in range(maxIt):
            # Step 1
            sqdiff = X[:, np.newaxis, :] - mu0[np.newaxis, :, :]
            sqdiff = np.sum(sqdiff * sqdiff, axis=2)

            # Compute the distortion if asked in the parameters
            if all_J:
                J.append(sqdiff.min(axis=1).sum())

            Z = np.zeros((N, K))
            Z[np.arange(N), sqdiff.argmin(axis=1)] = 1

            # Step 2
            mu1 = Z.T.dot(X) / Z.sum(axis = 0)[:,np.newaxis]

            # Step 3
            if (np.max(mu1 - mu0) < e):
                break
            mu0 = mu1

        self.n_iter = i
        self.K = K
        self.mu = mu0
        self.J = self.compute_distortion(X)

        if all_J:
            J.append(self.J)

        return J if all_J else self.J


    def predict(self, X):
            sqdiff = X[:, np.newaxis, :] - self.mu[np.newaxis, :, :]
            sqdiff = np.add.reduce(sqdiff * sqdiff, axis=2)

            return sqdiff.argmin(axis=1);

class Kmeanspp(Kmean):

    def train_init(self, X, K):
        (N, d) = X.shape
        P = np.empty((K, d))
        P[0] = X[np.random.randint(0, N)]

        D = math.inf * np.ones(N)

        for i in range(K - 1):
            T = X - P[i]
            D = np.minimum(D, np.sum(T * T, axis=1))

            W = D.cumsum()

            z = np.random.uniform(0, W[-1])
            j = np.searchsorted(W > z, True)
            P[i + 1] = X[j]

        return P


class spherical_EM:
    def __init__(self):
        pass

    def train_init(self, X, K):
        (N, d) = X.shape;

        init = Kmean()
        init.train(X, K)

        mu = init.mu

        pi = np.bincount(init.predict(X)) / N

        sqdiff = X[:, np.newaxis, :] - mu[np.newaxis, :, :]
        sqdiff = np.add.reduce(sqdiff * sqdiff, axis=2)

        mask = np.zeros_like(sqdiff)
        mask[np.arange(N), np.argmin(sqdiff, axis=1)] = 1

        var = np.sum(sqdiff * mask, axis=0) / np.sum(mask, axis=0)

        return (mu, pi, var)

    def compute_likelihood(self, X):
        N, d = X.shape
        sqdiff = X[:, np.newaxis, :] - self.mu[np.newaxis, :, :]
        sqdiff = np.sum(sqdiff * sqdiff, axis=2)
        L = np.exp(-0.5 * sqdiff / self.var).dot(self.pi / (self.var ** (0.5*d)))
        L = sum(np.log(L)) - N * 0.5 * d * np.log(2.0 * math.pi)
        return L

    def train(self, X, K, e, all_E = False):
        (N, d) = X.shape;

        (mu0, pi0, var0) = self.train_init(X, K)

        E = []

        E0 = math.inf

        maxIt = 200
        for n in range(maxIt):
            sqdiff0 = X[:, np.newaxis, :] - mu0[np.newaxis, :, :]
            sqdiff0 = np.add.reduce(sqdiff0 * sqdiff0, axis=2)

            T = pi0 / var0 * np.exp(-0.5 * sqdiff0 / var0)
            T /= np.sum(T, axis=1)[:, np.newaxis]

            pi1 = T.sum(axis=0)
            pi1 /= N

            mu1 = np.sum(T[:, :, np.newaxis] * X[:, np.newaxis, :], axis=0)
            mu1 /= T.sum(axis=0)[:, np.newaxis]

            sqdiff1 = X[:, np.newaxis, :] - mu1[np.newaxis, :, :]
            sqdiff1 = np.sum(sqdiff1 * sqdiff1, axis=2)

            var1 = np.sum(sqdiff1 * T, axis=0)
            var1 /= d * T.sum(axis=0)

            # Complete likelihood
            # E1 = np.sum(T.dot(np.log(pi1))
            #       - 0.5 * d * T.dot(np.log(var1))
            #       - 0.5 * np.add.reduce(T * sqdiff1 / var1, axis=1))

            # Likelihood
            E1 = np.exp(-0.5 * sqdiff1 / var1).dot(pi1 / (var1 ** (0.5*d)))
            E1 = sum(np.log(E1)) - N * 0.5 * d * np.log(2.0 * math.pi)

            if all_E:
                E.append(E1)

            if (np.abs(E1 - E0) < N * e):
                break

            (mu0, pi0, var0, E0) = (mu1, pi1, var1, E1)

        self.n_iter = n
        self.mu = mu1
        self.pi = pi1
        self.var = var1
        self.E = E1

        return E if all_E else self.E

    def predict(self, X):

        sqdiff = X[:, np.newaxis, :] - self.mu[np.newaxis, :, :]
        sqdiff = np.add.reduce(sqdiff * sqdiff, axis=2)

        T = self.pi / self.var * np.exp(-0.5 * sqdiff / self.var)

        return T.argmax(axis=1)


class full_EM:
    def __init__(self):
        pass

    def train_init(self, X, K):
        (N, d) = X.shape;

        init = Kmean()
        init.train(X, K)

        mu = init.mu

        pi = np.bincount(init.predict(X)) / N

        sqdiff = X[:, np.newaxis, :] - mu[np.newaxis, :, :]
        sqdiff = np.add.reduce(sqdiff * sqdiff, axis=2)

        mask = np.zeros_like(sqdiff)
        mask[np.arange(N), np.argmin(sqdiff, axis=1)] = 1

        #var = np.sum(sqdiff * mask, axis=0) / np.sum(mask, axis=0)

        cov = (X[:, np.newaxis, :] - mu[np.newaxis, :, :]) * mask[:,:,np.newaxis]
        cov = cov[:,:,:, np.newaxis] * cov[:,:,np.newaxis, :]
        cov = cov.sum(axis=0) / mask.sum(axis=0)[:, np.newaxis, np.newaxis]

        return (mu, pi, cov)

    def compute_likelihood(self, X):
        N, d = X.shape
        w = X[:, np.newaxis, :] - self.mu[np.newaxis, :, :]
        w = w[:,:,:, np.newaxis] * np.linalg.inv(self.var)[np.newaxis, :, :, :] * w[:,:,np.newaxis, :]
        w = np.add.reduce(w, axis=(2, 3))
        L = np.exp(-0.5 * w).dot(self.pi / np.sqrt(np.linalg.det(self.var)))
        L = sum(np.log(L)) - N * 0.5 * d * np.log(2.0 * math.pi)
        return L

    def train(self, X, K, e, all_E = False):
        (N, d) = X.shape;

        (mu0, pi0, var0) = self.train_init(X, K)

        E = []

        E0 = math.inf

        maxIt = 100
        for n in range(maxIt):

            sqdiff0 = X[:, np.newaxis, :] - mu0[np.newaxis, :, :]
            sqdiff0 = sqdiff0[:,:,:, np.newaxis] * np.linalg.inv(var0)[np.newaxis, :, :, :] * sqdiff0[:,:,np.newaxis, :]
            sqdiff0 = np.add.reduce(sqdiff0, axis=(2, 3))

            T = pi0 / np.sqrt(np.linalg.det(var0)) * np.exp(-0.5 * sqdiff0)
            T /= np.sum(T, axis=1)[:, np.newaxis]

            pi1 = T.sum(axis=0)
            pi1 /= N

            mu1 = np.sum(T[:, :, np.newaxis] * X[:, np.newaxis, :], axis=0)
            mu1 /= T.sum(axis=0)[:, np.newaxis]

            sqdiff1 = X[:, np.newaxis, :] - mu1[np.newaxis, :, :]
            sqdiff1 = sqdiff1[:, :, :, np.newaxis] * sqdiff1[:, :, np.newaxis, :]

            var1 = np.sum(sqdiff1 * T[:,:,np.newaxis, np.newaxis], axis=0)
            var1 /= T.sum(axis=0)[:, np.newaxis, np.newaxis]

            w = X[:, np.newaxis, :] - mu1[np.newaxis, :, :]
            w = w[:,:,:, np.newaxis] * np.linalg.inv(var1)[np.newaxis, :, :, :] * w[:,:,np.newaxis, :]
            w = np.add.reduce(w, axis=(2, 3))

            # Complete likelihood
            # E1 = np.sum(T.dot(np.log(pi1))
            #       - 0.5 * T.dot(np.log(np.linalg.det(var1)))
            #       - 0.5 * np.sum(T * w, axis=1))

            # Likelihood
            E1 = np.exp(-0.5 * w).dot(pi1 / np.sqrt(np.linalg.det(var1)))
            E1 = sum(np.log(E1)) - N * 0.5 * d * np.log(2.0 * math.pi)

            if all_E:
                E.append(E1)

            if (np.abs(E1 - E0) < N * e):
                break

            (mu0, pi0, var0, E0) = (mu1, pi1, var1, E1)

        self.n_iter = n
        self.mu = mu1
        self.pi = pi1
        self.var = var1
        self.E = E1

        return E if all_E else self.E

    def predict(self, X):

        sqdiff = X[:, np.newaxis, :] - self.mu[np.newaxis, :, :]
        sqdiff = sqdiff[:,:,:, np.newaxis] * np.linalg.inv(self.var)[np.newaxis, :, :] * sqdiff[:,:,np.newaxis, :]
        sqdiff = np.add.reduce(sqdiff, axis=(2, 3))

        T = self.pi / np.sqrt(np.linalg.det(self.var)) * np.exp(-0.5 * sqdiff)

        return T.argmax(axis=1)

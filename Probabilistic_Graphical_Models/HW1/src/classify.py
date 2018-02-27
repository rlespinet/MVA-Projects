import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from math import inf

# Author : Rémi Lespinet
#
# Description :
# This script trains all classifiers (LDA, QDA, Linear Regression and
# Logistic Regression) on the files data/{classificationA,
# classificationB, classificationC}.train compute and print the
# parameters, and test the trained classifier on the files
# data/{classificationA, classificationB, classificationC}.test
# it then computes and outputs the train misclassification error
# and the test misclassification error. It also save graphs under
# imgs/ showing for each classifier and each file, the
# data sample and the decision boundary.


# def get_line_points(W, Xs, Ys):
#     if (W[0]):
#         return [(- W[1] * Ys[0] - W[2]) / W[0], (- W[1] * Ys[1] - W[2]) / W[0]], [Ys[0], Ys[1]]
#     elif (W[1]):
#         return [Xs[0], Xs[1]],[(- W[0] * Xs[0] - W[2]) / W[1], (- W[0] * Xs[1] - W[2]) / W[1]]


class ClassificationModel:

    def classify(self, X):
        P = np.apply_along_axis(self.probability, 1, X)
        return np.where(np.less(P, 0.5), 0, 1)

    def contour(self, Xs, Ys):
        steps = 100
        x = np.arange(Xs[0], Xs[1], (Xs[1] - Xs[0]) / steps)
        y = np.arange(Ys[0], Ys[1], (Ys[1] - Ys[0]) / steps)

        X, Y = np.meshgrid(x, y)
        U = np.dstack(np.meshgrid(x, y))
        Z = np.apply_along_axis(self.probability, 2, U)

        CS = plt.contourf(X, Y, Z, [-inf, 0.5, inf], colors=("lightsalmon", "cornflowerblue"))
        plt.contour(X, Y, Z, [0.5])

class LDA(ClassificationModel):

    name = "Linear Discriminant Analysis (LDA)"
    short = "LDA"

    def __init__(self):
        pass

    def train(self, X, Y):
        X0 = X[Y == 0]
        X1 = X[Y == 1]
        self.pi = np.mean(Y)
        self.mu0 = np.mean(X0, axis=0)
        self.mu1 = np.mean(X1, axis=0)
        self.sigma = np.cov(np.r_[X0 - self.mu0, X1 - self.mu1].T)
        self.sigmainv = np.linalg.inv(self.sigma)

    def probability(self, X):
        X0 = (X - self.mu0)
        X1 = (X - self.mu1)
        E = (1 - self.pi) / self.pi * np.exp(-0.5 * X0.dot(self.sigmainv).dot(X0) + 0.5 * X1.dot(self.sigmainv).dot(X1) )
        return 1 / (1 + E)

    def print_params(self):
        print("pi:", self.pi)
        print("mu0:", self.mu0)
        print("mu1:", self.mu1)
        print("sigma:\n", self.sigma)

    # This was used to plot a line before using the function contour
    # ########################################
    # def line(self, Xs, Ys):
    #     w = (self.mu1 - self.mu0).dot(self.sigmainv)
    #     b = -.5 * (self.mu0.dot(self.sigmainv).dot(self.mu0.T) - self.mu1.dot(self.sigmainv).dot(self.mu1.T)) + np.log(1 - self.pi) - np.log(self.pi)
    #     return get_line_points(np.hstack((w, b)), Xs, Ys)

class QDA(ClassificationModel):

    name = "Quadratic Discriminant Analysis (QDA)"
    short = "QDA"

    def __init__(self):
        pass

    def train(self, X, Y):
        X0 = X[Y == 0]
        X1 = X[Y == 1]
        self.pi = np.mean(Y)
        self.mu0 = np.mean(X0, axis=0)
        self.mu1 = np.mean(X1, axis=0)
        self.sigma0 = np.cov(np.transpose(X0 - self.mu0))
        self.sigma1 = np.cov(np.transpose(X1 - self.mu1))
        self.sigma0detsqrt = np.sqrt(np.linalg.det(self.sigma0))
        self.sigma1detsqrt = np.sqrt(np.linalg.det(self.sigma1))
        self.sigma0inv = np.linalg.inv(self.sigma0)
        self.sigma1inv = np.linalg.inv(self.sigma1)

    def probability(self, X):
        X0 = (X - self.mu0)
        X1 = (X - self.mu1)
        E = (1 - self.pi) * self.sigma1detsqrt / (self.pi * self.sigma0detsqrt) * np.exp(-0.5 * X0.dot(self.sigma0inv).dot(X0) + 0.5 * X1.dot(self.sigma1inv).dot(X1))
        return 1 / (1 + E)

    def print_params(self):
        print("pi:", self.pi)
        print("mu0:", self.mu0)
        print("mu1:", self.mu1)
        print("sigma0:\n", self.sigma0)
        print("sigma1:\n", self.sigma1)


    # This is another way of computing the contour
    # There is no exponential, so it's more precise
    # to find the contour
    #########################################
    # def contour(self, Xs, Ys):

        # L = lambda X, Y : -.5 * ((np.array([X, Y]) - self.mu0).T.dot(self.sigma0inv).dot(np.array([X, Y]) - self.mu0) - (np.array([X, Y]) - self.mu1).T.dot(self.sigma1inv).dot(np.array([X, Y]) - self.mu1)) + np.log(1 - self.pi) - np.log(self.pi) + .5 * np.log(np.linalg.det(self.sigma1) / np.linalg.det(self.sigma0))
        # Z = np.vectorize(L)(X, Y)
        # plt.contour(X, Y, Z, [0])

    def params(self):
        return self.pi, self.mu0, self.mu1, self.sigma


class LogisticRegression(ClassificationModel):

    name = "Logistic Regression"
    short = "LogReg"

    def __init__(self, W0, it, regul):
        self.W0 = W0
        self.it = it
        self.regul = regul

    def train(self, X, Y):
        N = np.shape(X)[0]
        X = np.c_[X, np.ones(N)]
        self.W = self.W0.copy()
        for i in range(self.it):
            Xt = np.transpose(X)
            n = 1 / (1 + np.exp(np.dot(X, self.W)))
            D = np.diag(n * (1 - n))
            H = np.dot(np.dot(Xt, D), X) + np.diag(self.regul*np.ones(3))
            G = np.dot(Xt, Y - n) + self.regul * self.W
            self.W -= np.dot(np.linalg.inv(H), G)


    def probability(self, X):
        X = np.hstack((X, 1))
        return 1 / (1 + np.exp(self.W.dot(X)))

    def print_params(self):
        print("W :", self.W)

    # def line(self, Xs, Ys):
    #     return get_line_points(self.W, Xs, Ys)


class LinearRegression(ClassificationModel):

    name = "Linear Regression"
    short = "LinReg"

    def __init__(self):
        pass

    def train(self, X, Y):
        N = np.shape(X)[0]
        X = np.c_[X, np.ones(N)]
        K = np.linalg.inv(X.T.dot(X))
        self.W = K.dot(X.T).dot(Y)
        E = Y - X.dot(self.W)
        self.sigma = E.dot(E) / len(E)

    def probability(self, X):
        X = np.hstack((X, 1))
        return self.W.dot(X)

    def print_params(self):
        print("M :")
        print(self.W)
        print("sigma^2 : ", self.sigma)

    # def line(self, Xs, Ys):
    #     return get_line_points(self.W + [0, 0, - 0.5], Xs, Ys)



# Core of the script : this part uses the classes
# to compute parameters, misclassification error
# and plot the training and test data (in files)
# ########################################


data_files = ["classificationA",
              "classificationB",
              "classificationC"]

# Load the files once
traindata = []
for data_file in data_files:
    traindata.append(np.loadtxt("data/" + data_file + ".train"))

testdata = []
for data_file in data_files:
    testdata.append(np.loadtxt("data/" + data_file + ".test"))

models = [LDA(),
          LogisticRegression(np.zeros(3), 50, 0.001),
          LinearRegression(),
          QDA()]

for model in models:

    print()
    print("################################################################################")
    print("# Model :", model.name)
    print("################################################################################\n")


    for (id, (datafile, train, test)) in enumerate(zip(data_files, traindata, testdata)):
        X_train = train[:, 0:2]
        Y_train = train[:, 2]

        X_test = test[:, 0:2]
        Y_test = test[:, 2]

        print("TRAINING [", datafile, "]\n")
        model.train(X_train, Y_train)

        Y_predicted = model.classify(X_train)
        E_train = np.mean(np.absolute(Y_predicted - Y_train))

        Y_predicted = model.classify(X_test)
        E_test = np.mean(np.absolute(Y_predicted - Y_test))

        model.print_params()

        print("train_error:", E_train * 100, "%")
        print("test_error:", E_test * 100, "%", "\n")

        m = np.min([np.min(X_test, 0), np.min(X_train, 0)], 0)
        M = np.max([np.max(X_test, 0), np.max(X_train, 0)], 0)

        # Discriminate the data by Y to show different colors in the plot
        X0_test = X_test[Y_test == 0]
        X1_test = X_test[Y_test == 1]

        # Plot test data

        plt.figure(2 * id)
        plt.clf()
        plt.plot(X0_test[:, 0], X0_test[:, 1], 'x', color="darkred")
        plt.plot(X1_test[:, 0], X1_test[:, 1], 'x', color="navy")
        plt.plot([m[0], M[0]], [m[1], M[1]], "None") # Hack to re-scale without having a border bug

        # Show contour
        model.contour(plt.xlim(), plt.ylim())

        savename = model.short + "_" + datafile + "_test.pdf"
        plt.savefig("../imgs/" + savename, bbox_inches='tight')

        # Plot train data
        X0_train = X_train[Y_train == 0]
        X1_train = X_train[Y_train == 1]

        plt.figure(2 * id + 1)
        plt.clf()
        plt.plot(X0_train[:, 0], X0_train[:, 1], 'x', color="darkred")
        plt.plot(X1_train[:, 0], X1_train[:, 1], 'x', color="navy")
        plt.plot([m[0], M[0]], [m[1], M[1]], "None") # Hack to re-scale without having a border bug

        # Show contour
        model.contour(plt.xlim(), plt.ylim())

        savename = model.short + "_" + datafile + "_train.pdf"
        plt.savefig("../imgs/" + savename, bbox_inches='tight')

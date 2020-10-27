import numpy as np
from numpy.linalg import solve
import findMin
from scipy.optimize import approx_fprime
import utils

class logReg:
    # Logistic Regression
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True


    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)

class logRegL1(logReg):
    def __init__(self, lammy=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMinL1(self.funObj, self.w, self.lammy,
                                        self.maxEvals, X, y, verbose=self.verbose)

class logRegL2(logReg):
    def __init__(self, lammy=1, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.lammy = lammy

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + self.lammy / 2 * np.linalg.norm(w, ord=2)

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res) + self.lammy * w

        return f, g



class logRegL0(logReg):
    # L0 Regularized Logistic Regression
    def __init__(self, L0_lambda=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.L0_lambda = L0_lambda
        self.maxEvals = maxEvals

    def fit(self, X, y):
        n, d = X.shape
        minimize = lambda ind: findMin.findMin(self.funObj,
                                                  np.zeros(len(ind)),
                                                  self.maxEvals,
                                                  X[:, ind], y, verbose=0)
        selected = set()
        selected.add(0)
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        while minLoss != oldLoss:
            oldLoss = minLoss
            print("Epoch %d " % len(selected))
            print("Selected feature: %d" % (bestFeature))
            print("Min Loss: %.3f\n" % minLoss)

            for i in range(d):
                if i in selected:
                    continue

                selected_new = selected | {i} # tentatively add feature "i" to the seected set

                # TODO for Q2.3: Fit the model with 'i' added to the features,
                # then compute the loss and update the minLoss/bestFeature


            selected.add(bestFeature)

        self.w = np.zeros(d)
        self.w[list(selected)], _ = minimize(list(selected))


class leastSquaresClassifier:
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve the normal equationsrange
            # with a bit of regularization for numerical reasons
            self.W[i] = np.linalg.solve(X.T@X+0.0001*np.eye(d), X.T@ytmp)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)

class logLinearClassifier:
    def __init__(self, maxEvals=400, verbose=0):
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes, d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y == i] = 1
            ytmp[y != i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i], f = findMin.findMin(self.funObj, self.W[i], self.maxEvals, X, ytmp)

    def predict(self, X):
        return np.argmax(X @ self.W.T, axis=1)

class softmaxClassifier:
    def __init__(self, verbose=1, maxEvals=400):
        self.verbose = verbose
        self.maxEvals = maxEvals

        def funObj(self, w, X, y):
            n, d = X.shape
            W = w.reshape((self.n_classes, d))
            W_y_i = np.zeros((n, d))

            W_y_i[:] = W[y[:]]
            Wyi_X = np.einsum('ij,ij->i', W_y_i, X)

            sumexp = np.sum(np.exp(W @ X.T), axis=0)

            f = np.sum(- Wyi_X + np.log(sumexp))

            g = np.zeros((self.n_classes, d))

            for c in range(self.n_classes):
                for j in range(d):
                    expwx_i = np.exp(W[c].dot(X.T))
                    sumexpwx_i = np.sum(np.exp(W.dot(X.T)), axis=0)
                    indicator = y == c
                    g[c,j] = X[:,j].dot(expwx_i / sumexpwx_i - indicator)

            return f, g.flatten()

        def fit(self, X, y):
            n, d = X.shape
            self.n_classes = np.unique(y).size

            # Initial guess
            self.w = np.zeros(self.n_classes * d)

            utils.check_gradient(self, X, y)
            self.w, f = findMin.findMin(self.funObj, self.w, self.maxEvals, X, y)
            self.W = self.w.reshape((self.n_classes, d))


        def predict(self, X):
            return np.argmax(X @ self.W.T, axis=1)
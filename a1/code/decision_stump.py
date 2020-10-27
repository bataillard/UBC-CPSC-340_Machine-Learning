import numpy as np
import utils


class DecisionStumpEquality:

    def __init__(self):
        pass

    def fit(self, X, y):
        N, D = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.  
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.splitSat = y_mode
        self.splitNot = None
        self.splitVariable = None
        self.splitValue = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)

        # Loop over features looking for the best split
        X = np.round(X)

        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]

                # Find most likely class for each split
                y_sat = utils.mode(y[X[:, d] == value])
                y_not = utils.mode(y[X[:, d] != value])

                # Make predictions
                y_pred = y_sat * np.ones(N)
                y_pred[X[:, d] != value] = y_not

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.splitVariable = d
                    self.splitValue = value
                    self.splitSat = y_sat
                    self.splitNot = y_not

    def predict(self, X):

        M, D = X.shape
        X = np.round(X)

        if self.splitVariable is None:
            return self.splitSat * np.ones(M)

        yhat = np.zeros(M)

        for m in range(M):
            if X[m, self.splitVariable] == self.splitValue:
                yhat[m] = self.splitSat
            else:
                yhat[m] = self.splitNot

        return yhat


class DecisionStumpErrorRate:

    def __init__(self):
        pass

    def fit(self, X, y):
        N, D = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.splitSat = y_mode
        self.splitNot = None
        self.splitVariable = None
        self.splitValue = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)

        # Loop over features looking for the best split

        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]

                # Find most likely class for each split
                y_above = utils.mode(y[X[:, d] > value])
                y_below = utils.mode(y[X[:, d] <= value])

                # Make predictions
                y_hat = np.full(N, y_above)
                y_hat[X[:, d] <= value] = y_below

                # Compute error
                errors = np.sum(y_hat != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.splitVariable = d
                    self.splitValue = value
                    self.splitSat = y_above
                    self.splitNot = y_below

    def predict(self, X):
        M, D = X.shape
        X = np.round(X)

        if self.splitVariable is None:
            return self.splitSat * np.ones(M)

        y_hat = np.zeros(M)

        for m in range(M):
            if X[m, self.splitVariable] > self.splitValue:
                y_hat[m] = self.splitSat
            else:
                y_hat[m] = self.splitNot

        return y_hat


"""
A helper function that computes the entropy of the 
discrete distribution p (stored in a 1D numpy array).
The elements of p should add up to 1.
This function ensures lim p-->0 of p log(p) = 0
which is mathematically true (you can show this with l'Hopital's rule), 
but numerically results in NaN because log(0) returns -Inf.
"""


def entropy(p):
    plogp = 0 * p  # initialize full of zeros
    plogp[p > 0] = p[p > 0] * np.log(p[p > 0])  # only do the computation when p>0
    return -np.sum(plogp)


# This is not required, but one way to simplify the code is 
# to have this class inherit from DecisionStumpErrorRate.
# Which methods (init, fit, predict) do you need to overwrite?
class DecisionStumpInfoGain(DecisionStumpErrorRate):

    def fit(self, X, y):
        N, D = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y, minlength=2)

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.splitSat = y_mode
        self.splitNot = None
        self.splitVariable = None
        self.splitValue = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        max_info_gain = 0

        # Loop over features looking for the best split

        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]

                # Find most likely class for each split
                feat_above = X[:, d] > value
                feat_below = X[:, d] <= value

                y_above = utils.mode(y[feat_above])
                y_below = utils.mode(y[feat_below])

                # Make predictions
                y_hat = np.full(N, y_above)
                y_hat[X[:, d] <= value] = y_below

                # Compute information gain
                freq_yes = np.count_nonzero(feat_above) / N
                freq_no = np.count_nonzero(feat_below) / N

                y_dist = np.bincount(y, minlength=2) / N
                y_yes_dist = np.bincount(y[feat_above], minlength=2) / N
                y_no_dist = np.bincount(y[feat_below], minlength=2) / N

                info_gain = entropy(y_dist) - freq_yes * entropy(y_yes_dist)  - freq_no * entropy(y_no_dist)

                # Compare to max info gain so far
                if info_gain > max_info_gain:
                    # This is the highest information gain, store this value
                    max_info_gain = info_gain
                    self.splitVariable = d
                    self.splitValue = value
                    self.splitSat = y_above
                    self.splitNot = y_below
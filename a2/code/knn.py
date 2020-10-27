"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y 

    def predict(self, Xtest):
        sq_distances = utils.euclidean_dist_squared(self.X, Xtest)
        T, D = Xtest.shape

        y_test = np.zeros(T)

        for t in range(T):
            x_indicies_sorted = np.argsort(sq_distances[:, t])
            k_nearest_indicies = x_indicies_sorted[:self.k]

            y_knn = np.take(self.y, k_nearest_indicies)
            y_test[t] = utils.mode(y_knn)

        return y_test

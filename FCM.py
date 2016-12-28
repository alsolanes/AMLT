import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.spatial.distance import cdist

class FuzzyCMeans(BaseEstimator, ClusterMixin):

    def __init__(self,c=2, m=2, max_iterations=1000, error=0.005, init=None, seed=None):
        self.c= c
        self.m = m
        self.max_iterations = 1000
        self.error = error
        self.init = init
        self.seed = seed

    def dist(self,data, centers):
        return cdist(data, centers).T

    def singleStep(self,data, u, c, m):
        # Normalizing, then eliminating any potential zero values.
        #u /= np.ones((c, 1)).dot(np.atleast_2d(u_previous.sum(axis=0)))
        u = np.fmax(u, np.finfo(np.float64).eps)

        um = u ** m

        data = data.T

        centers = um.dot(data) / (np.ones((data.shape[1], 1)).dot(np.atleast_2d(um.sum(axis=1))).T)

        d = self.dist(data, centers)
        d = np.fmax(d, np.finfo(np.float64).eps)

        u = d** (- 2. / (m - 1))
        u /= np.ones((c, 1)).dot(np.atleast_2d(u.sum(axis=0)))

        return centers, u, d

    def singleStep_predict(self,data, centers, u_previous, c, m):

        # Normalizing, then eliminating any potential zero values.
        #u_previous /= np.ones((c, 1)).dot(np.atleast_2d(u_previous.sum(axis=0)))

        u_previous = np.fmax(u_previous, np.finfo(np.float64).eps)

        um = u_previous ** m
        data = data.T

        # For prediction, we do not recalculate cluster centers. The data is
        # forced to conform to the prior clustering.

        d = self.dist(data, centers)
        d = np.fmax(d, np.finfo(np.float64).eps)

        u = d ** (- 2. / (m - 1))
        u /= np.ones((c, 1)).dot(np.atleast_2d(u.sum(axis=0)))

        return u, d

    def fit(self,data):
        if self.init is None:
            if self.seed is not None:
                np.random.seed(self.seed)
            u0 = np.random.rand(self.c, data.shape[1])
            u0 /= np.ones((self.c, 1)).dot(np.atleast_2d(u0.sum(axis=0))).astype(np.float64)
            self.init = u0.copy()
        u0 = self.init
        u = np.fmax(u0, np.finfo(np.float64).eps)
        p= 1
        while p < self.max_iterations:
            u2 = u.copy()
            centers, u, d = self.singleStep(data, u2, self.c, self.m)
            p += 1
            if np.linalg.norm(u - u2) < self.error:
                break
        # Fuzzy partition Coefficient
        fpc = np.trace(u.dot(u.T)) / float(u.shape[1])

        return centers, u, u0, d, fpc

    def predict(self, data, trained_centers):
        c = trained_centers.shape[0]

        # Setup u0
        if self.init is None:
            if self.seed is not None:
                np.random.seed(self.seed)
            u0 = np.random.rand(c, data.shape[1])
            u0 /= np.ones((c, 1)).dot(np.atleast_2d(u0.sum(axis=0))).astype(np.float64)
            self.init = u0.copy()
        u0 = self.init
        u = np.fmax(u0, np.finfo(np.float64).eps)

        p = 1

        while p < self.max_iterations:
            u2 = u.copy()
            [u, d] = self.singleStep_predict(data, trained_centers, u2, c, self.m)
            p += 1

            if np.linalg.norm(u - u2) < self.error:
                break

        # Fuzzy partition Coefficient
        fpc = np.trace(u.dot(u.T)) / float(u.shape[1])

        return u, u0, d, fpc

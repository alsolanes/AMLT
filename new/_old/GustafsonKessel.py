import matplotlib.patches as shapes
import numpy as np

class GKCluster(object):
    def __init__(self, high, dim, det=1):
        self.r = high/5
        self.v = np.random.uniform(high, size=dim)
        self.det = det
        self.A = np.array([[det**.5, 0], [0, det**.5]])

    def update(self, xs, us, m, ci):
        """
        update its variables based on membership values
        :param xs: list of data points
        :param us: list of memberships for this cluster
        :param m:
        """
        uis = us[:,ci]
        self.v = sum([uis[i]**m * xs[i] for i in range(len(xs))]) / sum([uis[i]**m for i in range(len(xs))])

        S = sum([uis[i]**m * (xs[i]-self.v).reshape((2, 1)).dot((xs[i]-self.v).reshape((2, 1)).T) for i in range(len(xs))])
        S += 0.001 * np.eye(2)
        self.A = self.det * np.linalg.det(S)**(1.0/len(self.v)) * np.linalg.inv(S)

    def distance(self, x):
        return ((x-self.v).T.dot(self.A)).dot(x-self.v)

    def __repr__(self):
        return "GustafsonKessel cluster# v={0} A={1}".format(self.v, self.A)

    def draw(self):
        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:, order]

        vals, vecs = eigsorted(self.A)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        height, width = self.r * np.sqrt(vals)
        res = shapes.Ellipse(xy=self.v, width=width, height=height, angle=theta)
        return res

    def center(self):
        return self.v

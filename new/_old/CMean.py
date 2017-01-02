import matplotlib.patches as shapes
import numpy as np

class CMeanCluster(object):
    def __init__(self, high, dim):
        self.r = np.random.uniform(high/5)
        self.v = np.random.uniform(high, size=dim)


    def update(self, xs, us, m, ci):
        """
        update its variables based on membership values
        :param xs: list of data points
        :param us: list of memberships for all clusteres
        :param m:
        :param ci: cluster num
        """
        uis = us[:,ci]
        self.v = sum([uis[i]**m * xs[i] for i in range(len(xs))]) / sum([uis[i]**m for i in range(len(xs))])
        self.r = sum([uis[i]**m * np.linalg.norm(xs[i]-self.v) for i in range(len(xs))]) / sum([uis[i]**m for i in range(len(xs))])

    def distance(self, x):
        """
        distance from cluster to point x
        usage in membership evaluation
        :param x: data point
        :return: distance**2
        """
        return np.linalg.norm(x-self.v)**2

    def __repr__(self):
        return "CMeans cluster# v={0} r={1}".format(self.v, self.r)

    def draw(self):

        res = shapes.Circle(xy=self.v, radius=self.r)
        return res

    def center(self):
        return self.v

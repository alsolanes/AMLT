import numpy as np
import math
from matplotlib import pyplot as plt


class FuzzyClassifier(object):
    """
    Standard fuzzy classifier
    """
    def __init__(self, data, clusters, m=2):
        self.X = data
        self.C = clusters
        self.U = np.empty(tuple([len(self.X), len(self.C)]))
        self.m = m

    def fit(self, delta=0.1, **kwargs):
        """
        fit clusters to data
        :param delta: stop critrion value
        :param plot_level: different plotting levels, values 0:nothing,1:show all together,2:show detailed plots
        """
        plot_level = kwargs.get('plot_level', 0)
        verbose_level = kwargs.get('verbose_level', 0)
        verbose_iteration = kwargs.get('verbose_iteration', 10)
        delta_increase_iteration = kwargs.get('increase_iteration', 50)
        delta_increase_factore = kwargs.get('increase_factor', 2)

        self.update_memberships()

        iteration = 0
        while 1:
            iteration += 1

            for i, c in enumerate(self.C):
                c.update(self.X, self.U, self.m, i)

            dif = self.update_memberships()

            if dif < delta:
                print ("###", iteration)
                print ("finish", dif, ' < ', delta)
                print ("distance sum: ", sum([sum([c.distance(x) for x in self.X]) for c in self.C]))
                if plot_level == 1:
                    self.show_plot()
                elif plot_level == 2:
                    self.show_detailed_plot()
                return

            if verbose_level == 1:
                if iteration % verbose_iteration == 0:
                    print ("###", iteration)
                    print (dif, ' > ', delta)
                    print ("distance sum: ", sum([sum([c.distance(x) for x in self.X]) for c in self.C]))
            elif verbose_level == 2:
                if iteration % verbose_iteration == 0:
                    print ("###", iteration)
                    print (dif, ' > ', delta)
                    print ("distance sum: ", sum([sum([c.distance(x) for x in self.X]) for c in self.C]))
                    self.show_detailed_plot()

            if iteration % delta_increase_iteration == 0:
                delta *= delta_increase_factore

    def update_memberships(self):
        max_dif = -1

        for j, x in enumerate(self.X):
            distances = [c.distance(x) for c in self.C]
            for i, c in enumerate(self.C):
                old = self.U[j][i]
                self.U[j][i] = 1.0 / sum([(distances[i]/distances[k]) ** (1.0/(self.m-1)) for k in range(len(self.C))])

                if max_dif < abs(old - self.U[j][i]):
                    max_dif = abs(old - self.U[j][i])

        return max_dif

    def show_detailed_plot(self):
        if len(self.C) == 1:
            row_num, col_num = 1, 1
        elif len(self.C) <= 2:
            row_num, col_num = 1, 2
        elif len(self.C) <= 4:
            row_num, col_num = 2, 2
        elif len(self.C) <= 6:
            row_num, col_num = 2, 3
        elif len(self.C) <= 9:
            row_num, col_num = 3, 3
        elif len(self.C) <= 12:
            row_num, col_num = 3, 4

        fig = plt.figure(1)
        for selected in range(len(self.C)):
            shapes = [c.draw() for c in self.C]
            ax = fig.add_subplot(row_num,col_num,selected+1, aspect='equal')
            rgba_colors = np.zeros((len(self.X), 4))
            rgba_colors[:, 0] = 1.0
            rgba_colors[:, 3] = self.U[:, selected]
            ax.scatter(self.X[:, 0], self.X[:, 1], color=rgba_colors)

            for i, c in enumerate(self.C):
                ax.scatter(c.center()[0], c.center()[1], color='black')
                ax.add_artist(shapes[i])
                shapes[i].set_alpha(.5)
                if i == selected:
                    shapes[i].set_color('red')

            ax.set_xlim(np.min(self.X), np.max(self.X))
            ax.set_ylim(np.min(self.X), np.max(self.X))
        plt.show()

    def show_plot(self):
        shapes = [c.draw() for c in self.C]
        fig = plt.figure(1)
        ax = fig.add_subplot(111, aspect='equal')

        ax.scatter(self.X[:, 0], self.X[:, 1], lw=0)

        for i, c in enumerate(self.C):
            ax.scatter(c.center()[0], c.center()[1], color='black')
            ax.add_artist(shapes[i])
            shapes[i].set_alpha(.5)

        ax.set_xlim(np.min(self.X), np.max(self.X))
        ax.set_ylim(np.min(self.X), np.max(self.X))
        plt.show()

    def scatter_clusters_data(self):
        if self.X.shape[1] > 2:
            print ("Only 2d data can be plotted!")
            return

        clustered_data = [[] for i in range(len(self.C))]
        for j, x in enumerate(self.X):
            ci = np.argmax(self.U[j, :])
            clustered_data[ci].append(x)

        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow','0.75','0.25','black','0.5']
        for i, xs in enumerate(clustered_data):
            xs = np.array(xs)
            plt.scatter(xs[:, 0], xs[:, 1], color=colors[i], lw=0)
        plt.xlim(np.min(self.X), np.max(self.X))
        plt.ylim(np.min(self.X), np.max(self.X))
        plt.show()

import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# functions for clustering

# function for calculating Euclidean Distance
def Dist(X, Y):
    res = 0.0
    for i in range(0, X.size):
        res += (X.iloc[i] - Y.iloc[i]) ** 2
    res = math.sqrt(res)
    return res


# realization of C-means

# X - our dataset (Iris)
# N - number of clussters
# M - fuzziness index
# EPS - accuracy error

def Cmeans(X, N, M, EPS):

    accuracy = 1
    dimension = len(X.columns)
    clusterCentroids = pd.DataFrame(np.zeros((N, dimension), dtype=float))
    membershipMatrice = pd.DataFrame(np.zeros((len(X), N), dtype=float))
    distanceMatrix = pd.DataFrame(np.zeros((len(X), N), dtype=float))
    sumDistMatrix = pd.DataFrame(np.zeros((1, dimension), dtype=float))

    # rename columns
    lst = list(range(0, dimension))
    X.columns = lst

    # initialize random centrs
    for i in range(0, N):
        for u in range(0, dimension):
            temp = random.random()
            clusterCentroids.iloc[i, u] = temp

    max_iter = 20
    iter = 0
    # #main part of c-means
    while accuracy > EPS and iter < max_iter:

        iter += 1

        # calculate distance to each center of cluster for each point
        for q in range(0, len(X)):
            for w in range(0, N):
                tempDistance = Dist(X.loc[q], clusterCentroids.loc[w])
                distanceMatrix.loc[q, w] = tempDistance

        # calculate the fuzzy membership
        for q in range(0, len(X)):
            for w in range(0, N):
                tempSum = 0
                for s in range(0, N):
                    temp = (distanceMatrix.loc[q, w] / distanceMatrix.loc[q, s])
                    tempSum += temp ** (2.0 / M - 1.0)
                membershipMatrice.loc[q, w] = 1 / tempSum

        # calculate new centroids
        obs = 0
        for q in range(0, N):
            sumUt = 0
            sumDistMatrix = sumDistMatrix * 0

            for y in range(0, len(X)):
                sumDistMatrix += X.loc[y] * (membershipMatrice.loc[y, q] ** M)
                sumUt += membershipMatrice.loc[y, q] ** M

            sumDistMatrix = sumDistMatrix / sumUt

            # calculate accuracy
            temp = Dist(sumDistMatrix.loc[0], clusterCentroids.loc[q])
            if temp > obs:
                obs = temp
            clusterCentroids.loc[q] = sumDistMatrix.loc[0]
        accuracy = obs

    return clusterCentroids, membershipMatrice, accuracy

# end clustering function

# reading data file
dataset = pd.read_csv('iris.csv', sep=',', names=None, index_col=False)

# block of preprocessing
iris = dataset

# rename columns
lst = list(range(0, len(iris.columns)))
iris.columns = lst

iris = iris.loc[:, 1:4]

print (iris)

# normalization [0,1]
for u in range(1, 5):
    iris[u] = (iris[u] - iris[u].min()) / (iris[u].max() - iris[u].min())

# end of preprocessing block

# c-means
temo = iris.iloc[:, 2:4]

centroids, matrix, eps = Cmeans(temo, 3, 0.6, 0.01)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(temo.loc[:50, 0], temo.loc[:50, 1], color='g')
ax.scatter(temo.loc[50:100, 0], temo.loc[50:100, 1], color='b')
ax.scatter(temo.loc[100:150, 0], temo.loc[100:150, 1], color='r')

for x in range(0, len(centroids)):
    circ = patches.Circle((centroids.loc[x, 0], centroids.loc[x, 1]), 0.2, transform=ax.transAxes, facecolor='grey',
                          alpha=0.1)
    circ1 = patches.Circle((centroids.loc[x, 0], centroids.loc[x, 1]), 0.14, transform=ax.transAxes, facecolor='grey',
                           alpha=0.3)
    circ2 = patches.Circle((centroids.loc[x, 0], centroids.loc[x, 1]), 0.07, transform=ax.transAxes, facecolor='grey',
                           alpha=0.5)
    ax.add_patch(circ)
    ax.add_patch(circ1)
    ax.add_patch(circ2)

plt.show()

print (centroids)

print (matrix)

import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Machalanobis distance
def Dist(X, Y, A):
    res = ((X - Y).dot(np.linalg.inv(A))).dot((X-Y)) ** 2
    return math.sqrt(res)

# Gustafson Kessel fucntion
def GustafsonKessel(dataset, n, m, eps):
    accuracy = 1
    dimension = len(dataset.columns)
    clusterCentroids = pd.DataFrame(np.zeros((n, dimension), dtype=float))
    distanceMatrix = pd.DataFrame(np.zeros((len(dataset), n), dtype=float))
    membershipMatrice = pd.DataFrame(np.zeros((len(dataset), n), dtype=float))
    sumDistMatrix = pd.DataFrame(np.zeros((1, dimension), dtype=float))

    # rename columns
    lst = list(xrange(0, dimension))
    dataset.columns = lst

    # initialize random centers
    for i in range(0, n):
        for u in range(0, dimension):
            temp = random.random()
            clusterCentroids.iloc[i, u] = temp

    # default A matrix for Machalanobis distance
    A = np.identity(dimension)

    print A

    max_iter = 20
    iter = 0
    # #main part of c-means
    while accuracy > eps and iter < max_iter:
        iter += 1

        print iter

        # calculate distance to each center of cluster for each point
        for q in range(0, len(dataset)):
            for w in range(0, n):
                tempDistance = Dist(dataset.loc[q], clusterCentroids.loc[w], A)
                distanceMatrix.loc[q, w] = tempDistance

        # calculate the fuzzy membership
        for q in range(0, len(dataset)):
            for w in range(0, n):
                tempSum = 0
                for s in range(0, n):
                    temp = (distanceMatrix.loc[q, w] / distanceMatrix.loc[q, s])
                    tempSum += temp ** (2.0 / m - 1.0)
                membershipMatrice.loc[q, w] = 1 / tempSum

        # calculate new centroids
        obs = 0
        for q in range(0, n):
            sumUt = 0
            sumDistMatrix = sumDistMatrix * 0

            for y in range(0, len(dataset)):
                sumDistMatrix += dataset.loc[y] * (membershipMatrice.loc[y, q] ** m)
                sumUt += membershipMatrice.loc[y, q] ** m

            sumDistMatrix = sumDistMatrix / sumUt

            # calculate accuracy
            temp = Dist(sumDistMatrix.loc[0], clusterCentroids.loc[q], A)
            if temp > obs:
                obs = temp
            clusterCentroids.loc[q] = sumDistMatrix.loc[0]
        accuracy = obs

        A = (np.linalg.det(np.corrcoef(membershipMatrice.T)) ** 1.0/dimension) * np.linalg.inv(np.corrcoef(membershipMatrice.T))

        print A

    return clusterCentroids, membershipMatrice, accuracy

# reading data file
iris = pd.read_csv('iris.data', sep=',', header=None, names=None, index_col=False)

# block of preprocessing
iris = iris.loc[:, 0:3]

# normalization [0,1]
for u in range(0, 4):
    iris[u] = (iris[u] - iris[u].min()) / (iris[u].max() - iris[u].min())

# end of preprocessing block

# c-means
temo = iris.iloc[:, 1:4]

centroids, matrix, eps = GustafsonKessel(temo, 3, 0.2, 0.01)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(temo.loc[:50, 1], temo.loc[:50, 2], color='g')
ax.scatter(temo.loc[50:100, 1], temo.loc[50:100, 2], color='b')
ax.scatter(temo.loc[100:150, 1], temo.loc[100:150, 2], color='r')

print centroids

for x in range(0, len(centroids)):
    circ = patches.Circle((centroids.loc[x, 1], centroids.loc[x, 2]), 0.2, transform=ax.transAxes, facecolor='grey',
                          alpha=0.1)
    circ1 = patches.Circle((centroids.loc[x, 1], centroids.loc[x, 2]), 0.14, transform=ax.transAxes, facecolor='grey',
                           alpha=0.3)
    circ2 = patches.Circle((centroids.loc[x, 1], centroids.loc[x, 2]), 0.07, transform=ax.transAxes, facecolor='grey',
                           alpha=0.5)
    ax.add_patch(circ)
    ax.add_patch(circ1)
    ax.add_patch(circ2)

plt.show()

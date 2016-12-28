import pandas as pd
import matplotlib.pyplot as plt
import math

#functions for clustering

def dist(x1, x2, y1, y2):
    res = math.sqrt(pow((x1-x2),2)+pow((y1-y2),2))
    return res

def Kmeans(X, N):

    centroidsX = []
    centroidsY = []

    # initialize random centrs
    temp = 1/float(N+2);
    for i in range(0, N):
        centroidsX.append(temp)
        centroidsY.append(temp)
        temp += 1/float(N+2)
    print centroidsX
    print centroidsY

    for i in range(0,20):

        #new associations
        for y in range(0, 150):
            min = dist(X.loc[y,2],centroidsX[0],X.loc[y,3],centroidsY[0])
            minI = 0
            for q in range(1,3):
                temp = dist(X.loc[y,2],centroidsX[q],X.loc[y,3],centroidsY[q])
                if temp<min:
                    minI=q
            X.loc[y,4] = minI

        #calculate new centroids

        for q in range(0,N):
            tempX = 0
            tempY = 0
            col = 0
            for io, y in X.iterrows():
                if y[4]==q:
                    tempX += y[2]
                    tempY += y[3]
                    col += 1
            if col!=0:
                centroidsX[q]=float(tempX)/float(col)
                centroidsY[q]=float(tempY)/float(col)
            else:
                centroidsX[q] = float(tempX) / 1.0
                centroidsY[q] = float(tempY) / 1.0

        print "X and Y coordianates after " + str(i) + " iteration"
        print centroidsX
        print centroidsY

    return X, centroidsX, centroidsY

#reading data file
dataset = pd.read_csv('iris.data', sep=',', header=None, names=None)
dataset = dataset.loc[:,2:4]

#block of preprocessing
iris = dataset
b = iris[4].unique()
a = []
for temp in range(0, iris[4].unique().size):
    a.append(temp)
iris[4] = iris[4].replace(b, a)
for u in range(2, 4):
    iris[u] = (iris[u]-iris[u].min())/(iris[u].max()-iris[u].min())
#end of preprocessing block

#plotting default set
y = iris.iloc[0:150, 2].values
X = iris.iloc[0:150, [0, 1]].values

#K-means
x = []
y = []
iris1, x, y = Kmeans(iris, 3)

plt.scatter(x[:], y[:], color='black', marker='x', label='centers')
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='o', label='versicolor')
plt.scatter(X[100:150, 0], X[100:150, 1], color='green', marker='o', label='virginica')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()
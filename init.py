import numpy as np

X = np.loadtxt("iris.csv", usecols=(0,1,2), delimiter=',')

Xold = np.copy(X)
min = np.min(X, axis = 0)
max = np.max(X, axis = 0)
mean = np.max(X, axis = 0)
std = np.std(X, axis = 0)

data = {"X" : X, "Xold" : Xold, "min" : min, "max" : max, "mean" : mean, "std" : std}

c = 3
m = 2
e = 0.001
ro = 0
gamma = 1
val = 1
max = 100
alpha = 0.4
vis = True

param = {'c' : c, 'm' : m, 'e' : e, "ro" : ro, "vis" : vis, "gamma" : gamma, "val" : val, "max" : max, "alpha" : alpha}

# -*- coding: utf-8 -*-
"""
@Clusteval

@author: Peter Pigler
"""
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt


def clusteval(New, Result, Param):
    # load Parameters
    v = Result["Cluster"]['v']
    c = Param['c']
    if Param.has_key('m'):
        m = Param['m']
    else:
        m = 2
    X = New['X']
    [N, n] = map(int, X.shape)
    d = np.zeros((N, c))
    f0 = np.zeros_like(d)

    # Determine whether GK, GG or FCM
    if Result["Cluster"].has_key('Pi'):  # GGclust
        A = Result["Cluster"]['P']
        f0 = Result["Data"]['f']
        fm = pow(f0, m)
        Pi = 1.0 / N * fm.sum(axis=0)
        for i in range(c):
            Xv = X - v[i]
            Xf = np.multiply(Xv, fm.T[i][:, np.newaxis])
            aaa = 1.0 / (pow(np.linalg.det(np.linalg.pinv(A[i])), 1.0 / 2.0))
            bbb = 1.0 / Pi[i]
            ccc = 1. / 2. * np.sum(np.multiply(np.dot(Xv, np.linalg.pinv(A[i])), Xv), axis=1)
            ddd = np.multiply(np.dot(Xv, np.linalg.pinv(A[i])), Xv)
            # d[:,i] = 1.0/(pow(np.linalg.det(np.linalg.pinv(P[i])),1.0/2.0))*1.0/Pi[i]*np.exp(1./2.*np.sum(np.multiply(np.dot(Xv,np.linalg.pinv(P[i])),Xv),axis = 1))
            d[:, i] = aaa * bbb * np.exp(ccc)
        for i in range(N):
            tmp = d[i]  # Create a Distance matrix from D[i] 1-D vector
            tmp = np.repeat(tmp, c)
            tmp = tmp.reshape((c, c))
            f0[i] = 1.0 / np.dot(pow(tmp, 2.0 / (m - 1)), pow(1.0 / d[i], 2.0 / (m - 1)))

    elif Result["Cluster"].has_key('P'):  # GK
        P = np.zeros((c, n, n))
        for i in range(c):
            d[:, i] = np.apply_along_axis(np.linalg.norm, 1, X - v[i])
        for i in range(N):
            tmp = d[i]  # Create a Distance matrix from d[i] 1-D vector
            tmp = np.repeat(tmp, c)
            tmp = tmp.reshape((c, c))
            f0[i] = 1.0 / np.dot(pow(tmp, 2.0 / (m - 1)), pow(1.0 / d[i], 2.0 / (m - 1)))
        fm = pow(f0, m)
        for i in range(c):
            Xf = X - v[i]
            Xf = np.multiply(Xf, fm.T[i][:, np.newaxis])
            P[i] = np.cov(Xf.T)
        for i in range(c):
            A = np.dot(pow(np.linalg.det(P[i]), 1.0 / n), np.linalg.inv(P[i]))
            for j in range(N):
                d[j, i] = sqrt(np.dot((np.dot((X[j] - v[i]), A)), X[j] - v[i]))

    else:  # FCM
        d = np.zeros((N, c))
        for i in range(c):
            d[:, i] = np.apply_along_axis(np.linalg.norm, 1, X - v[i])
        for i in range(N):
            tmp = d[i]  # Create a Distance matrix from d[i] 1-D vector
            tmp = np.repeat(tmp, c)
            tmp = tmp.reshape((c, c))
            f0[i] = 1.0 / np.dot(pow(tmp, 2.0 / (m - 1)), pow(1.0 / d[i], 2.0 / (m - 1)))

    eval = {'d': d, 'f': f0}

    if n == 2:
        lower1 = np.min(X[:, 1])
        upper1 = np.max(X[:, 1])
        scale1 = (upper1 - lower1) / 200.0
        lower2 = np.min(X[:, 2])
        upper2 = np.max(X[:, 2])
        scale2 = (upper2 - lower2) / 200.00
        [x, y] = np.meshgrid(range(lower1, upper1, scale1), range(lower2, upper2, scale2))
        pair = np.array([x[:], y[:]])
        [pair1, pair2] = np.size(pair)
        X1 = np.ones(pair1, 1)
        d = np.zeros(pair1, c)

        if Result["Cluster"].has_key('Pi'):  # GGclust
            for i in range(c):
                Xv = pair - v[i]    # - sub list -> array(pair)
                Xf = np.multiply(Xv, fm.T[i][:, np.newaxis])
                aaa = 1.0 / (pow(np.linalg.det(np.linalg.pinv(A[i])), 1.0 / 2.0))
                bbb = 1.0 / Pi[i]
                ccc = 1. / 2. * np.sum(np.multiply(np.dot(Xv, np.linalg.pinv(A[i])), Xv), axis=1)
                ddd = np.multiply(np.dot(Xv, np.linalg.pinv(A[i])), Xv)
                # d[:,i] = 1.0/(pow(np.linalg.det(np.linalg.pinv(P[i])),1.0/2.0))*1.0/Pi[i]*np.exp(1./2.*np.sum(np.multiply(np.dot(Xv,np.linalg.pinv(P[i])),Xv),axis = 1))
                d[:, i] = aaa * bbb * np.exp(ccc)

            if m > 1:
                d = (d + 1e-10) ^ (-1 / (m - 1))
            else:
                d = (d + 1e-10) ^ (-1)
            tmp = d[i]  # Create a Distance matrix from D[i] 1-D vector
            tmp = np.repeat(tmp, c)
            tmp = tmp.reshape((c, c))
            f0[i] = 1.0 / np.dot(pow(tmp, 2.0 / (m - 1)), pow(1.0 / d[i], 2.0 / (m - 1)))

        elif Result["Cluster"].has_key('P'):  # GK
            P = np.zeros((c, n, n))
            for i in range(c):
                d[:, i] = np.apply_along_axis(np.linalg.norm, 1, pair - v[i])
            for i in range(N):
                tmp = d[i]  # Create a Distance matrix from d[i] 1-D vector
                tmp = np.repeat(tmp, c)
                tmp = tmp.reshape((c, c))
                f0[i] = 1.0 / np.dot(pow(tmp, 2.0 / (m - 1)), pow(1.0 / d[i], 2.0 / (m - 1)))
            fm = pow(f0, m)
            for i in range(c):
                Xf = pair - v[i]
                Xf = np.multiply(Xf, fm.T[i][:, np.newaxis])
                P[i] = np.cov(Xf.T)
            for i in range(c):
                A = np.dot(pow(np.linalg.det(P[i]), 1.0 / n), np.linalg.inv(P[i]))
                for j in range(N):
                    d[j, i] = sqrt(np.dot((np.dot((pair[j] - v[i]), A)), pair[j] - v[i]))

        else:   # FCM
            for i in range(c):
                d[:, i] = np.apply_along_axis(np.linalg.norm, 1, pair - v[i])
            for i in range(N):
                tmp = d[i]  # Create a Distance matrix from d[i] 1-D vector
                tmp = np.repeat(tmp, c)
                tmp = tmp.reshape((c, c))
                f0[i] = 1.0 / np.dot(pow(tmp, 2.0 / (m - 1)), pow(1.0 / d[i], 2.0 / (m - 1)))

        f = np.max(f0.T).T
        Z = np.reshape(f,np.size(x,axis = 0), np.size(x,axis = 1))

    plt.figure()
    CS = plt.contour(x, y, Z)
    plt.title("Contour plot")
    plt.show()

    return eval

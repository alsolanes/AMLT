#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 16:30:13 2016

@author: aleix
"""

import numpy as np
from matplotlib import pyplot as plt
from BaseClustering import GKCluster,BaseClustering
from sklearn import metrics


class GKFCM(object):
    def __init__(self, num_clusters , m=2, plot_level=0,seed=35,det=1):
        self.seed=seed
        self.m=m
        self.num_clusters=num_clusters
        self.centers=[]
        self.radius=[]
        self.det = det
        self.A = np.array([[det**.5, 0], [0, det**.5]])
        self.labels=[]
        self.plot_level=plot_level

    def fit(self,data):
        self.data=data
        self.r = np.max(data)/5
        clusters = [GKCluster(np.max(data), 2) for k in range(self.num_clusters)]
        fc = BaseClustering(data, clusters, m=self.m)
        fc.fit(delta=.001, increase_iteration=20, increase_factor=1.2, plot_level=self.plot_level, verbose_level=0, verbose_iteration=100) 
        self.classifier=fc
        self.centers=fc.C
        self.labels = []
        clustered_data = [[] for i in range(len(self.centers))]
        for j, x in enumerate(self.data):
            ci = np.argmax(self.get_memberships()[j, :])
            clustered_data[ci].append(x)
            self.labels.append(ci)
        self.clustered_data = clustered_data
        if len(data[0])==2:
            self.score=metrics.silhouette_score(np.array(self.data), np.array(self.labels), metric='euclidean')
        else:
            self.score=-1
        return fc.U
        
    def scatter_clusters_data(self,axis=1):
        if self.data.shape[1] > 2:
            print ("Only 2d data can be plotted!")
            return
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow','0.75','0.25','black','0.5']
        for i, xs in enumerate(self.clustered_data):
            xs = np.array(xs)
            plt.scatter(xs[:, 0], xs[:, 1], color=colors[i], lw=0)
        plt.xlim(np.min(self.data), np.max(self.data))
        plt.ylim(np.min(self.data), np.max(self.data))
        if axis==0:
            plt.axis('off')
        plt.show()
    
    def get_memberships(self):
        return self.classifier.U   
        
        
def scatter_2d(data):
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
    plt.scatter(data[:, 0], data[:, 1], color=colors[1], lw=0)
    plt.xlim(np.min(data[:,0])-0.1*np.min(data[:,0]), np.max(data[:,0])+0.1*np.min(data[:,0]))
    plt.ylim(np.min(data[:,1])-0.1*np.min(data[:,0]), np.max(data[:,1])+0.1*np.min(data[:,0]))
    plt.title('Original data')
    plt.show()
    
    
def generate_2d(num_clusters, num_samples=1000, seed=35):
    np.random.seed(seed)
    noise=10
    res = np.empty((num_samples, 2))
    centers = num_samples/5 + np.random.uniform(size=(num_clusters, 2)) * num_samples*num_clusters/5
    radiuses = 50 + np.random.uniform(size=num_clusters) * num_samples/5
    y = []
    for i in range(num_samples):
        
        ind = np.random.randint(num_clusters)
        y.append(ind)
        alpha = np.random.uniform(high=2*np.math.pi)
        r = np.random.uniform(high=radiuses[ind])
        res[i] = centers[ind] + \
                 np.array([r * np.math.cos(alpha), r * np.math.sin(alpha)]) + \
                 np.array([np.random.randint(noise), np.random.randint(noise)])
    return res,y
    
def scatter_clusters_data(data,y):
        if data.shape[1] > 2:
            print ("Only 2d data can be plotted!")
            return
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow','0.75','0.25','black','0.5']
        for i, xs in enumerate(data):
            xs = np.array(xs)
            plt.scatter(xs[0], xs[1], color=colors[y[i]], lw=0)
        plt.xlim(np.min(data), np.max(data))
        plt.ylim(np.min(data), np.max(data))
        plt.show()
    
if __name__ == "__main__":
    
    num_clusters = 5
    num_samples = 100 * num_clusters # number of samples to generate
    print('Generating sample data with {0} clusters and {1} samples...'.format(num_clusters,num_samples))
    data,y = generate_2d(num_clusters, num_samples)
    scatter_2d(data)
    scatter_clusters_data(data,y)
    print('Calculating memberships')
    fc = GKFCM(num_clusters=num_clusters,m=2, seed=5)
    memberships=fc.fit(data)
    fc.scatter_clusters_data()
    print('Memberships obtained:',memberships)
    from sklearn.metrics import confusion_matrix
    cross_val=confusion_matrix(fc.labels,y)
    print(cross_val)
    for i in range(num_clusters):
        print('Cluster {0} has {1} incorrect samples classified from {2}'.format(i,num_samples/num_clusters-np.max(cross_val[i]),num_samples/num_clusters))
    print('Score (Silhuete coefficient):{0}'.format(fc.score))
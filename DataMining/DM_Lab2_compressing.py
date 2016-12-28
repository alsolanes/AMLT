import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#compressing method Oya

#reading file
dataset = pd.read_csv('iris.data', sep=',', header=None, names=None)
dataset = dataset.loc[:,0:3]

#normalize
dataset = StandardScaler().fit_transform(dataset)
#dataset = 2*(dataset-dataset.min())/(dataset.max()-dataset.min())-1

#covariance matrix
mean_vec = np.mean(dataset, axis=0)
cov_mat = np.cov(dataset.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
iu = 0
res = []
for i in eig_pairs:
    iu+=i[0]

for i in eig_pairs:
    temp = i[0]/iu
    res.append(temp)

print res

#inbuild function
pca = PCA(n_components=4)
pca.fit_transform(dataset)
print pca.explained_variance_ratio_
import numpy as np
import pandas as pd

dataset = pd.read_csv('Data.dat', sep='\t', header=None)

upd_dataset = dataset.copy()

for i in range(0,59):
    upd_dataset.iloc[i+1,[0,1]] = dataset.iloc[i,[0,1]]

sum1 = 0
sum2 = 0

for i in range(1,60):
    dfA = pd.DataFrame(upd_dataset.iloc[i, [5,1,3,0]])
    dfB = pd.DataFrame(upd_dataset.iloc[i, [5,1,3,0]])
    sum1 += dfA.dot(dfB.T)
    sum2 += upd_dataset.iloc[i, [5, 1, 3, 0]].multiply(upd_dataset.iloc[i,2])

res = np.linalg.inv(sum1).dot(sum2)
print res
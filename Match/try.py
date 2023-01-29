import sklearn
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, SGDRegressor
import pandas as pd
features=['Days to MRI (From the Date of Diagnosis).pt','ER.pt','HER2.pt','Rows.pt','Race and Ethnicity.pt']
for f in features:
    a=torch.load('tcav/'+f).numpy()
    a=np.average(a,axis=0)
    # print(a.shape)

    import random
    X = []
    Y = []
    Z = []
    T = []
    # a=a[45]
    # m=np.max(a)
    # a=np.max(a)-a
    shape=a.shape
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):

                if a[i,j,k]<=0.1:
                    continue
                X.append(i)
                Y.append(j)
                Z.append(k)
                T.append(a[i,j,k])
    #
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    img = ax.scatter(X,Y,Z, c=T, s=100, cmap = plt.cm.hot_r, marker = 'o', alpha = 0.8)
    fig.colorbar(img)
    plt.title(f[:-3])
    plt.savefig('fig/'+f[:-3]+'.png',format='png')



# m=LinearRegression().fit(weight,b)
# print(m.coef_)
# print(m.intercept_)
import os
import pandas as pd
import numpy as np
import torch
from torch import nn, Tensor
import sys
import sklearn
from sklearn.linear_model import LogisticRegression,LinearRegression, Lasso
from joblib import Parallel,delayed
import gc
import torch
import time
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch import nn
write_ability=1 #write lock
class match:
    def __init__(self):
        # self.base='/jhcnas1/zhoutaichang/match/'
        self.base='/home/zhoutaichang/match/'
        self.model='cnn/'
        self.layers=sorted(os.listdir(self.base+self.model))
        print(self.layers)
        self.weights=[]

    def prepare(self,i:int,j: int):
        layer=self.base+self.model+self.layers[i]
        files=sorted(os.listdir(layer))
        # print(j)
        return torch.load(layer+'/'+files[j]).cpu().detach().tolist()

    def match(self,index:int,clinic, feature,files,j,local=True):
        print(clinic)
        print(index,len(self.layers))
        if index>len(self.layers):
            print("Error", len(self.layers),'is smaller than the input:',index)
            return None
        # print('pass')

        size = feature.shape[1:]
        # print("Layer ",index,":",size)
        num=len(feature)
        feature=feature.reshape(num,-1)
        # size=feature[0].shape
        time.sleep(1)
        # print(size)
        # feature=np.array(feature)
        # print(clinic,"Pairing feature map and clinic data")
        a = pd.ExcelFile('Clinical_and_Other_Features.xlsx')
        data = pd.read_excel(a, 'Data')
        f= data.iloc[0, :]
        data=data.iloc[2:,:]
        data.columns=f
        y=[]
        used=0
        # print(data.columns)
        # print(len(data['Patient ID']))

        for i in range(len(files)):
            for j in range(used,len(data)):
                file=files[i][:-3]
                if '_i' in file:
                    file=file[:-2]
                # if pd.isna(data[clinic].iloc[j]):
                #     data[clinic].iloc[j]=-1
                if file==data['Patient ID'].iloc[j]:
                    y.append(float(data[clinic].iloc[j]))
                    used=j
                    break

        y=(y-np.min(y))/(np.max(y)-np.min(y)) #min_max normalization
        # print(clinic,"Number of feature map: ",len(feature),"max:",np.max(feature),"min:",np.min(feature), end='\t')
        # print("Number of clinic feature: ",len(y),"max:",np.max(y),"min:",np.min(y))
        print(clinic,'Regression')
        #
        # class Linear(nn.Module):
        #     def __init__(self):
        #         super().__init__()
        #         self.linear = nn.Linear(len(feature),1)
        #     def forward(self,x):
        #         return self.linear(x)

        # regressor=LogisticRegression(max_iter=1000,penalty='elasticnet',solver='saga',l1_ratio=0.5).fit(feature,y)
        # regressor=Lasso(positive=True).fit(feature,y)
        regressor=LinearRegression().fit(feature,y)
        # regressor

        time.sleep(1)
        weight=np.array(regressor.coef_).reshape(-1)
        # print(weight.shape)
        #
        # print(self.layers,"Weight max:",np.max(weight),"min:",np.min(weight))
        # print(weight.shape)

        threshold = int(len(weight)*0.02)
        ind = np.argpartition(weight, -threshold)[-threshold:]
        threshold = np.min(weight[ind])
        print(clinic,"Threshold: "+str(threshold))
        weight = np.where((weight < threshold) | (weight < 0), 0, weight)
        # print(weight.shape)
        weight = torch.from_numpy(weight.reshape(size))
        save='/jhcnas1/zhoutaichang/explanation/'
        if not os.path.isdir(save):
            os.mkdir(save)
        if not os.path.isdir(save+self.model):
            os.mkdir(save+self.model)
        if not os.path.isdir(save+self.model+'/'+self.layers[index]):
            os.mkdir(save+self.model+'/'+self.layers[index])
        global write_ability
        global thresholds
        while write_ability==0:
            time.sleep(1)
        write_ability=0
        thresholds[i][j] = threshold
        print(clinic,'thresholds')
        torch.save(thresholds, '/jhcnas1/zhoutaichang/explanation/cnn/thresholds.pt')
        torch.save(weight,save+self.model+'/'+self.layers[index]+'/'+clinic+'.pt')
        write_ability=1
        return threshold


    def filter(self,index,clinic, weight,size):
        print(weight.shape)
        weight = np.where((weight < threshold) | (weight < 0), 0, 1)
        print(weight.shape)
        weight = torch.from_numpy(weight.reshape(size))
        save = '/jhcnas1/zhoutaichang/explanation/'

        if not os.path.isdir(save):
            os.mkdir(save)
        if not os.path.isdir(save + self.model):
            os.mkdir(save + self.model)
        if not os.path.isdir(save + self.model + '/' + self.layers[index]):
            os.mkdir(save + self.model + '/' + self.layers[index])
        torch.save(weight, save + self.model + '/' + self.layers[index] + '/' + clinic + '.pt')
        return threshold


#
#
m=match()
a = pd.ExcelFile('Clinical_and_Other_Features.xlsx')
data = pd.read_excel(a, 'Data')
features= data.iloc[0, 1:].tolist()
print('Feature loaded.')
# print(features)
# features=['Rows','Scan Options','Patient Position During MRI','Race and Ethnicity',
#           'HER2','ER','Mol Subtype','Recurrence event(s)','Adjuvant Chemotherapy'
#           ]
thresholds=[[0.0]*len(features)]*17
thresholds = np.array(thresholds).astype(float)
thresholds = torch.from_numpy(thresholds)
# for i in range(6,17):
def match(i, j,feature,files):
    m.match(i, features[j], feature, files,j, False)

def global_match(i):
    layer = m.base + m.model + m.layers[i]
    files = sorted(os.listdir(layer))
    print("Loading feature map: " + m.layers[i])

    feature= Parallel(n_jobs=100, backend='threading')(
        delayed(m.prepare)(i, j) for j in range(0, len(files)))
    feature = np.array(feature).astype(float)
    print("Feature map: " + str(i)+" loaded.")
    # for j in range(len(features)):



    Parallel(n_jobs=20, backend='multiprocessing')(delayed(match)(i, j,feature,files) for j in range(0, len(features)))
    torch.save(thresholds, '/jhcnas1/zhoutaichang/explanation/cnn/thresholds.pt')

# Parallel(n_jobs=3, backend='multiprocessing')(delayed(global_match)(i) for i in range(4, 16))
global_match(4)
global_match(12)

# weight = np.array(m.weights).reshape(-1)
# weight= weight[weight>0]
# threshold = int(len(weight)*0.02)
# ind = np.argpartition(weight, -threshold)[-threshold:]
# threshold = np.min(weight[ind])
torch.save(thresholds, '/jhcnas1/zhoutaichang/explanation/cnn/thresholds.pt')
def draw(thresholds):
    vegetables = m.layers
    farmers = features

    harvest = thresholds.numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(harvest)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(farmers)), labels=farmers)
    ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(vegetables)):
        for j in range(len(farmers)):
            text = ax.text(j, i, harvest[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Thresholds")
    fig.tight_layout()
    # plt.show()
    plt.savefig('/jhcnas1/zhoutaichang/explanation/cnn/thresholds.png',format='png')
draw(thresholds)
# print("Global Threshold: ",threshold)
# for w in range(len(m.weights)):
#     m.filter(5,features[w],m.weights[w],[128,12,12,12])





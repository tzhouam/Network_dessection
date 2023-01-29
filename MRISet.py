import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math
import os
class MRISet(Dataset):
    def __init__(self,dir):
        #data loading
        self.root=dir
        self.patient=os.listdir(dir)
        self.patient_num=len(self.patient)


    def __getitem__(self, item):
        #dataset[index]
        data_x=self.min_max(torch.load(os.path.join(self.root,self.patient[item])))
        if '_i' in self.patient[item].lower():
            data_y=torch.FloatTensor([0,1])
        else:
            data_y=torch.FloatTensor([1,0])
        return data_x,data_y

    def __len__(self):
        return self.patient_num

    def min_max(self,a:np.array):
        if np.min(a)==np.max(a):
            # print("Has 0")
            return np.zeros(a.shape).astype(np.float32)
        scaled_image = (np.maximum(a, 0) / a.max()) * 1.0

        return scaled_image
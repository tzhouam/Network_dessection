from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import os
# import utils
from tqdm import tqdm_notebook
import multiprocessing
import json
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from hyperparameter import Hyperpara
Hyper=Hyperpara()

class ClassificationModel3D(nn.Module):
    """The model we use in the paper."""

    def __init__(self, dropout:float=0, dropout2:float=0):
        nn.Module.__init__(self)
        self.Conv_1a = nn.Conv3d(3, 8, 3,padding=1)
        self.Conv_1b = nn.Conv3d(8, 8, 3,padding=1)
        self.mp1=nn.MaxPool3d(2)
        self.Conv_1_bn = nn.BatchNorm3d(8,affine=False)
        self.cdp1=nn.Dropout3d(p=0.2)

        self.Conv_2a = nn.Conv3d(8, 16, 3, padding=1)
        self.Conv_2b = nn.Conv3d(16, 16, 3, padding=1)
        self.mp2 = nn.MaxPool3d(2)
        self.Conv_2_bn = nn.BatchNorm3d(16)
        self.cdp2=nn.Dropout3d(p=0.2)


        self.Conv_3a = nn.Conv3d(16, 32, 3, padding=1)
        self.Conv_3b = nn.Conv3d(32, 32, 3, padding=1)
        self.mp3 = nn.MaxPool3d(2)
        self.Conv_3_bn = nn.BatchNorm3d(32)
        self.cdp3=nn.Dropout3d(p=0.2)


        self.Conv_4a = nn.Conv3d(32, 64, 3, padding=1)
        self.Conv_4b = nn.Conv3d(64, 64, 3, padding=1)
        self.mp4 = nn.MaxPool3d(2)
        self.Conv_4_bn = nn.BatchNorm3d(64)
        self.cdp4=nn.Dropout3d(p=0.2)


        self.Conv_5a = nn.Conv3d(64, 128, 3, padding=1)
        self.Conv_5b = nn.Conv3d(128, 128, 3, padding=1)
        self.mp5 = nn.MaxPool3d(2)
        self.Conv_5_bn = nn.BatchNorm3d(128)
        self.cdp5=nn.Dropout3d(p=0.2)
        self.gap = torch.nn.AdaptiveAvgPool3d(1)
        self.L1=nn.Linear(129,512)
        self.L2=nn.Linear(512,64)
        self.L3=nn.Linear(64,2)
        self.lbn=nn.BatchNorm1d(128)
        self.dp1=nn.Dropout(dropout)
        self.dp2=nn.Dropout(dropout2)



    def forward(self, input):
        x=input[0]
        thick=input[1]

        batch_size=x.size(0)
        shape = x.size()
        noise = torch.cuda.FloatTensor(shape) if torch.cuda.is_available() else torch.FloatTensor(shape)
        torch.randn(shape, out=noise)
        device = torch.device("cuda:" + str(Hyper.cuda))

        noise=noise.to(device)
        x += noise * Hyper.eps

        x=nn.ReLU()(self.cdp1(self.mp1(self.Conv_1_bn(self.Conv_1b(self.Conv_1a(x))))))
        x = nn.ReLU()(self.cdp2(self.mp2(self.Conv_2_bn(self.Conv_2b(self.Conv_2a(x))))))
        x=nn.ReLU()(self.cdp3(self.mp3(self.Conv_3_bn(self.Conv_3b(self.Conv_3a(x))))))
        x=nn.ReLU()(self.cdp4(self.mp4(self.Conv_4_bn(self.Conv_4b(self.Conv_4a(x))))))
        x = nn.ReLU()(self.cdp5(self.mp5(self.Conv_5_bn(self.Conv_5b(self.Conv_5a(x))))))
        thick=thick.view(batch_size,-1)
        x=self.gap(x)
        x = x.view(batch_size, -1)
        x=self.lbn(x)
        x=torch.cat([x,thick],dim=1)
        # x = self.dp1(x)
        x = nn.ReLU()(self.L1(x))
        x = self.dp1(x)
        x = nn.ReLU()(self.L2(x))
        x=self.dp2(x)
        x = self.L3(x)

        # Note that no sigmoid is applied here, because the network is used in combination with BCEWithLogitsLoss,
        # which applies sigmoid and BCELoss at the same time to make it numerically stable.

        return x

# resnet18 = ClassificationModel3D(dropout=0.2,dropout2=0)
# resnet18.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# summary(resnet18, (3, 100, 100,100))
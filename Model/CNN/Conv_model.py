import torch
from torch import nn, Tensor
import numpy
import pandas
from ConvBlock import ConvBlock
from torchsummary import summary
from hyperparameter import Hyperpara
Hyper=Hyperpara()
class CNN(nn.Module):
    def __init__(self, in_channels=3, block=ConvBlock, outputs=2,width=7):
        super().__init__()
        device = torch.device("cuda:" + str(Hyper.cuda))

        self.layer0 = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            block(64, 64, downsample=True),
            block(64, 64, downsample=False),
        )

        self.layer2 = nn.Sequential(
            block(64, 128, downsample=False),
            block(128, 128, downsample=False),
        )

        self.layer3 = nn.Sequential(
            block(128, 256, downsample=True),
            block(256, 256, downsample=False)
        )


        self.layer4 = nn.Sequential(
            block(256, 512, downsample=False),
            block(512, 512, downsample=False),
        )


        self.gap = torch.nn.AdaptiveAvgPool3d(1).to(device)
        self.fc1 = torch.nn.Linear(512, 1024).to(device)
        self.fc_dp=torch.nn.Dropout(0.5).to(device)
        self.fc2 = torch.nn.Linear(1024, outputs).to(device)
        self.layer0.apply(self.init_weights).to(device)
        self.layer1.apply(self.init_weights).to(device)
        self.layer2.apply(self.init_weights).to(device)
        self.layer3.apply(self.init_weights).to(device)
        self.layer4.apply(self.init_weights).to(device)


    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        input=input.to(torch.device("cuda:"+str(Hyper.cuda) if torch.cuda.is_available() else "cpu"))
        # shape = input.size()
        # noise = torch.cuda.FloatTensor(shape) if torch.cuda.is_available() else torch.FloatTensor(shape)
        # input=input+noise

        batch_size=input.shape[0]
        # shape = input.size()
        # noise = torch.FloatTensor(shape)
        # torch.randn(shape, out=noise)
        # device = torch.device("cuda:" + str(Hyper.cuda))
        #
        # noise = noise.to(device)
        # input += noise * Hyper.eps
        input = self.layer0(input)
        # print(input.shape)
        input = self.layer1(input)
        # print(input.shape)
        input = self.layer2(input)
        # print(input.shape)
        input = self.layer3(input)
        # print(input.shape)
        input = self.layer4(input)
        # print(input.shape)
        input = self.gap(input)
        # print(input.shape)
        input = input.view(batch_size,-1)
        # print(input.shape)
        input = self.fc1(input)
        input = self.fc_dp(input)
        input = nn.ReLU()(input)
        input = self.fc2(input)

        return input


resnet18 = CNN(3, ConvBlock, outputs=2)
resnet18.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
summary(resnet18, (3, 100, 100,100))
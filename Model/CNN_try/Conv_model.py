import torch
from torch import nn, Tensor
import numpy
import pandas
from ConvBlock import ConvBlock
from torchsummary import summary
from hyperparameter import Hyperpara
Hyper=Hyperpara()
class Conv(nn.Module):
    def __init__(self, in_channels, convblock=ConvBlock, outputs=1000):
        super().__init__()
        device = torch.device("cuda:" + str(Hyper.cuda))

        self.layer0 = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            convblock(64, 64, downsample=True),
            convblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            convblock(64, 128, downsample=True),
            # convblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            convblock(128, 256, downsample=True),
            # convblock(256, 256, downsample=False)
        )


        self.layer4 = nn.Sequential(
            convblock(256, 512, downsample=True),
            # convblock(512, 512, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool3d(1)
        self.fc1 = torch.nn.Linear(512*27, 1024)
        self.fc_dp=torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(1024, outputs)
        self.layer0.apply(self.init_weights)
        self.layer1.apply(self.init_weights)
        self.layer2.apply(self.init_weights)
        self.layer3.apply(self.init_weights)
        self.layer4.apply(self.init_weights)


    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        input=input.to(torch.device("cuda:"+str(Hyper.cuda) if torch.cuda.is_available() else "cpu"))
        batch_size=input.shape[0]
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
        # input = self.gap(input)
        # print(input.shape)
        input = input.view(batch_size,-1)
        # print(input.shape)
        input = self.fc1(input)
        input = self.fc_dp(input)
        input = nn.ReLU()(input)
        input = self.fc2(input)

        return input

#
# resnet18 = ResNet18(3, ResBlock, outputs=2)
# resnet18.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# summary(resnet18, (3, 100, 100,100))
from torch import nn
import torch
from hyperparameter import Hyperpara
Hyper=Hyperpara()
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, width=3):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=width, stride=1, padding=(width-1)//2)
            self.mp1= nn.MaxPool3d(kernel_size=2,stride=2)
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=width, stride=1,padding=(width-1)//2),

                nn.MaxPool3d(kernel_size=2, stride=2),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=width, stride=1, padding=(width-1)//2)
            self.shortcut = nn.Sequential(nn.Conv3d(in_channels,out_channels,kernel_size=1,stride=1))
            self.mp1 = nn.Sequential()
        self.ap1 = nn.AvgPool3d(kernel_size=2, stride=1,padding=1)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=width, stride=1, padding=(width-1)//2)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dp1= nn.Dropout3d()
        self.dp2= nn.Dropout3d()

    def forward(self, input):
        # print('In')
        device = torch.device("cuda:" + str(Hyper.cuda))
        shortcut = self.shortcut(input).to(device)

        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.mp1(self.conv2(input)))).to(device)
        # print(input.shape,shortcut.shape)
        input = input + shortcut
        return self.dp1(nn.ReLU()(input))
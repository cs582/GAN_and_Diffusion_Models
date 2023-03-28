import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiscaleConvolution(nn.Module):
    def __init__(self, img_size, in_channels, kernel):
        super(MultiscaleConvolution, self).__init__()
        self.img_size = img_size
        self.in_channels = in_channels

        self.bnS = nn.BatchNorm2d(in_channels)
        self.convS = nn.Conv2d(in_channels=in_channels, kernel_size=kernel, padding=0, stride=1, out_channels=in_channels)

        self.bnM = nn.BatchNorm2d(in_channels)
        self.convM = nn.Conv2d(in_channels=in_channels, kernel_size=kernel, padding=0, stride=1, out_channels=in_channels)

        self.bnL = nn.BatchNorm2d(in_channels)
        self.convL = nn.Conv2d(in_channels=in_channels, kernel_size=kernel, padding=0, stride=1, out_channels=in_channels)

    def forward(self, x):
        # Downsample
        x_s = F.interpolate(x, scale_factor=1/4)
        x_m = F.interpolate(x, scale_factor=1/2)

        # Convolutions
        x_s = self.convS(self.bnS(x_s))
        x_m = self.convM(self.bnM(x_m))
        x_l = self.convL(self.bnL(x))

        # Upsample
        x_s = F.interpolate(x_s, scale_factor=4)
        x_m = F.interpolate(x_m, scale_factor=2)
        return (x_s + x_m + x_l)


class DiffusionDense(nn.Module):
    def __init__(self, img_size, in_channels):
        super(DiffusionDense, self).__init__()

        self.in_size = np.prod(img_size) * in_channels
        self.img_size = img_size
        self.in_channels = in_channels

        self.fc1 = nn.Linear(in_features=self.in_size, out_features=self.in_size)
        self.fc2 = nn.Linear(in_features=self.in_size, out_features=self.in_size)

        self.tanh = nn.Tanh()

    def forward(self, x):
        print("in shape", x.shape)
        x = x.view(-1, self.in_size, 1)
        print("resized shape", x.shape)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        return x.view(-1, self.in_channels, *self.img_size)


class DiffusionBlock(nn.Module):
    def __init__(self, img_size, in_channels, kernel):
        super(DiffusionBlock, self).__init__()
        self.img_size = img_size
        self.in_channels = in_channels

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=kernel, padding=0, stride=1, out_channels=in_channels)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.tanh(self.conv(self.bn(x)))
        return out


class SimpleCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=0, stride=2):
        super(SimpleCNNBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=kernel_size, padding=padding, stride=stride, out_channels=out_channels)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        return x


class BinaryMLP(nn.Module):
    def __init__(self, in_size, out_size):
        super(BinaryMLP, self).__init__()
        self.linear1 = nn.Linear(in_size, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, out_size)

        self.relu = nn.ReLU(inplace=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        return x


class MLP(nn.Module):
    def __init__(self, in_size, out_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_size, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, out_size)

        self.relu = nn.ReLU(inplace=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.softmax(self.linear4(x))
        return x
import torch.nn as nn


class SimpleCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=0, stride=2):
        super(SimpleCNNBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=kernel_size, padding=padding, stride=stride, out_channels=out_channels)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
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
import torch
from torch import nn
from src.models.building_blocks import SimpleCNNBlock, MLP


class MicroCNN(nn.Module):
    def __init__(self, in_channels, out_size):
        """
        This MicroCNN is meant to be used for testing on MNIST only.
        :param in_channels:
        :param out_size:
        """
        super(MicroCNN, self).__init__()
        self.block1 = SimpleCNNBlock(in_channels=in_channels, kernel_size=3, padding=0, stride=1, out_channels=2)
        self.block2 = SimpleCNNBlock(in_channels=2, kernel_size=3, padding=0, stride=1, out_channels=4)
        self.mlp = MLP(in_size=24*24*4, out_size=out_size)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        x = torch.flatten(x, 1)

        x = self.mlp(x)
        return x


class MiniCNN(nn.Module):
    def __init__(self, in_channels, out_size):
        super(MiniCNN, self).__init__()
        self.block1 = SimpleCNNBlock(in_channels=in_channels, kernel_size=5, padding=0, stride=2, out_channels=8)
        self.block2 = SimpleCNNBlock(in_channels=8, kernel_size=5, padding=0, stride=2, out_channels=16)
        self.block3 = SimpleCNNBlock(in_channels=16, kernel_size=5, padding=0, stride=2, out_channels=32)

        self.mlp = MLP(in_size=29*29*32, out_size=out_size)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = torch.flatten(x, 1)

        x = self.mlp(x)
        return x

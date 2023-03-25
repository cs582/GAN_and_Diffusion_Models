import torch
from torch import nn


class MNISTGenerator(nn.Module):
    def __init__(self, in_size, out_shape):
        """
        This model is only meant to be used for testing on the MNIST dataset

        :param in_size:
        :param out_size:
        """
        super(MNISTGenerator, self).__init__()
        self.h, self.w = out_shape

        self.linear1 = nn.Linear(in_size, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 256)
        self.linear4 = nn.Linear(256, out_shape[0]*out_shape[1])

        self.relu = nn.ReLU(inplace=True)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.tanh(self.linear4(x))
        return x.view(-1, 1, self.h, self.w)
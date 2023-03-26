import torch.nn as nn
import numpy as np


class MNISTDiscriminator(nn.Module):
    def __init__(self, img_size):
        super(MNISTDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_size)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


class MNISTGenerator(nn.Module):
    def __init__(self, in_size, out_shape):
        """
        This model is only meant to be used for testing on the MNIST dataset

        :param in_size:
        :param out_size:
        """
        super(MNISTGenerator, self).__init__()
        self.h, self.w = out_shape

        self.linear1 = nn.Linear(in_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 512)
        self.linear4 = nn.Linear(512, out_shape[0]*out_shape[1])

        self.relu = nn.ReLU(inplace=True)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.tanh(self.linear4(x))
        return x.view(-1, 1, self.h, self.w)
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
        super(MNISTGenerator, self).__init__()

        self.out_shape = out_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_size, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(out_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.out_shape)
        return img

# class MNISTGenerator(nn.Module):
#     def __init__(self, in_size, out_shape):
#         super(MNISTGenerator, self).__init__()
#         self.h, self.w = out_shape
#
#         self.linear1 = nn.Linear(in_size, 128)
#         self.linear2 = nn.Linear(128, 256)
#         self.linear3 = nn.Linear(256, 512)
#         self.linear4 = nn.Linear(512, self.h*self.w)
#
#         self.activ = nn.LeakyReLU(0.2, inplace=True)
#
#         self.tanh = nn.Tanh()
#
#     def forward(self, x):
#         x = self.activ(self.linear1(x))
#         x = self.activ(self.linear2(x))
#         x = self.activ(self.linear3(x))
#         x = self.tanh(self.activ(self.linear4(x)))
#         return x.view(-1, 1, self.h, self.w)
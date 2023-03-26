import numpy as np
import torch
import torch.nn as nn
from src.models.blocks import DiffusionDense, DiffusionBlock


class MNISTDiffusion(nn.Module):
    def __init__(self, img_size, timesteps):
        super(MNISTDiffusion, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=timesteps, embedding_dim=np.prod(img_size))

        self.dense = DiffusionDense(img_size=img_size, in_channels=1)

        self.block1 = DiffusionBlock(img_size=img_size, in_channels=1, kernel=1)
        self.block2 = DiffusionBlock(img_size=img_size, in_channels=1, kernel=1)

        self.block_mean = DiffusionBlock(img_size=img_size, in_channels=1, kernel=1)
        self.block_cov = DiffusionBlock(img_size=img_size, in_channels=1, kernel=1)

    def forward(self, x, t):
        time_token = self.embedding(t).expand(x.shape[0], -1)
        time_token = time_token.view(x.shape[0], *x[0].shape)

        x = self.dense(x * time_token)

        x = self.block1(x)
        x = self.block2(x)

        mean = self.block_mean(x)
        logcov = self.block_cov(x)

        cov = torch.exp(0.5 * logcov)
        return mean + x * cov



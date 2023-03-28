import numpy as np
import torch
import torch.nn as nn
from src.models.blocks import DiffusionDense, DiffusionBlock


class MNISTDiffusion(nn.Module):
    def __init__(self, img_size, timesteps):
        super(MNISTDiffusion, self).__init__()

        self.image_size = img_size
        self.timesteps = timesteps
        self.flattened_image_size = img_size[0] * img_size[1]
        self.embedded_image_size = (img_size[0] + 1) * img_size[1]

        self.embedding = nn.Embedding(num_embeddings=self.timesteps, embedding_dim=self.image_size[1])

        self.dense = DiffusionDense(
            flattened_img_size=self.embedded_image_size,
            in_channels=1,
            flattened_out_size=self.flattened_image_size
        )

        self.block1 = DiffusionBlock(img_size=img_size, in_channels=1, kernel=1)
        self.block2 = DiffusionBlock(img_size=img_size, in_channels=1, kernel=1)

        self.block_mean = DiffusionBlock(img_size=img_size, in_channels=1, kernel=1)
        self.block_cov = DiffusionBlock(img_size=img_size, in_channels=1, kernel=1)

    def forward(self, x, t):
        # Embed the timestep
        t_embedded = self.embedding(t).expand(len(x), 1, -1, -1)

        # Concatenate the image and the timestep
        x = torch.cat([t_embedded, x], dim=2)

        # Flatten the input image
        x = x.view(-1, self.embedded_image_size)

        x = self.dense(x)

        # Reshape image
        x = x.view(-1, 1, *self.image_size)

        x = self.block1(x)
        x = self.block2(x)

        mean = self.block_mean(x)
        logcov = self.block_cov(x)

        cov = torch.exp(0.5 * logcov)
        return mean + x * cov




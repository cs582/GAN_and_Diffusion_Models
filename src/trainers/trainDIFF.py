import torch
from tqdm import tqdm
import numpy as np
from src.data_utils.transformations import noise_images


def train(model, device, training_dataset, optimizer, loss_function, times, beta_zero, beta_end):

    timesteps = torch.from_numpy(np.arange(0, times)).view(-1, 1).to(device, dtype=torch.int64)
    beta_range = torch.from_numpy(np.linspace(beta_zero, beta_end, times)).to(device)

    for i in range(0, times):
        print(f"Training Timestep {times-i}...")

        t = times - timesteps[i] - 1
        beta = beta_range[i]

        loss_history = []

        for j, (x, _) in tqdm(enumerate(training_dataset), total=len(training_dataset)):
            if j == 0:
                prev_imgs = x.to(device)
                continue

            curr_imgs = noise_images(prev_imgs, beta)

            optimizer.zero_grad()

            imgs_reconstructed = model(curr_imgs, t)

            loss = loss_function(imgs_reconstructed, prev_imgs)

            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            prev_imgs = curr_imgs

            print(f"finished taking step {j}")

        print(f"AVG LOSS: {np.round(np.mean(loss_history), 3)}")
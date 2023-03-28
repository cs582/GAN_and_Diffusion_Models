import torch
from tqdm import tqdm
import numpy as np
from src.data_utils.transformations import noise_images


def train(model, device, training_dataset, optimizer, loss_function, times, beta_zero, beta_end):

    timesteps = list(range(0, times))
    beta_domain = np.linspace(beta_zero, beta_end, times).tolist()

    assert len(timesteps) == len(beta_domain), f"Ts and beta_domain expected to have same length but obtained len(Ts)={len(Ts)} and len(beta_domain)={len(beta_domain)}."

    for i, (t, beta) in enumerate(zip(timesteps, beta_domain)):
        print(f"Training Timestep {times-i}...")

        t = torch.tensor([t]).to(device)

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

        print(f"AVG LOSS: {np.round(np.mean(loss_history), 3)}")
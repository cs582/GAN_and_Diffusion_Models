import torch
from tqdm import tqdm
import numpy as np
from src.utils.transformations import noice_image


def train(model, device, training_dataset, optimizer, loss_function, times, beta_range):
    beta_zero, beta_end = beta_range

    Ts = list(range(0, times))
    beta_domain = np.linspace(beta_zero, beta_end, times).tolist()

    assert len(Ts) == len(beta_domain), f"Ts and beta_domain expected to have same length but obtained len(Ts)={len(Ts)} and len(beta_domain)={len(beta_domain)}."

    for i, (t, beta) in enumerate(zip(Ts, beta_domain)):
        print(f"Training Timestep {times-i}...")

        loss_history = []

        for j, (x, _) in tqdm(enumerate(training_dataset), total=len(training_dataset)):
            if j == 0:
                prev_imgs = x.to(device)
                continue

            curr_imgs = noice_image(prev_imgs, mu=(1-beta)**0.5, sigma=beta)

            optimizer.zero_grad()

            imgs_reconstructed = model(curr_imgs, t)

            loss = loss_function(imgs_reconstructed, prev_imgs)

            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            prev_imgs = curr_imgs

        print(f"AVG LOSS: {np.round(np.mean(loss_history), 3)}")
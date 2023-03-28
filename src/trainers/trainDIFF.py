import torch
from tqdm import tqdm
import numpy as np
from src.data_utils.transformations import noise_images
from src.visualization.tools import preview_images


def train(model, device, training_dataset, optimizer, loss_function, times, beta_zero, beta_end):

    timesteps = torch.from_numpy(np.arange(0, times)).view(-1, 1).to(device, dtype=torch.int64)
    beta_range = torch.from_numpy(np.linspace(beta_zero, beta_end, times)).to(device)

    for batch_n, (x, _) in enumerate(training_dataset):
        print(f"Training Batch {batch_n}...")
        loss_history = []

        for i in tqdm(range(0, times), total=times):
            t = times - timesteps[i]
            beta = beta_range[i]

            if i == 0:
                prev_imgs = x.to(device)
                if batch_n % 100 == 0:
                    preview_images(prev_imgs, 5, 5, "preview/MNIST_DIFF", f"batch_{batch_n}_T_{t.item()}")
                continue

            curr_imgs = noise_images(prev_imgs, beta)

            optimizer.zero_grad()

            imgs_reconstructed = model(curr_imgs, t)

            loss = loss_function(imgs_reconstructed, prev_imgs)

            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if t % 200 == 0 and batch_n % 100 == 0:
                preview_images(imgs_reconstructed, 5, 5, "preview/MNIST_DIFF", f"batch_{batch_n}_T_{t.item()}")
                prev_imgs = curr_imgs

        print(f"AVG LOSS: {np.round(np.mean(loss_history), 3)}")
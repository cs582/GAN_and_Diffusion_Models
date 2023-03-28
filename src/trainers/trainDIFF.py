import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from src.data_utils.transformations import noise_images
from src.visualization.tools import preview_images


def train(model, device, training_dataset, optimizer, loss_function, times, beta_zero, beta_end):

    train_history = {"loss": []}

    plot_every_batch = 800
    plot_every_time = 50

    timesteps = torch.from_numpy(np.arange(0, times)).view(-1, 1).to(device, dtype=torch.int64)
    beta_range = torch.from_numpy(np.linspace(beta_zero, beta_end, times)).to(device)

    prev_images_memory = [x for x, y in training_dataset]

    n_batches = len(training_dataset)

    for i in range(0, times-1):
        t = timesteps[i] + 1

        print(f"Training T {t.item()}...")
        beta = beta_range[i]

        loss_history = []

        for batch_n in tqdm(range(0, n_batches), total=n_batches):

            # Retrieve previous images (t-1) from memory
            prev_imgs = prev_images_memory[batch_n].to(device)

            # Add noice to previous images (t-0) to get current images (t)
            curr_imgs = noise_images(prev_imgs, beta)

            optimizer.zero_grad()

            # Reconstruct current image to recover previous image
            imgs_reconstructed = model(curr_imgs, t)

            # Transform images to represent their distributions
            q = F.log_softmax(imgs_reconstructed, dim=0)
            p = F.softmax(prev_imgs, dim=0)

            # Computing KLDivergence
            loss = loss_function(q, p)

            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            # Print original image
            if i == 0 and batch_n % plot_every_batch == 0:
                preview_images(prev_imgs, 5, 5, "preview/MNIST_DIFF", f"batch{batch_n}_T{t.item()-1}")

            # Print reconstruction every interval of batches and interval of time
            if t % plot_every_time == 0 and batch_n % plot_every_batch == 0:
                preview_images(imgs_reconstructed, 5, 5, "preview/MNIST_DIFF", f"batch{batch_n}_T{t.item()}")

            prev_images_memory[batch_n] = curr_imgs

        avg_loss = np.mean(loss_history)
        print(f"AVG LOSS: {np.round(avg_loss,3)}")

        train_history['loss'].append(avg_loss)

    return train_history
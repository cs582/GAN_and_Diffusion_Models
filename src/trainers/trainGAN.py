import torch
import numpy as np
from src.utils.viz_tools import preview_images


def train(G, D, device, latent_vector_size, training_dataset, epochs, D_optimizer, G_optimizer, loss_function):
    progress = []

    for epoch in range(epochs):
        G = G.train()

        LOSS_D = 0
        LOSS_G = 0

        for i, (x, y) in enumerate(training_dataset):
            # Calculate number of elements in batch size
            batch_size = len(x)

            ########################################
            ### GENERATOR OPTIMIZATION
            ########################################

            # Generate latent vector
            z_noice = torch.rand(batch_size, latent_vector_size).to(device)

            # Zero the parameter gradients
            G_optimizer.zero_grad()

            # Generate data
            y_hat = D(G(z_noice)).unsqueeze(1).to(device)
            y_true = torch.zeros_like(y_hat, dtype=torch.float32).to(device)

            # Calculate loss
            loss_g = loss_function(y_hat, y_true)
            LOSS_G += loss_g.item()

            # Backward pass and optimize
            loss_g.backward()
            G_optimizer.step()


            ########################################
            ### DISCRIMINATOR OPTIMIZATION
            ########################################

            # Generate latent vector
            z_noice = torch.rand(batch_size, latent_vector_size).to(device)

            # Zero the parameter gradients
            D_optimizer.zero_grad()

            # Generate data
            gen_data = G(z_noice)

            # Classify data
            gen_data_labels = D(gen_data).to(device)
            true_data_labels = D(x).to(device)

            # Concatenate labels
            y_hat = torch.cat((true_data_labels, gen_data_labels), dim=0).unsqueeze(1).to(device)
            y_true = torch.zeros_like(y_hat, dtype=torch.float32).to(device)
            y_true[:len(x)] = 1.0

            # Calculate loss
            loss_d = loss_function(y_hat, y_true)
            LOSS_D += loss_d.item()

            # Backward pass and optimize
            loss_d.backward()
            D_optimizer.step()

        #---------------------------------------------
        # Print current progress
        #---------------------------------------------

        if (epoch+1) % (epochs//10) == 0:
            G = G.eval()

            percnt = np.round(100*(epoch+1)/epochs, 3)
            print(f"{percnt}% completed!")

            # Generate latent vector
            z_noice = torch.rand(4, latent_vector_size).to(device)

            generated_images = G(z_noice)

            # Preview Images
            preview_images(generated_images, name=f"MNIST_GAN_{percnt}", folder="preview/MNIST_GAN")

            progress.append(generated_images)

        LOSS_D = np.round(LOSS_D/len(training_dataset), 3)
        LOSS_G = np.round(LOSS_G/len(training_dataset), 3)

        print(f"EPOCH {epoch}: AVG. Loss Dis = {LOSS_D}. AVG. Loss Gen = {LOSS_G}")


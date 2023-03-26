import torch
from tqdm import tqdm
import numpy as np
from src.utils.viz_tools import preview_gen_vs_real_images

real_images_preview = None
generated_images_preview = None


def train(G, D, device, latent_vector_size, training_dataset, epochs, D_optimizer, G_optimizer, loss_function):

    progress = []

    for epoch in range(epochs):
        global real_images_preview
        global generated_images_preview

        LOSS_D = 0
        LOSS_G = 0

        for i, (x, _) in tqdm(enumerate(training_dataset), total=len(training_dataset)):
            x = x.to(device)

            if i == 0:
                real_images_preview = x

            # Calculate number of elements in batch size
            batch_size = len(x)

            # Ground truths
            real_labels = torch.ones(batch_size, 1, dtype=torch.float32).to(device)
            gen_labels = torch.zeros(batch_size, 1, dtype=torch.float32).to(device)

            ########################################
            ### GENERATOR OPTIMIZATION
            ########################################

            # Zero the parameter gradients
            G_optimizer.zero_grad()

            # Generate latent vector
            z_noice = torch.tensor(np.random.normal(0, 1, (batch_size, latent_vector_size)), dtype=torch.float32).to(device)

            # Generate images
            gen_images = G(z_noice)

            if i == 0:
                generated_images_preview = gen_images

            # Calculate loss
            loss_g = loss_function(D(gen_images), real_labels)
            LOSS_G += loss_g.item()

            # Backward pass and optimize
            loss_g.backward()
            G_optimizer.step()

            ########################################
            ### DISCRIMINATOR OPTIMIZATION
            ########################################

            # Zero the parameter gradients
            D_optimizer.zero_grad()

            # Classify data
            real_imgs_loss = loss_function(D(x).to(device), real_labels)
            gen_imgs_loss = loss_function(D(gen_images.detach()).to(device), gen_labels)
            loss_d = (real_imgs_loss + gen_imgs_loss)/2

            # Calculate loss
            LOSS_D += loss_d.item()

            # Backward pass and optimize
            loss_d.backward()
            D_optimizer.step()

        #---------------------------------------------
        # Print current progress
        #---------------------------------------------

        if (epoch+1) % (epochs//10) == 0:
            percnt = np.round(100*(epoch+1)/epochs, 3)
            print(f"{percnt}% completed!")

            n_images = 4

            generated_images = generated_images_preview[:n_images]
            real_images = real_images_preview[:n_images]

            # Preview Images
            preview_gen_vs_real_images(generated_images, real_images, vmin=0.0, vmax=1.0, name=f"MNIST_GAN_epoch{epoch}", folder="preview/MNIST_GAN")

            progress.append(generated_images)

        LOSS_D = np.round(LOSS_D/len(training_dataset), 3)
        LOSS_G = np.round(LOSS_G/len(training_dataset), 3)

        print(f"EPOCH {epoch}: AVG. Loss Dis = {LOSS_D}. AVG. Loss Gen = {LOSS_G}")


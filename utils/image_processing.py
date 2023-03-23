import matplotlib.pyplot as plt
import numpy as np
import torch


def normalize_image(img):
    return 2*(np.float64(img)/255) - 1.0


def denormalize_image(img):
    return np.int64(255*((img + 1.0)/2.0))


def preview_image(img, vmin=0, vmax=255, show_channels=True, normalized_image=False):
    if normalized_image:
        img = denormalize_image(img)

    if show_channels:
        fig, ax = plt.subplots(1, 3, figsize=(30, 10))

        ax[0].set_title("Blue Channel")
        ax[0].imshow(img[:,:,0], vmin=vmin, vmax=vmax, cmap="Blues") # Blue channel
        ax[0].axis('off')

        ax[1].set_title("Green Channel")
        ax[1].imshow(img[:,:,1], vmin=vmin, vmax=vmax, cmap="Greens") # Green channel
        ax[1].axis('off')

        ax[2].set_title("Red Channel")
        ax[2].imshow(img[:,:,2], vmin=vmin, vmax=vmax, cmap="Reds") # Red channel
        ax[2].axis('off')

        plt.show()

    img_fixed = img.copy()
    img_fixed[:, :, 0],  img_fixed[:, :, 2] = img[:, :, 2],  img[:, :, 0]


    plt.figure(figsize=(6,6))
    plt.imshow(img_fixed, vmin=vmin, vmax=vmax)
    plt.axis('off')
    return plt.show()


def noice_image(img, mu, sigma, noice_type="mult"):
    length = len(img.flatten())
    size = img.shape
    noice = np.random.normal(mu, sigma, length).reshape(size)

    if noice_type == "add":
        noiced_image = img + noice
    elif noice_type == "mult":
        noiced_image = img * noice

    noiced_image[noiced_image > 1.0] = 1.0
    noiced_image[noiced_image < -1.0] = -1.0

    return noiced_image, noice

def markov_chain_noice(img0, starting_beta, final_beta, T=1000):
    images = []
    prev_img = img0

    betas = np.linspace(starting_beta, final_beta, T).tolist()

    for i, beta in zip(range(0, T), betas):
        mu = (1-beta)**0.5
        sigma = beta

        curr_img, _ = noice_image(prev_img, mu=mu, sigma=sigma)

        if i % (T//10) == 0:
            print(f"Noicing data {np.round(100*i/T, 3)}% with mean={mu} and var={sigma}...")
            images.append(curr_img)
        prev_img = curr_img

    return images


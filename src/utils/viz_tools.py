from src.utils.transformations import denormalize_image
import matplotlib.pyplot as plt
import os


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


def preview_noiced_images(imgs, num=1, vmin=0, vmax=255):
    height = 10
    width = height*len(imgs)

    fig, ax = plt.subplots(1, len(imgs), figsize=(width, height))

    for i, img in enumerate(imgs):
        img = denormalize_image(img)
        fxdimg = img.copy()
        fxdimg[:,:,0], fxdimg[:,:,2] = img[:,:,2], img[:,:,0]
        ax[i].imshow(fxdimg, vmin=vmin, vmax=vmax) # Blue channel
        ax[i].axis('off')

    plt.subplots_adjust(wspace=0)

    plt.savefig(f"images/noiced_preview/preview_{num}.png")

    plt.show()
    return


def preview_helper(img, row, ax, vmin, vmax):
    for i, img in enumerate(img):
        if i == 0:
            print(img.shape)
        # Detach tensor from graph and move it to cpu
        img = img if img.device.type == "cpu" else img.cpu()
        img = img.detach().numpy()

        fxdimg = img.copy()
        if img.shape[0] == 3:
            fxdimg[:,:,0], fxdimg[:,:,2] = img[:,:,2], img[:,:,0]
            ax[row, i].imshow(fxdimg, vmin=vmin, vmax=vmax)
        elif img.shape[0] == 1:
            ax[row, i].imshow(fxdimg[0], vmin=vmin, vmax=vmax, cmap="Blues")
        ax[row, i].axis('off')


def preview_gen_vs_real_images(gen, real, name=None, vmin=0, vmax=255, folder=None):
    height = 10
    width = height*len(gen)

    fig, ax = plt.subplots(2, len(gen), figsize=(width, height))

    preview_helper(gen, 0, ax, vmin=vmin, vmax=vmax)
    preview_helper(real, 1, ax, vmin=vmin, vmax=vmax)

    plt.subplots_adjust(wspace=0)

    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.savefig(f"{folder}/{name}.png")

    return
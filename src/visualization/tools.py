import os
import cv2
import matplotlib.pyplot as plt
from src.data_utils.transformations import denormalize_image


def preview_images(images, rows, columns, dir_path, file_name, is_color_image=False):
    vmin, vmax = (0.0, 1.0) if not is_color_image else (0, 255)

    h, w = (5, 5)

    fig, ax = plt.subplots(nrows=rows, ncols=columns, figsize=(h*columns, w*columns))

    for i in range(0, rows):
        for j in range(0, columns):

            # Detach image from graph
            curr_img = images[i*j].detach()

            # If tensor in gpu move to cpu
            if curr_img.device.type != "cpu":
                curr_img = curr_img.cpu()

            # Change tensor to numpy array
            curr_img = curr_img.numpy()

            # Denormalize image for preview if it's a color image
            if is_color_image:
                curr_img = denormalize_image(curr_img)
            else:
                curr_img = curr_img[0]

            ax[i, j].imshow(curr_img, vmin=vmin, vmax=vmax, cmap="gray")
            ax[i, i].axis('off')

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.subplots_adjust(wspace=0)

    plt.savefig(f"{dir_path}/{file_name}.png")
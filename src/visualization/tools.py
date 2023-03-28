import os
import cv2
import matplotlib.pyplot as plt
from src.data_utils.transformations import denormalize_image


def preview_images(images, rows, columns, dir_path, file_name):
    h, w = (5, 5)

    ax, fig = plt.subplots(nrows=rows, ncols=columns, figsize=(h*columns, w*columns))

    for i in range(0, rows):
        for j in range(0, columns):
            img = denormalize_image(images[i*j].numpy())

            ax[i, j].imshow(img, vmin=0, vmax=255)
            ax[i, i].axis('off')

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.savefig(f"{dir_path}/{file_name}.png")
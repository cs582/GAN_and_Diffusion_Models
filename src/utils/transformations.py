import numpy as np
import cv2


def normalize_image(img):
    return 2*(np.float64(img)/255) - 1.0


def denormalize_image(img):
    return np.int64(255*((img + 1.0)/2.0))


def crop_resize_img(img):
    img_resized = cv2.resize(img, None, fx=0.10, fy=0.10)

    h, w = (256, 256)
    img_h, img_w, _ = img_resized.shape
    y0 = (img_h - h)//2
    x0 = (img_w - w)//2

    img_cropped = img_resized[y0:(y0+h), x0:(x0+w), :]

    return img_cropped


def image_preprocessing(img):
    return normalize_image(crop_resize_img(img))


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

        if (i+1) % (T//10) == 0:
            print(f"Noicing data {np.round(100*(i+1)/T, 3)}% with mean={np.round(mu,3)} and var={np.round(sigma,3)}...")
            images.append(curr_img)
        prev_img = curr_img

    return images

import numpy as np
import torch
import cv2


def normalize_image(img):
    return 2*(np.float32(img)/255) - 1.0


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


def noise_images(img, beta):
    return img + torch.sqrt(beta) * torch.randn(size=img.shape)

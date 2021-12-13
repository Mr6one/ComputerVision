import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread, imsave
from skimage import filters
from skimage.filters import sobel
from skimage.color import rgb2gray

from tqdm.notebook import tqdm
from .fast_hough_transform import fast_hough_transform


def rotate_image(image, angle, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE):
    width, height = image.shape[:2]

    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, rotation_matrix, (height, width), flags=flags, borderMode=borderMode)

    return rotated


def transform_images(path):
    image_names = os.listdir(path)
    angles = []
    for image_name in tqdm(image_names):
        image = imread(os.path.join(path, image_name))
        gray_image = rgb2gray(image)
        edges = sobel(gray_image)

        hough_space = fast_hough_transform(edges)
        angle = np.arctan((hough_space.var(axis=0).argmax() - hough_space.shape[1] // 2) / image.shape[1]) * 180 / np.pi
        angles.append(angle)

    angles = np.array(angles)
    return angles


def apply_and_save(path, angles):
    image_names = os.listdir(path)
    for angle, image_name in tqdm(zip(angles, image_names)):
        image = imread(os.path.join(path, image_name))

        rotated = rotate_image(image, angle, flags=cv2.INTER_NEAREST)
        imsave(os.path.join('transformed_images', 'nearest neighbor interpolation', image_name), rotated)

        rotated = rotate_image(image, angle, flags=cv2.INTER_LINEAR)
        imsave(os.path.join('transformed_images', 'billinear interpolation', image_name), rotated)


def show_images(path):
    image_names = os.listdir(path)
    plt.figure(figsize=(16, 20))
    for i, image_name in tqdm(enumerate(image_names)):
        image = imread(os.path.join(path, image_name))

        plt.subplot(5, 2, i + 1)
        plt.imshow(image)
        plt.axis('off')

    plt.show()

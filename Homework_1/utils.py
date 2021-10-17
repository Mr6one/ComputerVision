import numpy as np


def pad_image(image):
    image = np.concatenate((image[:2, :], image))
    image = np.concatenate((image, image[-2:, :]))
    image = np.concatenate((image[:, :2], image), axis=1)
    image = np.concatenate((image, image[:, -2:]), axis=1)
    return image


def get_color(pixel, color):
    if color == 'r':
        return pixel[0]
    elif color == 'g':
        return pixel[1]
    else:
        return pixel[2]
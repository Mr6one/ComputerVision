import numpy as np
from .utils import pad_image


def countsort(unsorted):
    return np.repeat(np.arange(1+unsorted.max()), np.bincount(unsorted))


def median_filter(image, R):
    r = R//2
    restored_image = np.zeros(image.shape)
    padded_image = pad_image(image, R)
    for c in range(3):
        for i in range(r, restored_image.shape[0] + r):
            for j in range(r, restored_image.shape[1] + r):
                window = padded_image[i - r:i + r + 1, j - r:j + r + 1, c].reshape(-1)
                restored_image[i - r, j - r, c] = countsort(window)[R * R // 2]

    return restored_image.astype(int)

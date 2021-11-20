import numpy as np
from .utils import pad_image
from .histogram_array import HistogramArray


def median_filter(image, R):
    r = R// 2
    restored_image = np.zeros(image.shape)
    padded_image = pad_image(image, R)
    for c in range(3):
        for i in range(r, restored_image.shape[0] + r):

            window = padded_image[i - r:i + r + 1, :R, c].reshape(-1)
            histogram_array = HistogramArray(window)
            restored_image[i - r, 0, c] = histogram_array.median()

            for j in range(1 + r, restored_image.shape[1] + r):
                histogram_array.delete(padded_image[i - r:i + r + 1, j - r - 1, c])
                histogram_array.add(padded_image[i - r:i + r + 1, j + r, c])
                restored_image[i - r, j - r, c] = histogram_array.median()

    return restored_image.astype(int)

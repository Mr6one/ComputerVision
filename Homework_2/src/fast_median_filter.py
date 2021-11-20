import numpy as np
from .utils import pad_image
from .histogram_array import HistogramArray


def median_filter(image, R):
    r = R // 2
    restored_image = np.zeros(image.shape)
    padded_image = pad_image(image, R)
    for c in range(3):
        column_histograms = [np.bincount(padded_image[:R - 1, i, c], minlength=256) for i in range(padded_image.shape[1])]
        column_histograms = np.array(column_histograms)

        for i in range(r, restored_image.shape[0] + r):

            window = padded_image[i - r:i + r + 1, :R, c]
            histogram_array = HistogramArray(window.reshape(-1))
            restored_image[i - r, 0, c] = histogram_array.median()

            for j in range(-r, restored_image.shape[1] + r):
                if i > r:
                    column_histograms[j + r, padded_image[i - r - 1, j + r, c]] -= 1
                column_histograms[j + r, padded_image[i + r, j + r, c]] += 1

                if j < r + 1:
                    continue

                histogram_array.bincounts -= column_histograms[j - r - 1]
                histogram_array.bincounts += column_histograms[j + r]
                restored_image[i - r, j - r, c] = histogram_array.median()

    return restored_image.astype(int)

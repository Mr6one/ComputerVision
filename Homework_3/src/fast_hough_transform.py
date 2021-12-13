import numpy as np


def roll_columns(matrix, r):
    row_indices, columns = np.ogrid[:matrix.shape[0], :matrix.shape[1]]
    r[r < 0] += matrix.shape[1]
    row_indices = row_indices - r
    result = matrix[row_indices, columns]
    return result


def hough_vectorized(image, left, right):
    result = np.zeros((image.shape[0], right - left + 1))

    if right - left == 1:
        result[:, 0] = image[:, left]
    else:
        mid = right - (right - left) // 2
        left_column = hough_vectorized(image, left, mid)
        right_column = hough_vectorized(image, mid, right)

        shift = np.arange(right - left + 1).astype(int)
        r = (shift // 2 + shift % 2)
        rolled_matrix = roll_columns(right_column[:, shift // 2 - shift % 2], r)

        result[:, shift] = left_column[:, shift // 2] + rolled_matrix

    return result


def fast_hough_transform(image):
    hough_space = hough_vectorized(image, 0, image.shape[1])
    hough_space_horizontal_flip = hough_vectorized(image[:, ::-1], 0, image.shape[1])

    return np.hstack([hough_space[:, ::-1], hough_space_horizontal_flip])

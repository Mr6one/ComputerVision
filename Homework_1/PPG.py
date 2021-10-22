import numpy as np
from tqdm.notebook import tqdm

def normalize(value):
    if value < 0:
        return 0
    elif value > 255:
        return 255
    else:
        return value


def green_interpolation(cell, i=0):
    N = abs(cell[2, 2, i] - cell[0, 2, i]) * 2 + abs(cell[1, 2, 1] - cell[3, 2, 1])
    E = abs(cell[2, 2, i] - cell[2, 4, i]) * 2 + abs(cell[2, 1, 1] - cell[2, 3, 1])
    W = abs(cell[2, 2, i] - cell[2, 0, i]) * 2 + abs(cell[2, 1, 1] - cell[2, 3, 1])
    S = abs(cell[2, 2, i] - cell[4, 2, i]) * 2 + abs(cell[1, 2, 1] - cell[3, 2, 1])

    argmin = np.argmin([N, E, W, S])
    if argmin == 0:
        return (cell[1, 2, 1] * 3 + cell[3, 2, 1] + cell[2, 2, i] - cell[0, 2, i]) / 4
    elif argmin == 1:
        return (cell[2, 3, 1] * 3 + cell[2, 1, 1] + cell[2, 2, i] - cell[2, 4, i]) / 4
    elif argmin == 2:
        return (cell[2, 1, 1] * 3 + cell[2, 3, 1] + cell[2, 2, i] - cell[2, 0, i]) / 4
    else:
        return (cell[3, 2, 1] * 3 + cell[1, 2, 1] + cell[2, 2, i] - cell[4, 2, i]) / 4


def hue_transit(L1, L2, L3, V1, V3):
    if L1 < L2 < L3 or L1 > L2 > L3:
        return V1 + (V3 - V1) * (L2 - L1) / (L3 - L1)
    else:
        return (V1 + V3) / 2 + (L2 - (L1 + L3) / 2) / 2


def red_blue_interpolation(cell, swap=False):
    i, j = 0, 2
    if swap:
        i, j = j, i
    r = hue_transit(cell[1, 2, 1], cell[2, 2, 1], cell[3, 2, 1], cell[1, 2, i], cell[3, 2, i])
    b = hue_transit(cell[2, 1, 1], cell[2, 2, 1], cell[2, 3, 1], cell[2, 1, j], cell[2, 3, j])
    if swap:
        r, b = b, r
    return r, b


def color_interpolation(cell, swap=False):
    i, j = 0, 2
    if swap:
        i, j = j, i

    NE = abs(cell[1, 3, j] - cell[3, 1, j]) + abs(cell[0, 4, i] - cell[2, 2, i]) \
         + abs(cell[2, 2, i] - cell[4, 0, i]) + abs(cell[1, 3, 1] - cell[2, 2, 1]) \
         + abs(cell[3, 1, 1] - cell[2, 2, 1])
    NW = abs(cell[1, 3, j] - cell[3, 1, j]) + abs(cell[0, 0, i] - cell[2, 2, i]) \
         + abs(cell[2, 2, i] - cell[4, 4, i]) + abs(cell[1, 1, 1] - cell[2, 2, 1]) \
         + abs(cell[3, 3, 1] - cell[2, 2, 1])

    if NE < NW:
        return hue_transit(cell[1, 3, 1], cell[2, 2, 1], cell[3, 1, 1], cell[1, 3, j], cell[3, 1, j])
    else:
        return hue_transit(cell[1, 1, 1], cell[2, 2, 1], cell[3, 3, 1], cell[1, 1, j], cell[3, 3, j])


def green_reconstruction(cell, bayer_filter, reconstructed_image, i, j):
    if bayer_filter[0, 0] == 'r':
        value = green_interpolation(cell)
        reconstructed_image[i - 2, j - 2, 1] = normalize(value)
    elif bayer_filter[0, 0] == 'b':
        value = green_interpolation(cell, 2)
        reconstructed_image[i - 2, j - 2, 1] = normalize(value)


def reb_blue_reconstruction(cell, bayer_filter, reconstructed_image, i, j):
    if bayer_filter[0, 0] == 'g':
        if bayer_filter[1, 0] == 'r':
            r, b = red_blue_interpolation(cell)
            reconstructed_image[i - 2, j - 2, (0, 2)] = normalize(r), normalize(b)
        else:
            r, b = red_blue_interpolation(cell, True)
            reconstructed_image[i - 2, j - 2, (0, 2)] = normalize(r), normalize(b)


def color_reconstraction(cell, bayer_filter, reconstructed_image, i, j):
    if bayer_filter[0, 0] == 'r':
        value = color_interpolation(cell)
        reconstructed_image[i - 2, j - 2, 2] = normalize(value)
    elif bayer_filter[0, 0] == 'b':
        value = color_interpolation(cell, True)
        reconstructed_image[i - 2, j - 2, 0] = normalize(value)


def interpolation(padded_image, reconstructed_image, bayer_filter, reconstruction):
    for i in tqdm(range(2, padded_image.shape[0] - 2)):
        for j in range(2, padded_image.shape[1] - 2):
            cell = padded_image[i - 2:i + 3, j - 2:j + 3]
            reconstruction(cell, bayer_filter, reconstructed_image, i, j)
            bayer_filter[:, (0, 1)] = bayer_filter[:, (1, 0)]

        bayer_filter[(0, 1), :] = bayer_filter[(1, 0), :]


def pad_image(image):
    image = np.concatenate((image[:2, :][::-1, :, :], image))
    image = np.concatenate((image, image[-3:, :][::-1, :, :]))
    image = np.concatenate((image[:, :2][:, ::-1, :], image), axis=1)
    image = np.concatenate((image, image[:, -2:][:, ::-1, :]), axis=1)
    return image


def reconstruct(image, bayer_filter):
    padded_image = pad_image(image).astype(float)
    reconstructed_image = image.copy().astype(float)
    reconstructed_image = np.concatenate((reconstructed_image, reconstructed_image[-1:, :]))

    interpolation(padded_image, reconstructed_image, bayer_filter, green_reconstruction)
    interpolation(padded_image, reconstructed_image, bayer_filter, reb_blue_reconstruction)
    interpolation(padded_image, reconstructed_image, bayer_filter, color_reconstraction)

    reconstructed_image = reconstructed_image[:-1, :]
    reconstructed_image[reconstructed_image < 0] = 0
    reconstructed_image[reconstructed_image > 255] = 255
    reconstructed_image = reconstructed_image.astype(np.uint8)

    return reconstructed_image

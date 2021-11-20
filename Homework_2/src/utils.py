import numpy as np


def salt_and_pepper_noise(image, prob):
    output = image.copy()
    black = np.array([0, 0, 0], dtype='uint8')
    white = np.array([255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    output[probs < prob] = black
    output[probs > 1 - prob] = white
    return output


def pad_image(image, R):
    image = np.concatenate((image[:R//2, :], image))
    image = np.concatenate((image, image[-R//2:, :]))
    image = np.concatenate((image[:, :R//2], image), axis=1)
    image = np.concatenate((image, image[:, -R//2:]), axis=1)
    return image

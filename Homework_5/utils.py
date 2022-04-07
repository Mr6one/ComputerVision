import os
import cv2
import time
import numpy as np

from keypoints import detect_keypoints, match_keypoints, match_shi_tomasi


def read_images(path):
    images = []
    for filename in os.listdir(path):
        path_to_image = os.path.join(path, filename)
        image = cv2.imread(path_to_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

    return images


def get_repeatability(images, method):
    image0 = images[0].copy()

    if method in ['sift', 'orb']:
        keypoints0, descriptors0 = detect_keypoints(image0, method=method)
    elif method == 'shi-tomasi':
        keypoints0 = detect_keypoints(image0, method='shi-tomasi')[0]
    else:
        raise ValueError('Invalid method')

    repeatability_per_image = np.ones(len(keypoints0))
    repeatability = [repeatability_per_image.mean()]
    for j in range(1, len(images)):
        image1 = images[j].copy()

        if method in ['sift', 'orb']:
            keypoints1, descriptors1 = detect_keypoints(image1, method=method)
            matches = match_keypoints(descriptors0, descriptors1)
        else:
            matches = match_shi_tomasi(image0, image1, keypoints0)[2]

        repeatability_per_image[matches[:, 0]] += 1
        repeatability.append((repeatability_per_image / (j + 1)).mean())

    repeatability = np.array(repeatability)
    return repeatability


def measure_time(images, method):
    start = time.time()
    num_keypoints = np.sum([len(detect_keypoints(image.copy(), method=method)[0]) for image in images])
    end = time.time()
    time_per_keypoints = (end - start) / num_keypoints
    return time_per_keypoints

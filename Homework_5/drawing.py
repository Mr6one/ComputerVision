import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image(image):
    plt.imshow(image)
    plt.axis('off')


def plot_keypoints(image, keypoints, radius=1, color=(0, 255, 0)):
    for keypoint in keypoints:
        image = cv2.circle(image, keypoint.astype(int), radius=radius, color=color, thickness=-1)

    return image


def draw_matches(image0, image1, kpts0, kpts1, matches, border=0):
    kpts0 = kpts0.astype(int)
    kpts1 = kpts1.astype(int)

    mkpts0 = kpts0[matches[:, 0]]
    mkpts1 = kpts1[matches[:, 1]]

    image0 = plot_keypoints(image0, mkpts0, color=(0, 255, 0), radius=2)
    image1 = plot_keypoints(image1, mkpts1, color=(0, 255, 0), radius=2)

    h, w = image0.shape[:2]
    stacked_image = np.zeros((max(image0.shape[0], image1.shape[0]), image0.shape[1] + image1.shape[1] + border, 3))
    stacked_image[:h, :w] = image0
    stacked_image[:h, w + border:] = image1

    for kpts0, kpts1 in zip(mkpts0, mkpts1):
        stacked_image = cv2.line(stacked_image, kpts0, (kpts1[0] + w + border, kpts1[1]), color=(0, 0, 255),
                                 thickness=1)

    plt.figure(figsize=(16, 12))
    show_image(stacked_image.astype(np.uint8))
    plt.show()

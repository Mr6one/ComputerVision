import cv2
import numpy as np


def detect_keypoints(image, method='sift'):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    descriptors = None
    if method in ['sift', 'orb']:
        if method == 'sift':
            detector = cv2.SIFT_create()
        else:
            detector = cv2.ORB_create()
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        keypoints = np.array([keypoint.pt for keypoint in keypoints])
    elif method == 'shi-tomasi':
        keypoints = cv2.goodFeaturesToTrack(gray, maxCorners=1024, qualityLevel=0.01, minDistance=10)
        keypoints = keypoints.reshape(-1, 2)
    else:
        raise ValueError('Invalid method')

    return keypoints, descriptors


def match_keypoints(desc0, desc1):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.match(desc0, desc1)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = np.array([(match.queryIdx, match.trainIdx) for match in matches])
    return matches


def match_shi_tomasi(image0, image1, keypoints0=None):
    if keypoints0 is None:
        keypoints0 = detect_keypoints(image0, method='shi-tomasi')[0]

    lk_params = {
        'winSize': (15, 15),
        'maxLevel': 2,
        'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    }
    gray0 = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

    keypoints1, status, err = cv2.calcOpticalFlowPyrLK(gray0, gray1, keypoints0, None, **lk_params)
    matches = np.arange(len(keypoints0))[:, None]
    matches = np.concatenate((matches, matches), axis=1)
    matches = matches[status.ravel() == 1]

    return keypoints0, keypoints1, matches

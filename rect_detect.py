import sys
from collections import defaultdict

import cv2
import numpy as np

'''
:argument HSV masked binary image

It's more or less noisy, so these operation are made before HoughLines:

    1. Erosion/dilation - for noise removal, with large kernel, because we receive image with white pixels on whole 
                          colored area, so we will cut edges with erosion (which are noisy) and then will add them 
                          back with dilation 
    
    2. Canny edge - for edge detection, lower and upper threshold are computed automatically
    
Then we are detecting lines with HoughLines function and finding their intersection (by help of StackOverFlow).    

:return found segments, interceptions, edged image      
'''


def rect_detect(threshold_img):
    kernel_ero = np.ones((7, 7), np.uint8)
    kernel_dil = np.ones((7, 7), np.uint8)

    img_erosion = cv2.erode(threshold_img, kernel_ero, iterations=3)
    img_dilation = cv2.dilate(img_erosion, kernel_dil, iterations=3)

    SIGMA = 0.33

    # compute the median of the single channel pixel intensities
    v = np.median(img_dilation)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - SIGMA) * v))
    upper = int(min(255, (1.0 + SIGMA) * v))
    edged = cv2.Canny(img_dilation, lower, upper)

    lines = cv2.HoughLines(edged, 1, np.pi / 180, 40)

    intersections = None
    if lines is not None:
        # Cluster line angles into 2 groups (vertical and horizontal)
        segmented = segment_by_angle_kmeans(lines, 2)

        # Find the intersections of each vertical line with each horizontal line
        intersections = segmented_intersections(segmented)

    return intersections, edged


def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """
    Group lines by their angle using k-means clustering.

    Code from here:
    https://stackoverflow.com/a/46572063/1755401
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))

    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # Get angles in [0, pi] radians
    angles = np.array([line[0][1] for line in lines])

    # Multiply the angles by two and find coordinates of that angle on the Unit Circle
    pts = np.array([[np.cos(2 * angle), np.sin(2 * angle)] for angle in angles], dtype=np.float32)

    # Run k-means
    if sys.version_info[0] == 2:
        # python 2.x
        ret, labels, centers = cv2.kmeans(pts, k, criteria, attempts, flags)
    else:
        # python 3.x, syntax has changed.
        labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]

    labels = labels.reshape(-1)  # Transpose to row vector

    # Segment lines based on their label of 0 or 1
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)

    segmented = list(segmented.values())

    return segmented


def intersection(line1, line2):
    """
    Find the intersection of two lines
    specified in Hesse normal form.

    Returns closest integer pixel locations.

    See here:
    https://stackoverflow.com/a/383527/5087436
    """
    try:
        rho1, theta1 = line1[0]
        rho2, theta2 = line2[0]
        A = np.array([[np.cos(theta1), np.sin(theta1)],
                      [np.cos(theta2), np.sin(theta2)]])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [[x0, y0]]
    except np.linalg.LinAlgError:
        pass


def segmented_intersections(lines):
    """
    Find the intersection between groups of lines.
    """

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i + 1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))

    return intersections

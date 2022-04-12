import cv2
import numpy as np
from distance import dist


def rect_detect(threshold_img):
    SIGMA = 0.33

    # compute the median of the single channel pixel intensities
    v = np.median(threshold_img)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - SIGMA) * v))
    upper = int(min(255, (1.0 + SIGMA) * v))
    edged = cv2.Canny(threshold_img, lower, upper)

    # Taking a matrix of size 5 as the kernel
    kernel_ero = np.ones((1, 1), np.uint8)
    kernel_dil = np.ones((9, 9), np.uint8)

    img_erosion = cv2.erode(edged, kernel_ero, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel_dil, iterations=1)

    cnts = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    min_area = 1500
    max_area = 40000
    chosen = None
    for c in cnts:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            x_r, y_r, w, h = cv2.boundingRect(c)
            rect = (x_r, y_r, w, h)
            chosen = rect

    # return the edged image
    return chosen



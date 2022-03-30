import cv2
import numpy as np
from distance import dist


def rect_detect(img, x, y):
    SIGMA = 0.33

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 9)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

    thresh = cv2.threshold(sharpen, 200, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # compute the median of the single channel pixel intensities
    v = np.median(close)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - SIGMA) * v))
    upper = int(min(255, (1.0 + SIGMA) * v))
    edged = cv2.Canny(close, lower, upper)

    # Taking a matrix of size 5 as the kernel
    kernel_ero = np.ones((1, 1), np.uint8)
    kernel_dil = np.ones((9, 9), np.uint8)

    img_erosion = cv2.erode(edged, kernel_ero, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel_dil, iterations=1)

    cnts = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    min_area = 1500
    max_area = 10000
    chosen = None
    for c in cnts:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            x_r, y_r, w, h = cv2.boundingRect(c)
            rect = (x_r, y_r)
            distance = dist(rect, (x, y))
            if distance < 1000:
                chosen = rect

    # return the edged image
    return chosen



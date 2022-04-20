import cv2
import numpy as np


def circle_detect(threshold_img):

    kernel_ero = np.ones((5, 5), np.uint8)
    kernel_dil = np.ones((5, 5), np.uint8)

    img_erosion = cv2.erode(threshold_img, kernel_ero, iterations=3)
    img_dilation = cv2.dilate(img_erosion, kernel_dil, iterations=3)

    circles = cv2.HoughCircles(img_dilation, cv2.HOUGH_GRADIENT, 1, minDist=100, param1=30, param2=15, minRadius=0)

    return circles, img_dilation

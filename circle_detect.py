import cv2
import numpy as np

from distance import dist


def circle_detect(threshold_img):
    circles = cv2.HoughCircles(threshold_img, cv2.HOUGH_GRADIENT, 1, minDist=100, param1=30, param2=15, minRadius=0)
    return circles

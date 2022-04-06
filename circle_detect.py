import cv2
import numpy as np

from distance import dist


def circle_detect(img, x, y):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray, (5, 5))
    v = np.median(blurred)
    sigma = 0.3
    upper = int(min(255, (1.0 + sigma) * v))
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist=100, param1=upper, param2=20, minRadius=50)
    chosen = None
    if circles is not None:
        circles = np.uint16(np.around(circles))
        min_distance = 1000
        min_rad = 2000
        for circle in circles[0, :]:
            distance = dist(circle, (x, y))
            if distance < circle[2] < min_rad and min_distance > distance:
                min_distance = distance
                min_rad = circle[2]
                chosen = circle

    return chosen

import cv2
import numpy as np


'''
:argument HSV masked binary image

It's more or less noisy, so these operation are made before HoughLines:

    1. Erosion/dilation - for noise removal, with large kernel, because we receive image with white pixels on whole 
                          colored area, so we will cut edges with erosion (which are noisy) and then will add them 
                          back with dilation 
    
    2. Canny edge - for edge detection, lower and upper threshold are computed automatically
    
Then we are detecting lines with HoughLinesP function.    

:return found lines, edged image      
'''


def rect_detect(threshold_img):

    kernel_ero = np.ones((12, 12), np.uint8)
    kernel_dil = np.ones((12, 12), np.uint8)

    img_erosion = cv2.erode(threshold_img, kernel_ero, iterations=3)
    img_dilation = cv2.dilate(img_erosion, kernel_dil, iterations=3)

    SIGMA = 0.33

    # compute the median of the single channel pixel intensities
    v = np.median(img_dilation)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - SIGMA) * v))
    upper = int(min(255, (1.0 + SIGMA) * v))
    edged = cv2.Canny(img_dilation, lower, upper)

    minLineLength = 30
    maxLineGap = 10
    lines = cv2.HoughLinesP(edged, 1, np.pi / 180, 15, minLineLength=minLineLength, maxLineGap=maxLineGap)

    return lines, edged



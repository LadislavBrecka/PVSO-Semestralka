import cv2
import imutils
import numpy as np

from circle_detect import circle_detect
from color import Colors
from rect_detect import rect_detect
from shape import Shapes


def detect(img, color: Colors, shape: Shapes):

    # Convert to the HSV color space
    blur = cv2.medianBlur(img, 5)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a mask based on medium to high Saturation and Value
    # Hue 100-130 is close to blue, which we are detecting
    # These values can be changed (the lower ones) to better fit environment
    # TODO: manipulate with this to get better matching
    thresh = cv2.inRange(hsv, (color.value.h_range[0], 100, 100), (color.value.h_range[1], 255, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Finds contours and converts it to a list
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Loops over all objects found
    for contour in contours:

        # Skip if contour is small (can be adjusted)
        if cv2.contourArea(contour) < 350:
            continue

        if shape == Shapes.CIRCLE:
            circles = circle_detect(thresh)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    cv2.circle(img, (circle[0], circle[1]), circle[2], color.value.mark_bgr, 3)

        if shape == Shapes.RECTANGLE:
            chosen = rect_detect(thresh)
            if chosen is not None:
                cv2.rectangle(img, (chosen[0], chosen[1]), (chosen[0] + chosen[2], chosen[1] + chosen[3]), color.value.mark_bgr, 2)

    return img, thresh

import cv2
import imutils
import numpy as np

from circle_detect import circle_detect
from color import Colors
from rect_detect import rect_detect
from shape import Shapes


def detect(img, color: Colors, shape: Shapes, additional_img_selector):

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

    additional_img = thresh

    # Loops over all objects found
    for contour in contours:

        # Skip if contour is small (can be adjusted)
        if cv2.contourArea(contour) < 350:
            continue

        if shape == Shapes.CIRCLE:
            if additional_img_selector == 2:
                circles, additional_img = circle_detect(thresh)
            else:
                circles, _ = circle_detect(thresh)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    cv2.circle(img, (circle[0], circle[1]), circle[2], color.value.mark_bgr, 3)

        if shape == Shapes.RECTANGLE:
            if additional_img_selector == 2:
                intersections,  additional_img = rect_detect(thresh)
            else:
                intersections, _ = rect_detect(thresh)
            if intersections is not None:
                # Draw intersection points in magenta
                for point in intersections:
                    try:
                        pt = (point[0][0], point[0][1])
                        length = 5
                        cv2.line(img, (pt[0], pt[1] - length), (pt[0], pt[1] + length), color.value.mark_bgr, 1)
                        cv2.line(img, (pt[0] - length, pt[1]), (pt[0] + length, pt[1]), color.value.mark_bgr, 1)
                    except cv2.error:
                        pass

    return img, additional_img

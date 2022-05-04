import cv2
import imutils
import numpy as np
import math
from circle_detect import circle_detect
from color import Colors
from rect_detect import rect_detect
from shape import Shapes


def detect(img, color: Colors, shape: Shapes, additional_img_selector):
    # Convert to the HSV color space
    blur = cv2.medianBlur(img, 5)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a mask based on specified values in color.py
    # These values can be changed (the lower ones) to better fit environment
    thresh = cv2.inRange(hsv, (color.value.h_range[0], color.value.s_range[0], color.value.v_range[0]),
                         (color.value.h_range[1], color.value.s_range[1], color.value.v_range[1]))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
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
                intersections, additional_img = rect_detect(thresh)
            else:
                intersections, _ = rect_detect(thresh)
                if intersections is not None:
                    if len(intersections) == 4:
                        distance0 = math.dist([intersections[0][0][0], intersections[0][0][1]],
                                              [intersections[1][0][0], intersections[1][0][1]])
                        distance1 = math.dist([intersections[0][0][0], intersections[0][0][1]],
                                              [intersections[2][0][0], intersections[2][0][1]])
                        distance2 = math.dist([intersections[0][0][0], intersections[0][0][1]],
                                              [intersections[3][0][0], intersections[3][0][1]])
                        distance3 = math.dist([intersections[1][0][0], intersections[1][0][1]],
                                              [intersections[2][0][0], intersections[2][0][1]])
                        distance4 = math.dist([intersections[2][0][0], intersections[2][0][1]],
                                              [intersections[3][0][0], intersections[3][0][1]])
                        distance5 = math.dist([intersections[1][0][0], intersections[1][0][1]],
                                              [intersections[3][0][0], intersections[3][0][1]])

                        dists = [distance0, distance1, distance2, distance3, distance4, distance5]

                        max1, max2, d_prev = None, None, 0
                        for i in range(0, 6):
                            if dists[i] > d_prev:
                                d_prev = dists[i]
                                max1 = i
                        d_prev = 0
                        for i in range(0, 6):
                            if dists[i] > d_prev and max1 != i:
                                d_prev = dists[i]
                                max2 = i

                        if max1 != 0 and max2 != 0:
                            cv2.line(img, (intersections[0][0][0], intersections[0][0][1]),
                                     (intersections[1][0][0], intersections[1][0][1]), color.value.mark_bgr, 4)
                        if max1 != 1 and max2 != 1:
                            cv2.line(img, (intersections[0][0][0], intersections[0][0][1]),
                                     (intersections[2][0][0], intersections[2][0][1]), color.value.mark_bgr, 4)
                        if max1 != 2 and max2 != 2:
                            cv2.line(img, (intersections[0][0][0], intersections[0][0][1]),
                                     (intersections[3][0][0], intersections[3][0][1]), color.value.mark_bgr, 4)
                        if max1 != 3 and max2 != 3:
                            cv2.line(img, (intersections[2][0][0], intersections[2][0][1]),
                                     (intersections[1][0][0], intersections[1][0][1]), color.value.mark_bgr, 4)
                        if max1 != 4 and max2 != 4:
                            cv2.line(img, (intersections[2][0][0], intersections[2][0][1]),
                                     (intersections[3][0][0], intersections[3][0][1]), color.value.mark_bgr, 4)

                        if max1 != 5 and max2 != 5:
                            cv2.line(img, (intersections[3][0][0], intersections[3][0][1]),
                                     (intersections[1][0][0], intersections[1][0][1]), color.value.mark_bgr, 4)

    return img, additional_img




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

    # Create a mask based on medium to high Saturation and Value
    # Hue 100-130 is close to blue, which we are detecting
    # These values can be changed (the lower ones) to better fit environment
    # TODO: manipulate with this to get better matching
    thresh = cv2.inRange(hsv, (color.value.h_range[0], 100, 100), (color.value.h_range[1], 255, 255))

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
            radius = 50
            if intersections is not None:
                # Draw intersection points in magenta
                for point in intersections:
                    try:
                        pt = (point[0][0], point[0][1])
                        length = 5
                        # cv2.line(img, (pt[0], pt[1] - length), (pt[0], pt[1] + length), color.value.mark_bgr, 1)
                        # cv2.line(img, (pt[0] - length, pt[1]), (pt[0] + length, pt[1]), color.value.mark_bgr, 1)
                        point_number = 0
                        for point1 in intersections:
                            try:
                                if math.dist([point[0][0], point[0][1]], [point1[0][0], point1[0][1]]) < radius:
                                    point1[0][0] = point[0][0]
                                    point1[0][1] = point[0][1]
                                    point_number += 1

                            except cv2.error:
                                pass
                            except TypeError:
                                pass
                        if point_number == 0:
                            intersections.remove(point)
                    except cv2.error:
                        pass
                    except TypeError:
                        pass

                intersections = without_duplicates(intersections)
                for point in intersections:
                    try:
                        pt = (point[0][0], point[0][1])
                        length = 5
                        cv2.line(img, (pt[0], pt[1] - length), (pt[0], pt[1] + length), color.value.mark_bgr, 1)
                        cv2.line(img, (pt[0] - length, pt[1]), (pt[0] + length, pt[1]), color.value.mark_bgr, 1)
                    except cv2.error:
                        pass
                    except TypeError:
                        pass

                if len(intersections) == 4:
                    cv2.line(img, (intersections[0][0][0], intersections[0][0][1]), (intersections[1][0][0], intersections[1][0][1]), color.value.mark_bgr, 1)
                    cv2.line(img, (intersections[1][0][0], intersections[1][0][1]), (intersections[2][0][0], intersections[2][0][1]), color.value.mark_bgr, 1)
                    cv2.line(img, (intersections[2][0][0], intersections[2][0][1]), (intersections[3][0][0], intersections[3][0][1]), color.value.mark_bgr, 1)
                    cv2.line(img, (intersections[3][0][0], intersections[3][0][1]), (intersections[0][0][0], intersections[0][0][1]), color.value.mark_bgr, 1)





    return img, additional_img


def without_duplicates(objs):
    if len(objs) > 1:
        objs = sorted(objs)
        last = objs[0]
        result = [last]
        for current in objs[1:]:
            if current != last:
                result.append(current)
            last = current
        return result
    else:
        return objs

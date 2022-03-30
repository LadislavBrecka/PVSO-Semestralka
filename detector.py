import cv2
import imutils

from circle_detect import circle_detect
from rect_detect import rect_detect


def detect(img, color, shape):

    # Convert to the HSV color space
    blur = cv2.medianBlur(img, 5)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a mask based on medium to high Saturation and Value
    # Hue 100-130 is close to blue, which we are detecting
    # These values can be changed (the lower ones) to better fit environment
    # TODO: manipulate with this to get better matching
    mask = cv2.inRange(hsv, (color.h_range[0], 100, 100), (color.h_range[1], 255, 255))

    # Dilates with two iterations (makes it more visible)
    thresh = cv2.dilate(mask, None, iterations=2)

    # Finds contours and converts it to a list
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Loops over all objects found
    for contour in contours:

        # Skip if contour is small (can be adjusted)
        if cv2.contourArea(contour) < 750:
            continue

        # Get the box boundaries
        (x, y, w, h) = cv2.boundingRect(contour)

        # Compute size
        size = (h + w) // 2 // 2

        if shape == 'CIRCLE':
            chosen = circle_detect(img, x, y)
            if chosen is not None:
                cv2.circle(img, (x+size, y+size), size, color.bgr, 3)

        if shape == 'RECT':
            chosen = rect_detect(img, x, y)
            if chosen is not None:
                ROI = img[y:y + h, x:x + w]
                cv2.rectangle(img, (x, y), (x + w, y + h), color.bgr, 2)

    return img

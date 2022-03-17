import cv2
import numpy as np


def main():
    # naming of windows
    app_name = "PVSO-Semestralka"
    window_1_name = "1"

    # setting up camera and capturing object
    imcap = cv2.VideoCapture(0)
    imcap.set(3, 640)  # set width as 640
    imcap.set(4, 480)  # set height as 480

    # Create the background subtractor object
    # Use the last 700 video frames to build the background
    back_sub = cv2.createBackgroundSubtractorMOG2(history=700, varThreshold=25, detectShadows=True)

    # application loop
    while True:

        # capture frame from video
        _, img = imcap.read()

        # Display the resulting frame
        cv2.imshow(app_name, img)

        # If "q" is pressed on the keyboard,
        # exit this loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # if loop was terminated, close and erase window
    imcap.release()
    cv2.destroyWindow(app_name)
    cv2.destroyWindow(window_1_name)


# call main function
main()

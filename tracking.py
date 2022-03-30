import cv2
import time

from config import *
from detector import detect


def main():
    # setting up camera and capturing object
    imcap = cv2.VideoCapture(0)
    imcap.set(3, width)  # set width as 640
    imcap.set(4, height)  # set height as 480
    imcap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    imcap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Time is just used to get the Frames Per Second (FPS)
    last_time = time.time()

    while True:
        # Capture the frame
        _, img = imcap.read()

        color_img = detect(img, GREEN, 'CIRCLE')
        # color_img = detect(img, RED, 'RECT')

        # Add a FPS label to image
        text = f"FPS: {int(1 / (time.time() - last_time))}"
        last_time = time.time()
        cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Detector", color_img)

        # If q is pressed terminate
        if cv2.waitKey(1) == ord('q'):
            break

    # Release and destroy all windows
    imcap.release()
    cv2.destroyAllWindows()


main()

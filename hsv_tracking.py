import cv2
import time
import imutils
import numpy as np


def main():
    # setting up camera and capturing object
    width = 640
    height = 480
    imcap = cv2.VideoCapture(0)
    imcap.set(3, width)  # set width as 640
    imcap.set(4, height)  # set height as 480
    imcap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    imcap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Read the marker to use later
    logo_org = cv2.imread('marker.png')

    # Time is just used to get the Frames Per Second (FPS)
    last_time = time.time()

    while True:
        # Capture the frame
        _, img = imcap.read()

        # Convert to the HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Create a mask based on medium to high Saturation and Value
        # Hue 100-130 is close to blue, which we are detecting
        # These values can be changed (the lower ones) to better fit environment
        # TODO: manipulate with this to get better matching
        mask = cv2.inRange(hsv, (100, 100, 100), (130, 255, 255))

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
            size = (h + w)//2

            # Check if marker will be inside frame
            if y + size < height and x + size < width:

                # Resize marker
                logo = cv2.resize(logo_org, (size, size))

                # Create a mask of marker
                img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
                _, logo_mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

                # Region of Image (ROI), where we want to insert marker
                roi = img[y:y+size, x:x+size]

                # Mask out marker region and insert
                roi[np.where(logo_mask)] = 0
                roi += logo

        # Add a FPS label to image
        text = f"FPS: {int(1 / (time.time() - last_time))}"
        last_time = time.time()
        cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Webcam", img)
        cv2.imshow("hsv", hsv)

        # If q is pressed terminate
        if cv2.waitKey(1) == ord('q'):
            break

    # Release and destroy all windows
    imcap.release()
    cv2.destroyAllWindows()


main()

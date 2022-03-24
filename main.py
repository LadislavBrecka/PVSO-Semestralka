import cv2
import numpy as np


def main():
    # setting up camera and capturing object
    imcap = cv2.VideoCapture(0)
    imcap.set(3, 640)  # set width as 640
    imcap.set(4, 480)  # set height as 480

    # Create the background subtractor object
    # Use the last 700 video frames to build the background
    back_sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)

    # application loop
    while True:
        # capture frame from video
        _, img = imcap.read()
        img2 = img.copy()
        # applying on each frame
        fg_mask = back_sub.apply(img)

        # Convert the image to gray-scale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Blur the image to reduce noise
        gausian_blur_img = cv2.GaussianBlur(gray_img, (5, 5), 1)
        median_blur_img = cv2.medianBlur(gray_img, 5)



        # apply automatic Canny edge detection using the computed median
        v = np.median(median_blur_img)
        sigma = 0.1
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(median_blur_img, lower, upper)

        contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), -1)  # ---set the last parameter to -1

        # detect circles
        gray2_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        blur_img2 = cv2.blur(gray2_img, (6,6))
        edged2 = cv2.Canny(blur_img2, lower, upper)
        threshold = cv2.threshold(blur_img2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        edged2 = cv2.Canny(threshold, lower, upper)
        circles = cv2.HoughCircles(blur_img2, cv2.HOUGH_GRADIENT, 1, minDist=2000, param1=200, param2=25, minRadius=5)

        if circles is not None:
            circles = np.uint16(np.around(circles))

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(img2, (x, y), r, (36, 255, 12), 3)


        # Display the resulting frames
        # cv2.imshow("main", img)
        # cv2.imshow("grayed", gray_img)
        # cv2.imshow("blurred", median_blur_img)
        cv2.imshow("canny", edged2)
        cv2.imshow("Circle Detect", img2)
        # cv2.imshow("blur", blur_img2)
        cv2.imshow('Threshold', threshold)
        # cv2.imshow("other", erode)

        # If "q" is pressed on the keyboard,
        # exit this loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # if loop was terminated, close and erase window
    imcap.release()
    cv2.destroyWindow("main")
    cv2.destroyWindow("grayed")
    cv2.destroyWindow("blurred")
    cv2.destroyWindow("canny")
    cv2.destroyWindow("other")


# call main function
main()

#
# _, thresh = cv2.threshold(gray_img, 240, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, cnts, -1, (0, 255, 0), 3)
# mask = np.zeros(img.shape[:2], dtype=np.uint8)
# cv2.drawContours(mask, cnts, -1, 255, 1)
# kernel = np.ones((100, 100), np.uint8)
# mask = cv2.dilate(mask, kernel, iterations=1)
# cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, cnts, 1, (255, 0, 0), 3)

import cv2
import time
import imutils
import numpy as np

RED = (150, 180)
GREEN = (50, 80)

SIGMA = 0.33

width = 640
height = 480
# Read the marker to use later
marker = cv2.imread('marker.png')
prevCircle = None


def dist(point1, point2):
    return np.linalg.norm(point1[0:2] - point2[0:2])


def color_coord(img, color):
    # Convert to the HSV color space
    blur = cv2.medianBlur(img, 5)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a mask based on medium to high Saturation and Value
    # Hue 100-130 is close to blue, which we are detecting
    # These values can be changed (the lower ones) to better fit environment
    # TODO: manipulate with this to get better matching
    mask = cv2.inRange(hsv, (color[0], 100, 100), (color[1], 255, 255))

    # Dilates with two iterations (makes it more visible)
    thresh = cv2.dilate(mask, None, iterations=2)

    # Finds contours and converts it to a list
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Loops over all objects found
    coordinates = []
    for contour in contours:

        # Skip if contour is small (can be adjusted)
        if cv2.contourArea(contour) < 750:
            continue

        # Get the box boundaries
        (x, y, w, h) = cv2.boundingRect(contour)
        coordinates.append((x, y))

    return coordinates


def color_detect(img, color):
    # Convert to the HSV color space
    blur = cv2.medianBlur(img, 5)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a mask based on medium to high Saturation and Value
    # Hue 100-130 is close to blue, which we are detecting
    # These values can be changed (the lower ones) to better fit environment
    # TODO: manipulate with this to get better matching
    mask = cv2.inRange(hsv, (color[0], 100, 100), (color[1], 255, 255))

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
        size = (h + w) // 2

        # Check if marker will be inside frame
        if y + size < height and x + size < width:
            # Resize marker
            res_marker = cv2.resize(marker, (size, size))

            # Create a mask of marker
            img2gray = cv2.cvtColor(res_marker, cv2.COLOR_BGR2GRAY)
            _, marker_mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

            # Region of Image (ROI), where we want to insert marker
            roi = img[y:y + size, x:x + size]

            # Mask out marker region and insert
            roi[np.where(marker_mask)] = 0
            roi += res_marker

    return img


def rect_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7)

    # compute the median of the single channel pixel intensities
    v = np.median(blurred)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - SIGMA) * v))
    upper = int(min(255, (1.0 + SIGMA) * v))
    edged = cv2.Canny(blurred, lower, upper)

    # Taking a matrix of size 5 as the kernel
    kernel_ero = np.ones((1, 1), np.uint8)
    kernel_dil = np.ones((7, 7), np.uint8)

    img_erosion = cv2.erode(edged, kernel_ero, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel_dil, iterations=1)

    lines = cv2.HoughLinesP(img_dilation, rho=1, theta=1 * np.pi / 180, threshold=100, minLineLength=250, maxLineGap=50)
    N = lines.shape[0]
    for i in range(N):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # return the edged image
    return img


def circle_detect(img):
    global prevCircle

    color_coordinates = color_coord(img, GREEN)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # blurred = cv2.blur(gray, (5, 5))
    # v = np.median(blurred)
    # sigma = 0.3
    # upper = int(min(255, (1.0 + sigma) * v))
    # # threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist=10, param1=upper, param2=25, minRadius=0)
    config = {
        "img_default_width": 600,
        "img_default_height": 400,
        "minDist": 22,
        "minRadius": 0,
        "maxRadius": 100,
        "gaussian_default": 5,
        "median_default": 7,
        "threshold": 0
    }

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (config["gaussian_default"], config["gaussian_default"]), 0)
    gray_image = cv2.medianBlur(gray_image, config["median_default"])
    # gray_image= cv2.GaussianBlur(gray_image, (7, 7), 0)
    ret, th1 = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
    th3 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 5)

    # https://www.pyimagesearch.com/2021/04/28/opencv-morphological-operations/#:~:text=Morphological%20operations%20are%20simple%20transformations,and%20structures%20inside%20of%20images.
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(th2, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)

    imgray = cv2.Canny(erosion, 30, 100)

    circles = cv2.HoughCircles(imgray, method=cv2.HOUGH_GRADIENT, dp=1, minDist=config["minDist"], param1=50, param2=30,
                               minRadius=config["minRadius"], maxRadius=config["maxRadius"])

    if circles is not None:
        circles = np.uint16(np.around(circles))
        min_distance = 1000
        min_rad = 2000
        chosen = None
        for circle in circles[0, :]:
            for coord in color_coordinates:
                distance = dist(circle, coord)
                print(distance)
                if distance < circle[2]*0.8 < min_rad and min_distance > distance:
                    min_distance = distance
                    min_rad = circle[2]
                    chosen = circle
            # cv2.circle(img, (circle[0], circle[1]), 1, (0, 100, 100), 3)
            # cv2.circle(img, (circle[0], circle[1]), circle[2], (255, 0, 255), 3)

        if chosen is not None:
            cv2.circle(img, (chosen[0], chosen[1]), 1, (0, 100, 100), 3)
            cv2.circle(img, (chosen[0], chosen[1]), chosen[2], (255, 0, 255), 3)

    return img


# # for coord in color_coordinates:
#     distance = dist(circle, coord)
#     print(distance)
#     if distance <= circle[2]*1.2:
#         cv2.circle(img, (circle[0], circle[1]), 1, (0, 100, 100), 3)
#         cv2.circle(img, (circle[0], circle[1]), circle[2], (255, 0, 255), 3)


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

        # color_img = color_detect(img, RED)
        img_color = img.copy()
        img_circle = img.copy()
        circle_img = circle_detect(img_circle)
        color_img = color_detect(img_color, GREEN)

        # Add a FPS label to image
        text = f"FPS: {int(1 / (time.time() - last_time))}"
        last_time = time.time()
        cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("circle", circle_img)
        cv2.imshow("color", color_img)

        # If q is pressed terminate
        if cv2.waitKey(1) == ord('q'):
            break

    # Release and destroy all windows
    imcap.release()
    cv2.destroyAllWindows()


main()

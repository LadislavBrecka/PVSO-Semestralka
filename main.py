import cv2
import numpy as np

# https://debuggercafe.com/moving-object-detection-using-frame-differencing-with-opencv/
# https://pysource.com/2021/01/28/object-tracking-with-opencv-and-python/


# low pass filter (programmed by Ladislav in 2021 - Furrier transform used)
def fft_filter(input_data, filter_const: int):
    furrier_transform = np.fft.fft2(input_data)
    shifted_furrier_transform = np.fft.fftshift(furrier_transform)
    hp_filter = np.zeros(shifted_furrier_transform.shape, dtype=int)
    filter_shape = hp_filter.shape
    n = int(filter_shape[0])
    m = int(filter_shape[1])
    n_center = int(n / 2)
    m_center = int(m / 2)
    n_shift = int((n_center / 100) * filter_const)
    m_shift = int((m_center / 100) * filter_const)
    hp_filter[m_center - m_shift: m_center + m_shift, n_center - n_shift: n_center + n_shift] = 1
    output = shifted_furrier_transform * hp_filter
    output = np.fft.ifftshift(output)
    output = abs(np.fft.ifft2(output))
    return np.int8(output)


def get_background(camera):
    _, cap = camera.read()
    # we will select random 50 frames from 500 for the calculating the median
    all_frames = []
    for i in range(0, 500):
        _, cap = camera.read()
        all_frames.append(cap)

    frames = []
    rand_ind = np.random.randint(0, 500, 50)
    for i in range(0, 50):
        frames.append(all_frames[rand_ind[i]])
    # calculate the median
    median_frame = np.median(frames, axis=0).astype(np.uint8)
    return median_frame


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
    back_sub = cv2.createBackgroundSubtractorMOG2(history=700,
                                                  varThreshold=25, detectShadows=True)

    # Create kernel for morphological operation
    # You can tweak the dimensions of the kernel
    # e.g. instead of 20,20 you can try 30,30.
    kernel = np.ones((20, 20), np.uint8)

    # capture fist frame from video
    _, img = imcap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_prev = img

    # application loop
    while True:

        # capture frame from video
        _, img = imcap.read()

        # converting image from color to grayscale
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.fastNlMeansDenoising(img, None, 20, 7, 21)

        # Use every frame to calculate the foreground mask and update
        # the background
        fg_mask = back_sub.apply(img)

        # Close dark gaps in foreground object using closing
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Remove salt and pepper noise with a median filter
        fg_mask = cv2.medianBlur(fg_mask, 5)

        # Threshold the image to make it either black or white
        _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        # Find the index of the largest contour and draw bounding box
        fg_mask_bb = fg_mask
        contours, hierarchy = cv2.findContours(fg_mask_bb, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        areas = [cv2.contourArea(c) for c in contours]

        # If there are no countours
        if len(areas) < 1:

            # Display the resulting frame
            cv2.imshow(app_name, img)

            # If "q" is pressed on the keyboard,
            # exit this loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Go to the top of the while loop
            continue

        else:
            # Find the largest moving object in the image
            max_index = np.argmax(areas)

        # Draw the bounding box
        cnt = contours[max_index]
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Draw circle in the center of the bounding box
        x2 = x + int(w / 2)
        y2 = y + int(h / 2)
        cv2.circle(img, (x2, y2), 4, (0, 255, 0), -1)

        # Print the centroid coordinates (we'll use the center of the
        # bounding box) on the image
        text = "x: " + str(x2) + ", y: " + str(y2)
        cv2.putText(img, text, (x2 - 10, y2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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

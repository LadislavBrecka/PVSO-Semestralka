import cv2
import numpy as np

# https://debuggercafe.com/moving-object-detection-using-frame-differencing-with-opencv/


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

    # capture background as median of 50 frames
    background_img = get_background(imcap)
    background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow(window_1_name, background_img)

    frame_count = 0
    consecutive_frame = 4
    frame_diff_list = []

    # capture fist frame from video
    _, img = imcap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_prev = img

    # application loop
    while True:

        # capture frame from video
        success, img = imcap.read()
        orig_img = img.copy()

        # converting image from color to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.fastNlMeansDenoising(img, None, 20, 7, 21)

        frame_count += 1
        if frame_count % consecutive_frame == 0 or frame_count == 1:
            frame_diff_list = []

        # find the difference between current frame and base frame
        frame_diff = cv2.absdiff(img, background_img)
        # thresholding to convert the frame to binary
        ret, thres = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
        # dilate the frame a bit to get some more white area...
        # ... makes the detection of contours a bit easier
        dilate_frame = cv2.dilate(thres, None, iterations=2)
        # append the final result into the `frame_diff_list`
        frame_diff_list.append(dilate_frame)

        # if we have reached `consecutive_frame` number of frames
        if len(frame_diff_list) == consecutive_frame:
            # add all the frames in the `frame_diff_list`
            sum_frames = sum(frame_diff_list)

            # find the contours around the white segmented areas
            contours, hierarchy = cv2.findContours(sum_frames, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # draw the contours, not strictly necessary
            for i, cnt in enumerate(contours):
                cv2.drawContours(img, contours, i, (0, 0, 255), 3)

            for contour in contours:
                # continue through the loop if contour area is less than 500...
                # ... helps in removing noise detection
                if cv2.contourArea(contour) < 500:
                    continue
                # get the xmin, ymin, width, and height coordinates from the contours
                (x, y, w, h) = cv2.boundingRect(contour)
                # draw the bounding boxes
                cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow(app_name, orig_img)


        # # TODO: do not know what is doing
        # img = cv2.GaussianBlur(img, (21, 21), 0)
        #
        # # TODO: do not know what is doing
        # se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        # bg = cv2.morphologyEx(img, cv2.MORPH_DILATE, se)
        # img = cv2.divide(img, bg, scale=255)
        # img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
        #
        # # differencing frames
        # diff = img - img_prev
        #
        # # thresholding frames
        # _, img_diff = cv2.threshold(diff, 12, 12, cv2.THRESH_BINARY)
        # count_white = np.count_nonzero(img_diff)
        #
        # # counting white pixels - white == moving
        # if count_white > 150000:
        #     print("MOVE")
        #
        # # display frame(s) (image(s))
        # cv2.imshow(app_name, img_diff)
        #
        # loop will be broken when 'q' is pressed on the keyboard
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        #
        # # save state of current captured frame to state variable for next loop (frame differencing)
        # img_prev = img

    # if loop was terminated, close and erase window
    imcap.release()
    cv2.destroyWindow(app_name)
    cv2.destroyWindow(window_1_name)


# call main function
main()

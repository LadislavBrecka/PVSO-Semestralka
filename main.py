import cv2
import numpy as np


# def low_pass_filtering(image, size):  # Transfer parameters are Fourier transform spectrogram and filter size
#     h, w = image.shape[0:2]  # Getting image properties
#     h1, w1 = int(h / 2), int(w / 2)  # Find the center point of the Fourier spectrum
#     image2 = np.zeros((h, w), np.uint8)  # Define a blank black image with the same size as the Fourier Transform Transfer
#     image2[h1 - int(size / 2):h1 + int(size / 2), w1 - int(size / 2):w1 + int(
#         size / 2)] = 1  # Center point plus or minus half of the filter size, forming a filter size that defines the size, then set to 1, preserving the low frequency part
#     image3 = image2 * image  # A low-pass filter is obtained by multiplying the defined low-pass filter with the incoming Fourier spectrogram one-to-one.
#     return image3
#
#
# def fft_filter(image, filter_const: int):
#     fft_image = np.fft.fft2(image)
#     shifted_fft_image = np.fft.fftshift(fft_image)
#
#     # Low-pass filter
#     shifted_fft_image = low_pass_filtering(shifted_fft_image, filter_const)
#     res = np.log(np.abs(shifted_fft_image))
#
#     # Inverse Fourier Transform
#     idft_shift = np.fft.ifftshift(shifted_fft_image)  # Move the frequency domain from the middle to the upper left corner
#     ifimg = np.fft.ifft2(idft_shift)  # Fourier library function call
#     ifimg = np.abs(ifimg)
#     return np.int8(ifimg)


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


app_name = "PVSO-Semestralka"
window_1_name = "1"

imcap = cv2.VideoCapture(0)
imcap.set(3, 640)  # set width as 640
imcap.set(4, 480)  # set height as 480

# capture fist frame from video
_, img = imcap.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_prev = img

while True:
    success, img = imcap.read()  # capture frame from video

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting image from color to grayscale

    img = cv2.GaussianBlur(img, (21, 21), 0)

    se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    bg = cv2.morphologyEx(img, cv2.MORPH_DILATE, se)
    img = cv2.divide(img, bg, scale=255)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]

    diff = img - img_prev

    _, img_diff = cv2.threshold(diff, 12, 12, cv2.THRESH_BINARY)
    count_white = np.count_nonzero(img_diff)

    if count_white > 150000:
        print("MOVE")

    cv2.imshow(app_name, img_diff)  # loop will be broken when 'q' is pressed on the keyboard
    cv2.imshow(window_1_name, img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    img_prev = img

imcap.release()
cv2.destroyWindow(app_name)
cv2.destroyWindow(window_1_name)

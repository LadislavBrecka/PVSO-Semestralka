import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


def laplacian_of_gaussian(x, y, sigma):
    nominator = ((y ** 2) + (x ** 2) - 2 * (sigma ** 2))
    denominator = (2 * math.pi * (sigma ** 6))
    exponent = math.exp(-((x ** 2) + (y ** 2)) / (2 * (sigma ** 2)))
    return nominator * exponent / denominator


def create_log_mask(sigma, size):
    w = math.ceil(float(size) * float(sigma))
    if w % 2 == 0:
        w = w + 1

    log_mask = []

    w_range = int(math.floor(w / 2))
    print("Going from " + str(-w_range) + " to " + str(w_range))
    for i in range(-w_range, w_range + 1):
        for j in range(-w_range, w_range + 1):
            log_mask.append(laplacian_of_gaussian(i, j, sigma))
    log_mask = np.array(log_mask)
    log_mask = log_mask.reshape(w, w)
    return log_mask


def convolution(image, mask):
    height, width = image.shape
    w_range = int(math.floor(mask.shape[0] / 2))
    res_image = np.zeros((height, width))

    for i in range(w_range, width - w_range):
        for j in range(w_range, height - w_range):
            # Then convolute with the mask
            for k in range(-w_range, w_range + 1):
                for h in range(-w_range, w_range + 1):
                    res_image[j, i] += mask[w_range + h, w_range + k] * image[j + h, i + k]
    return res_image


def zero_crossing(img):
    zero_crossing_image = np.zeros(img.shape)

    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            negative = 0
            positive = 0
            for a in range(-1, 1 + 1):
                for b in range(-1, 1 + 1):
                    if a != 0 and b != 0:
                        if img[i + a, j + b] < 0:
                            negative += 1
                        elif img[i + a, j + b] > 0:
                            positive += 1

            z_c = ((negative > 0) and (positive > 0))
            if z_c:
                zero_crossing_image[i, j] = 1

    return zero_crossing_image


def main():
    file_path = 'images/img3.jpg'

    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img is not None:
        plt.imshow(img, cmap='gray')
        plt.title("Original Image")
        plt.xlabel("Width Pixels")
        plt.ylabel("Height Pixels")
        plt.show()
        # vypocet LoG masky
        sigma = 2
        log_mask = create_log_mask(sigma, 5)
        # vykreslenie masky
        fig = plt.figure(figsize=(6, 3.2))
        ax = fig.add_subplot(111)
        ax.set_title('LoG Mask')
        plt.imshow(log_mask)
        ax.set_aspect('equal')
        cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.patch.set_alpha(0)
        cax.set_frame_on(False)
        plt.colorbar(orientation='vertical')
        plt.show()
        # aplikovanie masky pomocou konvolúcie
        new_img = convolution(img, log_mask)
        plt.imshow(new_img, cmap='gray')
        plt.title("LoG image with sigma = {}".format(sigma))
        plt.xlabel("Width Pixels")
        plt.ylabel("Height Pixels")
        plt.show()
        # zero crossing na detekciu hrán
        new_img1 = zero_crossing(new_img)
        plt.imshow(new_img1, cmap='gray')
        plt.title(" ZC LoG image with sigma = {}".format(sigma))
        plt.xlabel("Width Pixels")
        plt.ylabel("Height Pixels")
        plt.show()
    else:
        print("File not found.")


if __name__ == "__main__":
    main()

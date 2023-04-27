import dip.image as im
import numpy as np
import cv2

def image_preprocessor(image):
    brightness = im.brightness(image)

    # adjust the brightness for images that are too bright
    if brightness > 10:
        image = im.light(image, bright=-10, contrast=0)

    ref_image = im.light(image, bright=20, contrast=0)

    image = im.equalize_light(image, limit=1, grid=(2,2),gray=True)
    #image, alpha, beta = im.automatic_brightness_and_contrast(image,clip_hist_percent=10)

    # black_level = im.back_in_black(image)
    # if not black_level:
    #     image = cv2.bitwise_not(image)

    image = im.gauss_filter(image, (3,3))
    image = im.light(image, bright=0, contrast=40)

    # the bigger the kernel, more fractal the label is
    # kernel = np.ones((5,5), np.uint8)
    # mask = cv2.erode(image, kernel, iterations=1)
    # mask = cv2.dilate(image, kernel, iterations=1)
    # image = np.subtract(image, mask)

    image = im.otsu(image)
    image, contour = im.contour(ref_image, image)
    image = im.removePixelsOutsite(image, contour)

    return image

def label_preprocessor(label):
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    label = im.threshold(label, min_limit=127)
    return label

def posprocessor(image):
    # image = im.threshold(image)
    # kernel = np.ones((5,5), np.uint8)
    # image = cv2.dilate(image, kernel, iterations=1)

    # image = im.median_filter(image, 11)
    # image = im.median_filter(image, 7, iterations=6)
    # image = im.median_filter(image, 5, iterations=6)

    # image = cv2.erode(image, kernel, iterations=1)
    # image = im.median_filter(image, 3, iterations=6)

    return im.threshold(image)

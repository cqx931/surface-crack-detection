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
    image = im.light(image, bright=0, contrast=20)

    # the bigger the kernel, more fractal the label is
    # kernel = np.ones((5,5), np.uint8)
    # mask = cv2.erode(image, kernel, iterations=1)
    # mask = cv2.dilate(image, kernel, iterations=1)
    # image = np.subtract(image, mask)

    # image = im.otsu(image)
    image, contour = im.contour(ref_image, image)
    image = im.removePixelsOutsite(image, contour)
    image = mark_orientation(image, contour)
    return image

def mark_orientation(img, contours):
    img_o = img.copy()
    img_o = cv2.cvtColor(img_o,cv2.COLOR_GRAY2BGR)

    for i, c in enumerate(contours):
      # Calculate the area of each contour
      area = cv2.contourArea(c)
      print(area)
      # Ignore contours that are too small or too large
      if area < 10000:
        continue

      # Draw each contour only for visualisation purposes
      cv2.drawContours(img_o, contours, i, (255,0,0), 1)

      # Find the orientation of each shape
      angle = im.getOrientation(c, img_o)
    return img_o

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

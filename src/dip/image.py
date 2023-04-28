import setting.constant as const
import numpy as np
import cv2
from math import atan2, cos, sin, sqrt, pi

def overlay(image, layer):
    if (len(layer.shape) == 2):
        layer = cv2.cvtColor(layer, cv2.COLOR_GRAY2BGR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    layer = cv2.cvtColor(layer, cv2.COLOR_BGR2BGRA)

    layer[np.where((layer == [0,0,0,255]).all(axis=2))] = const.BACKGROUND_COLOR + [255]
    layer[np.where((layer == [255,255,255,255]).all(axis=2))] = const.SEGMENTATION_COLOR + [255]
    layer = cv2.addWeighted(image, 0.6, layer, 0.4, 0)
    return layer

def light(image, bright, contrast):
    bright = bright * 1.2
    contrast = contrast * 2
    image = image * ((contrast/127)+1) - contrast + bright
    image = np.clip(image, 0, 255)
    return np.uint8(image)

def contour(image, canvas):
    thresh = cv2.threshold(image, 25, 255, cv2.THRESH_BINARY)[1]
    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

    # # get external contours
    cnts = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # draw white contour on black background
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        if perimeter > 200:
            cv2.drawContours(canvas, [c], 0, 255, 1)

    return (canvas, cnts)

def removePixelsOutsite(image, contours):
    big_contour = max(contours, key=cv2.contourArea)

    # draw filled contour on black background
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [big_contour], 0, (255,255,255), -1)
    new_image = cv2.bitwise_and(image, mask)
    return new_image

def brightness(image):
    if len(image.shape) == 3:
        # Colored RGB or BGR (*Do Not* use HSV images with this function)
        # create brightness with euclidean norm
        return np.average(norm(image, axis=2)) / np.sqrt(3)
    else:
        # Grayscale
        return np.average(image)

def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    # Calculate grayscale histogram
    hist = cv2.calcHist([image],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

def threshold(image, min_limit=None, max_limit=255, clip=0):
    if min_limit is None:
        min_limit = int(np.mean(image) + clip)
        # change from binary to otsu?
    _, image = cv2.threshold(image, min_limit, max_limit, cv2.THRESH_BINARY)
    return np.uint8(image)

def threshold_otsu(image, min_limit=None, max_limit=255, clip=0):
    if min_limit is None:
        min_limit = int(np.mean(image) + clip)
        # change from binary to otsu?
    _, image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return np.uint8(image)

def gauss_filter(image, kernel=(3,3), iterations=1):
    for _ in range(iterations):
        image = cv2.GaussianBlur(image, kernel, 0)
    return np.uint8(image)

def median_filter(image, kernel=3, iterations=1):
    for _ in range(iterations):
        image = cv2.medianBlur(image, kernel, 0)
    return np.uint8(image)

def equalize_light(image, limit=3, grid=(7,7), gray=False):
    if (len(image.shape) == 2):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = True

    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))

    image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    if gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return np.uint8(image)

def back_in_black(image):
    image = light(image.copy(), bright=120, contrast=60)
    black_level = 0

    for x in range(6):
        bi = threshold(image, clip=-x)
        if (bi==0).sum() > (bi==255).sum():
            black_level += 1

    return (black_level > 3)


def drawAxis(img, p_, q_, color, scale):
  p = list(p_)
  q = list(q_)

  ## [visualization1]
  angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
  hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

  # Here we lengthen the arrow by a factor of scale
  q[0] = p[0] - scale * hypotenuse * cos(angle)
  q[1] = p[1] - scale * hypotenuse * sin(angle)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

  # create the arrow hooks
  p[0] = q[0] + 9 * cos(angle + pi / 4)
  p[1] = q[1] + 9 * sin(angle + pi / 4)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

  p[0] = q[0] + 9 * cos(angle - pi / 4)
  p[1] = q[1] + 9 * sin(angle - pi / 4)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
  ## [visualization1]

def getOrientation(pts, img):
  ## [pca]
  # Construct a buffer used by the pca analysis
  sz = len(pts)
  data_pts = np.empty((sz, 2), dtype=np.float64)
  for i in range(data_pts.shape[0]):
    data_pts[i,0] = pts[i,0,0]
    data_pts[i,1] = pts[i,0,1]

  # Perform PCA analysis
  mean = np.empty((0))
  mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

  # Store the center of the object
  cntr = (int(mean[0,0]), int(mean[0,1]))
  ## [pca]

  ## [visualization]
  # Draw the principal components
  cv2.circle(img, cntr, 3, (255, 0, 255), 2)
  p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
  p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
  drawAxis(img, cntr, p1, (255, 255, 0), 1)
  drawAxis(img, cntr, p2, (0, 0, 255), 2)

  angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
  ## [visualization]

  # Label with the rotation angle
  label = str(-int(np.rad2deg(angle)) - 90) + " degrees"
  textbox = cv2.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
  cv2.putText(img, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

  return angle

######### Unused #########


def edges(image, threshold1=250, threshold2=350, kernel=3):
    image = cv2.Canny(image, threshold1, threshold2, kernel)
    image = cv2.bitwise_not(image)
    return np.uint8(image)

def equalize_hist(image):
    image = cv2.equalizeHist(image)
    return np.uint8(image)

def otsu(img):
    hist = np.zeros(256, dtype=int)

    for y in range(len(img)):
        for x in range(len(img[0])):
            hist[int(img[y,x])] += 1

    total = (len(img) * len(img[0]))

    current_max, threshold = 0, 0
    sumT, sumF, sumB = 0, 0, 0

    weightB, weightF = 0, 0
    varBetween, meanB, meanF = 0, 0, 0

    for i in range(0,256):
        sumT += (i * hist[i])

    for i in range(0,256):
        weightB += hist[i]
        weightF = total - weightB
        if (weightF <= 0):
            break
        if (weightB <= 0):
            weightB = 1

        sumB += (i * hist[i])
        sumF = sumT - sumB
        meanB = sumB/weightB
        meanF = sumF/weightF
        varBetween = (weightB * weightF)
        varBetween *= (meanB-meanF) * (meanB-meanF)

        if (varBetween > current_max):
            current_max = varBetween
            threshold = i

    img[img <= threshold] = 0
    img[img > threshold] = 255
    return np.array(img, dtype=np.uint8)

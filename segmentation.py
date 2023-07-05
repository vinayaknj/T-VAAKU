import cv2
import sys
import numpy as np
from scipy import ndimage


def openImage(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def showImage(img1):
    cv2.imshow('image', img1)
    cv2.waitKey()
    cv2.destroyAllWindows()


def find_connected(img, label, id, i, j, threshold, xmin, ymin, xmax, ymax):
    h, w = img.shape
    stack = [(i, j)]

    while stack:
        i, j = stack.pop()

        if img[i, j] < threshold or i >= h - 1 or j >= w - 1 or label[i, j] != 0:
            continue

        if j < xmin:
            xmin = j
        if j > xmax:
            xmax = j
        if i < ymin:
            ymin = i
        if i > ymax:
            ymax = i

        label[i, j] = id

        if i + 1 < h:
            stack.append((i + 1, j))
        if i + 1 < h and j + 1 < w:
            stack.append((i + 1, j + 1))
        if j + 1 < w:
            stack.append((i, j + 1))
        if i - 1 >= 0 and j + 1 < w:
            stack.append((i - 1, j + 1))
        if i - 1 >= 0:
            stack.append((i - 1, j))
        if i - 1 >= 0 and j - 1 >= 0:
            stack.append((i - 1, j - 1))
        if j - 1 >= 0:
            stack.append((i, j - 1))
        if i + 1 < h and j - 1 >= 0:
            stack.append((i + 1, j - 1))

    return label, xmin, ymin, xmax, ymax


def _main(img):
    try:
        height, width = img.shape
    except:
        print('exception occurred in segmentation.py')
        return None

    threshold = 127
    id = 0
    label = np.zeros((height, width))
    result = []

    for j in range(width):
        for i in range(height):
            if img[i, j] >= threshold and label[i, j] == 0:
                id += 1
                label, xmin, ymin, xmax, ymax = find_connected(img, label, id, i, j, threshold, width, height, -1, -1)
                crop_img = img[ymin:ymax, xmin:xmax]
                result.append(crop_img)

                img[ymin - 2:ymin - 1, xmin - 1:xmax + 1] = 100
                img[ymax + 1:ymax + 2, xmin - 1:xmax + 1] = 100
                img[ymin - 1:ymax + 1, xmin - 2:xmin - 1] = 100
                img[ymin - 1:ymax + 1, xmax + 1:xmax + 2] = 100

    return result


def main():
    sys.setrecursionlimit(2500)
    img_path = input('Enter the image file name: ')
    img = openImage(img_path)
    print('--> Input Image')
    showImage(img)

    ret, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY_INV)
    print('--> Inverse Binary Image')
    showImage(img)

    img = ndimage.median_filter(img, 5)
    print('--> Applying Median Filter')
    showImage(img)

    height, width = img.shape
    threshold = 127
    id = 0
    label = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            if img[i, j] >= threshold and label[i, j] == 0:
                id += 1
                label, xmin, ymin, xmax, ymax = find_connected(img, label, id, i, j, threshold, width, height, -1, -1)
                crop_img = img[ymin:ymax, xmin:xmax]
                showImage(crop_img)

                img[ymin - 2:ymin - 1, xmin - 1:xmax + 1] = 100
                img[ymax + 1:ymax + 2, xmin - 1:xmax + 1] = 100
                img[ymin - 1:ymax + 1, xmin - 2:xmin - 1] = 100
                img[ymin - 1:ymax + 1, xmax + 1:xmax + 2] = 100

    print('--> final image: ')
    showImage(img)


def boundingBoxes(img):
    sys.setrecursionlimit(9999)
    x0 = []
    y0 = []
    x1 = []
    y1 = []

    ret, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY_INV)

    try:
        height, width = img.shape
    except:
        print('exception occurred in segmentation.py')
        return None

    threshold = 127
    id = 0
    label = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            if img[i, j] >= threshold and label[i, j] == 0:
                id += 1
                label, xmin, ymin, xmax, ymax = find_connected(img, label, id, i, j, threshold, width, height, -1, -1)
                x1.append(xmax)
                x0.append(xmin)
                y1.append(ymax)
                y0.append(ymin)

    return x0, y0, x1, y1


# main()  # Commented out to avoid running it during code execution

import numpy as np
import cv2
import math


# This function gets upper and lower bounds of line written


def getBounds(img):
    img = np.array(img)
    original = np.copy(img)
    ## get histogram of horizontal projection
    [r, c] = img.shape

    horizontalProjection = np.sum(img, axis=1)
    gradient1 = np.gradient(horizontalProjection, edge_order=1)
    peak = np.argmax(horizontalProjection)
    p = horizontalProjection[peak]

    # get lb
    lb = peak + 1
    while lb < r and horizontalProjection[lb] > p / 2:
        lb = lb + 1

    if lb != r:
        img[lb, :] = np.ones([1, c])

    ##get ub
    ub = peak - 1
    while ub >= 0 and horizontalProjection[ub] > p / 2:
        ub = ub - 1
    if ub != 0:
        img[ub, :] = np.ones([1, c])

    return ub, lb


# This function sets features vectors for a specific form image using sliding window technique
def getFeaturesVectors(extractedLines):
    featuresCount = 13  # chafeaturesCountnge here
    imageFeaturesVectors = np.empty([0, featuresCount])
    for index, img in enumerate(extractedLines):

        if len(img[img == 255]) > 10:
            lineFeaturesVector = []
            indices = np.where(img == [255])
            topContour = indices[1][0]
            bottomContour = indices[1][-1]
            ub, lb = getBounds(img)
            f1 = math.fabs(topContour - ub)
            f2 = math.fabs(ub - lb)
            f3 = math.fabs(lb - bottomContour)
            f4 = f1 / f2
            avgDist = interwordDistance(img)
            lineFeaturesVector.append(f1)
            lineFeaturesVector.append(f2)
            lineFeaturesVector.append(f3)
            lineFeaturesVector.append(avgDist)
            histogram = computeSlantHistogram(img)
            sumHistogram = np.sum(histogram)
            if sumHistogram != 0:
                histogram = histogram / sumHistogram
            lineFeaturesVector = np.reshape(
                lineFeaturesVector, (1, featuresCount-9))  # change
            histogram = np.reshape(histogram, (1, 9))
            allFeatures = np.hstack((lineFeaturesVector, histogram))
            imageFeaturesVectors = np.vstack(
                (imageFeaturesVectors, allFeatures))  # change
        else:
            continue

    if imageFeaturesVectors.shape[0] > 0:
        return imageFeaturesVectors
    else:
        print("Failed to extract features , shape ", imageFeaturesVectors.shape)
        return None


# This function computes the interword distance feature
'''
It calculates the average word distance between words in a line
INPUT: Binary segmented lines
'''


def interwordDistance(thresh):

    scale_percent = 25
    width = int(thresh.shape[1] * scale_percent / 100)
    height = int(thresh.shape[0] * scale_percent / 100)
    dim = (width, height)
    resizedLine = cv2.resize(thresh, dim, interpolation=cv2.INTER_AREA)/255
    verticalSum = np.sum(resizedLine, axis=0)
    zeroIndicesX = np.where(verticalSum == 0)
    diffIndices = np.diff(zeroIndicesX)
    diffIndices[diffIndices < 20] = 0
    avgInterwordDistance = np.average(diffIndices)
    #print("AVG DISTANCE ",avgInterwordDistance)
    return avgInterwordDistance


'''
    This function calculates the slant of a window in eight different directions
    It computes it by &ing different pixels in the upper half of the window
    Each pair of pixels represents different direction
'''
# This function is used to get the slant feature in eight directions


def eightDirections(window):
    x = 2
    y = 2
    windowHist = np.zeros((1, 9))
    if window[x, y] == 0:
        return windowHist
    windowHist[0][0] = window[x + 1, y] & window[x + 2, y]
    windowHist[0][1] = window[x + 1, y - 1] & window[x + 2, y - 1]
    windowHist[0][2] = window[x + 1, y - 1] & window[x + 2, y - 2]
    windowHist[0][3] = window[x, y - 1] & window[x + 1, y - 2]
    windowHist[0][4] = window[x, y - 1] & window[x, y - 2]
    windowHist[0][5] = window[x, y - 1] & window[x - 1, y - 2]
    windowHist[0][6] = window[x - 1, y - 1] & window[x - 2, y - 2]
    windowHist[0][7] = window[x - 1, y - 1] & window[x - 2, y - 1]
    windowHist[0][8] = window[x - 1, y] & window[x - 2, y]
    return windowHist


def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value


# This function is used to compute the slant histogram
'''
    This function computes the slant of a single line
    It uses the sliding windows concept
    It applies the eight directions function to each window
    And accumulate the result
'''


def computeSlantHistogram(line):
    line = np.array(line)
    #print(line)
    scale_percent = 25
    width = int(line.shape[1] * scale_percent / 100)
    height = int(line.shape[0] * scale_percent / 100)
    dim = (width, height)
    lineResize = cv2.resize(line, dim, interpolation=cv2.INTER_AREA)
    histogram = np.zeros((1, 9))
    lbpHist = np.zeros(256)
    h, w = lineResize.shape
    lineResize[lineResize == 255] = 1
    for i in range(2, h - 2):
        for j in range(2, w - 2):
            window = lineResize[i - 2:i + 3, j - 2:j + 3]
            if not np.all((window == 0)):
                windowHistogram = eightDirections(window)
                histogram = histogram + windowHistogram
    return histogram



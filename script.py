# import packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gaussianKern(w, s):
    x = np.arange(0, w, 1, float)
    y = x[:, np.newaxis]
    x = x - w/2
    y = y - w/2
    gausKern = np.exp(-((x * x + y * y)/ (2. * s * s)))
    return gausKern / np.sum(gausKern)

#function to turn into grayscale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# function for image thresholding with values under 0 and over 0
def imThreshold0(img, threshold, maxValT):
    assert len(img.shape) == 2 # input image has to be gray
    
    height, width = img.shape
    bi_img = np.zeros((height, width), dtype=np.float)
    for x in range(1, height - 1):
        for y in range(1, width - 1):
            if img.item(x, y) > 0:
                minVal = 0;
                for z in range (0, 3):
                    for a in range (0, 3):
                        if (img.item(x-z + 1, y-a + 1) < minVal):
                            minVal = img.item(x-z + 1, y-a + 1)
                if (np.abs(minVal - img.item(x, y)) > threshold):
                    bi_img.itemset((x, y), maxValT)
            if img.item(x, y) > 0:
                maxVal = 0;
                for z in range (0, 3):
                    for a in range (0, 3):
                        if (img.item(x-z + 1, y-a + 1) < maxVal):
                            maxVal = img.item(x-z + 1, y-a + 1)
                if (np.abs(maxVal - img.item(x, y)) > threshold):
                    bi_img.itemset((x, y), maxValT)
    return bi_img
            
def imThreshold(img, threshold, maxVal):
    assert len(img.shape) == 2 # input image has to be gray
    
    height, width = img.shape
    bi_img = np.zeros((height, width), dtype=np.float)
    for x in xrange(height):
        for y in xrange(width):
            if img.item(x, y) > threshold:
                bi_img.itemset((x, y), maxVal)
                
    return bi_img

def laplacian(img, kern):
    img = img.astype(float)
    data = rgb2gray(img)
    lapLImage = data.copy()
    (w, h) = img.shape[:2]
    for y in range (5, w-5):
        for x in range (5, h-5):
            result = 0
            for z in range (0, 11):
                for a in range (0, 11):
                    result = result + (kern[10 - z, 10 - a] * data[(y + 5 - z), x + 5 - a])
            lapLImage[y, x] = result
    return imThreshold0(lapLImage, 800, 255)

def canny_enhancer_nonmax_sup(img, kern):
    data = rgb2gray(img)
    (w, h) = img.shape[:2]
    cannyImage = np.zeros((w, h), dtype=np.float)
    for y in range (5, w-5):
        for x in range (5, h-5):
            result = 0
            for z in range (0, 11):
                for a in range (0, 11):
                    result = result + (kern[10 - z, 10 - a] * data[(y + 5 - z), x + 5 - a])
            cannyImage[y, x] = result
    data = cannyImage
    cannyImage = np.zeros((w, h), dtype=np.float)
    cannyDir = np.zeros((w, h), dtype=np.float)
    for y in range (1, w-1):
        for x in range (1, h-1):
            sumx = data[(y - 1), x - 1] - data[(y - 1), x + 1] + (2 * data[y, x - 1]) - (2 * data[y, x + 1]) + data[(y + 1), x - 1] - data[(y + 1), + x + 1]
            sumy = data[(y - 1), x - 1] + (2 * data[(y - 1), x]) + data[(y - 1), x + 1] - data[(y + 1), x - 1] - (2 * data[(y + 1), x]) - data[(y + 1), x + 1]
            result = np.sqrt(sumx * sumx + sumy * sumy)
            cannyImage[y, x] = result
            if not(sumx == 0):
                cannyDir[y, x] = np.arctan(sumy/sumx)
    for y in range (1, w-1):
        for x in range (1, h-1):
            curPix = cannyImage[y, x]
            if not(((curPix > cannyImage[y - 1, x]) and (curPix < cannyImage[y + 1, x])) or ((curPix > cannyImage[y, x-1] and curPix < cannyImage[y, x+1]))):    
                cannyImage[y, x] = 0
    return imThreshold(cannyImage, 75, 255)

img = cv2.imread('jon.bmp')

matrix = gaussianKern(11, 1)
matrixcon = gaussianKern(11, 1)

for y in range (1, 10):
    for x in range (1, 10):
        value = matrix[(y - 1), x] + matrix[(y), x + 1] + (matrix[y, x - 1]) + (4 * matrix[y, x]) + matrix[(y + 1), x]
        matrixcon[y, x] = value

lapImg = laplacian(img, matrixcon)
cannyImg = canny_enhancer_nonmax_sup(img, matrix)
plt.subplot(4, 5, 1)
plt.title('Zero Crossing')
plt.imshow(lapImg, cmap = plt.get_cmap('gray'))
plt.axis('off')

plt.subplot(4, 5, 2)
plt.title('Canny')
plt.imshow(cannyImg, cmap = plt.get_cmap('gray'))
plt.axis('off')

plt.show()
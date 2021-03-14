import cv2
import numpy as np
import json
import os
import time
from matplotlib import pyplot as plt

# des = cv2.bitwise_not(meansImage)
# contour,hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

# for cnt in contour:
#     cv2.drawContours(des,[cnt],0,255,-1)

#h, w = closing.shape[:2]
#mask = np.zeros((h+2, w+2), np.uint8)
#cv2.floodFill(closing, mask, (0,0), 255)
#closing = cv2.bitwise_not(closing)

def prepareEdgeImage(edgeImage):
    """
    Applies morphological transformations to binary (edge) image.
    """
    kernel = np.ones((2,2),np.uint8)
    dilated_img = cv2.dilate(edgeImage, kernel, iterations = 2)

    smallComponentsImage = removeLargeConnectedComponents(dilated_img, 300)
    #cv2.imshow("Small Connected Component Image", smallComponentsImage)

    kernel = np.ones((10,10),np.uint8)
    closing = cv2.morphologyEx(smallComponentsImage, cv2.MORPH_CLOSE, kernel)
    return closing

def blobDetection(image):
    """
    Applies blob detection to prepared image. Returns (keypoints, image with keypoints)
    """
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 200
    params.filterByArea = True
    params.maxArea = 500
    params.minArea = 10
    params.filterByColor = True
    params.blobColor = 255
    params.filterByCircularity = False
    params.minCircularity = 0.1
    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return (keypoints, im_with_keypoints)

def getTargetImages(targetCoordinates, image, bigImage):
    croppedImages = []
    padding = 5
    for coord in targetCoordinates:
        x,y= coord.pt
        radius = (coord.size//2) + padding
        maxX = int((x + radius)/image.shape[1] * bigImage.shape[1])
        minX = int((x - radius)/image.shape[1] * bigImage.shape[1])
        maxY = int((y + radius)/image.shape[0] * bigImage.shape[0])
        minY = int((y - radius)/image.shape[0] * bigImage.shape[0])
        crop = bigImage[minY:maxY, minX:maxX]
        croppedImages.append(crop)
    return croppedImages

def removeLargeConnectedComponents(image, max_size):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] <= max_size:
            img2[output == i + 1] = 255
    return img2.astype(np.uint8)

def findTargets(origionalImage):
    startTime = time.time()
    image = cv2.resize(origionalImage, (800, 600))
    #cv2.imshow("Origional", image)
    meansImage = cv2.pyrMeanShiftFiltering(image, 30, 70)
    edgeImage = cv2.Canny(meansImage,100,200)
    closingImage = prepareEdgeImage(edgeImage)
    targetCoordinates, blobImage = blobDetection(closingImage)
    im_with_keypoints = cv2.drawKeypoints(image, targetCoordinates, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print ("Detection Procesing Time:", str(time.time() - startTime))
    targets = getTargetImages(targetCoordinates, image, origionalImage)
    return (im_with_keypoints, targets)

imageDir = "2.jpg"
image = cv2.imread(imageDir)
finalImage, targets = findTargets(image)
cv2.imshow("Final", finalImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

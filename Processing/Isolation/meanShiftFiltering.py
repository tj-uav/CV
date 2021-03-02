import cv2
import numpy as np
import json
import os
import time

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
    kernel = np.ones((3,3),np.uint8)
    dilated_img = cv2.dilate(edgeImage, kernel, iterations = 2)
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(dilated_img, cv2.MORPH_CLOSE, kernel)
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

def getTargetImages(targetCoordinates, closingImage):
    return

def removeLargeConnectedComponents(image):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 150

    #your answer image
    img2 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] <= min_size:
            img2[output == i + 1] = 255
    return img2.astype(np.uint8)

def findTargets(image):
    startTime = time.time()
    image = cv2.resize(image, (800, 600))
    meansImage = cv2.pyrMeanShiftFiltering(image, 30, 70)
    edgeImage = cv2.Canny(meansImage,100,200)
    smallComponentImage = removeLargeConnectedComponents(edgeImage)
    cv2.imshow("Test", smallComponentImage)
    closingImage = prepareEdgeImage(smallComponentImage)
    targetCoordinates, blobImage = blobDetection(closingImage)
    print ("Detection Procesing Time:", str(time.time() - startTime))
    return (blobImage)

imageDir = "sampleTest.jpg"
image = cv2.imread(imageDir)
finalImage = findTargets(image)
cv2.imshow("Final", finalImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

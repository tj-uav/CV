import cv2
import numpy as np
import json
import os
import time

startTime = time.time()
imageDir = "3.jpg"
image = cv2.imread(imageDir)
image = cv2.resize(image, (800, 600))
meansImage = cv2.pyrMeanShiftFiltering(image, 30, 70)

# des = cv2.bitwise_not(meansImage)
# contour,hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

# for cnt in contour:
#     cv2.drawContours(des,[cnt],0,255,-1)

edgeImage = cv2.Canny(meansImage,100,200)

kernel = np.ones((3,3),np.uint8)
dilated_img = cv2.dilate(edgeImage, kernel, iterations = 2)
kernel = np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(dilated_img, cv2.MORPH_CLOSE, kernel)

#h, w = closing.shape[:2]
#mask = np.zeros((h+2, w+2), np.uint8)
#cv2.floodFill(closing, mask, (0,0), 255)
#closing = cv2.bitwise_not(closing)

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
keypoints = detector.detect(closing)
im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("lol", im_with_keypoints)

cv2.imshow("Closing", closing)
print (time.time() - startTime)
#cv2.imshow("yuh", edgeImage)
#cv2.imshow("yuhyuh", meansImage)
#cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
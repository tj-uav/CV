import cv2
import numpy as np
import json
import os
import time


startTime = time.time()
imageDir = "1.jpg"
image = cv2.imread(imageDir)
image = cv2.resize(image, (800, 600))
meansImage = cv2.pyrMeanShiftFiltering(image, 30, 70)
edgeImage = cv2.Canny(meansImage,100,200)

kernel = np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(edgeImage, cv2.MORPH_CLOSE, kernel)

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 10
params.maxThreshold = 200
params.filterByColor = True
params.blobColor = 255
params.filterByArea = False

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(closing)
im_with_keypoints = cv2.drawKeypoints(edgeImage, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("lol", im_with_keypoints)

cv2.imshow("Closing", closing)
print (time.time() - startTime)
cv2.imshow("yuh", edgeImage)
cv2.imshow("yuhyuh", meansImage)
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np
import json
import os
import time


startTime = time.time()
imageDir = "pic588.jpg"
image = cv2.imread(imageDir)
meansImage = cv2.pyrMeanShiftFiltering(image, 20, 70)
edgeImage = cv2.Canny(meansImage,100,200)
edgeImage = cv2.resize(edgeImage, (500, 1000))
meansImage = cv2.resize(meansImage, (500, 1000))
print (time.time() - startTime)
cv2.imshow("yuh", edgeImage)
cv2.imshow("yuhyuh", meansImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
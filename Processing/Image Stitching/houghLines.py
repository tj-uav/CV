import cv2
import os, signal
import numpy as np
import math

test = cv2.imread(os.getcwd() + "/Processing/Image Stitching/Images/test1.jpg")
test = cv2.resize(test, (900, 600))

filtered = cv2.bilateralFilter(test, 10, 100, 100)
cv2.imshow("test2", filtered)
gray = cv2.cvtColor(filtered,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,150,200,apertureSize = 3)
cv2.imshow("test", edges)
minLineLength = 100
maxLineGap = 10
#lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)

#lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=100)
lines = cv2.HoughLines(edges,1,np.pi/180,200)
if (len(lines)!=0):
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(test,pt1,pt2,(0,0,255),2)

# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(test, (x1, y1), (x2, y2), (255, 0, 0), 3)

cv2.imshow("lines", test)
cv2.waitKey(0)
cv2.destroyAllWindows()
# use opencv to read and then apply sobel operator to the image
import cv2
import numpy as np
import time
from os import path, getcwd

# from tqdm import tqdm
name = "476"
img = cv2.imread(r"C:\Users\taof\Downloads\476.png")
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_grey = cv2.GaussianBlur(img_grey, (7, 7), 0)
# cv2.imshow("grey", img)
sobelx = cv2.Sobel(img_grey, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img_grey, cv2.CV_64F, 0, 1, ksize=3)

thresh = 60
sobelx[abs(sobelx) > thresh] = 255
sobely[abs(sobely) > thresh] = 255
sobelx[abs(sobelx) < thresh] = 0
sobely[abs(sobely) < thresh] = 0
# show the overlap
sobel = cv2.bitwise_or(sobelx, sobely)

_, sobel_u8 = cv2.threshold(sobel.astype(np.uint8), 0, 255, cv2.THRESH_BINARY_INV)

contours = cv2.findContours(sobel_u8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
ct = 0
for contour in contours:
    size = cv2.contourArea(contour)
    if 100 < size < 1000:
        x, y, width, height = cv2.boundingRect(contour)
        # r = cv2.boundingRect(contour)
        # print(r)
        roi = img[y - 5 : y + height + 5, x - 5: x + width + 5]
        cv2.imwrite(rf"C:\Users\taof\Downloads\{name}_{ct}.png", roi)
        ct += 1
print(f"found {ct} targets")


# cv2.erode(sobel, np.ones((2, 2), np.uint8), sobel, iterations=1)
# resized = cv2.resize(sobel, (1280, 720))
# cv2.imshow("sobel", sobel)
# cv2.waitKey(0)
# input()
# cv2.destroyAllWindows()
cv2.imwrite("sobel.png", sobel_u8)
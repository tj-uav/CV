import cv2
import numpy as np
import imutils
import time

height = 40
width = 40
def make_training_data():
    for i in range(0,36):
        count = 41
        img = cv2.imread("images/"+str(i)+".png")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)
        for angle in range(-20,21):
            rotated = imutils.rotate_bound(thresh,angle)
            rotated = cv2.resize(rotated,(height,width))
            cnts = cv2.findContours(rotated,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            cnt = max(cnts,key = lambda c: cv2.contourArea(c))
            x, y, w, h = cv2.boundingRect(cnt)
            l = x
            r = x + w
            t = y
            b = y + h
            for i in range(-10,6,1):
                for j in range(-10,6,1):
                    border = cv2.copyMakeBorder(rotated, i, i, j, j borderType=cv2.BORDER_CONSTANT, value=[0,0,0])

            count += 1

make_training_data()
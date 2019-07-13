import random
import cv2
import numpy as np

rand = np.random.randint(2)
SHAPE_OPTIONS = ["circle", "semicircle", "quartercircle", "triangle", "square", "rectangle", "trapezoid", "pentagon", "hexagon", "heptagon", "octagon", "star", "cross"]
#COLOR_OPTIONS = ["Black", "Gray", "White", "Red", "Blue", "Green", "Brown", "Orange", "Yellow", "Purple"]
ALPHA_OPTIONS = [chr(i) for i in range(48,58)]
ALPHA_OPTIONS.extend([chr(i) for i in range(65,65+26)])

def createRandomTarget(alpha, shape):
    alpha_img = cv2.imread('Alphas/' + alpha + '.png')
    alpha_gray = cv2.cvtColor(alpha_img, cv2.COLOR_BGR2GRAY)
    ret, alpha_thresh = cv2.threshold(alpha_gray,127,255,cv2.THRESH_BINARY)
    shape_img = cv2.imread('Shapes/' + shape + '.png')
    shape_gray = cv2.cvtColor(shape_img, cv2.COLOR_BGR2GRAY)
    ret, shape_thresh = cv2.threshold(shape_gray,127,255,cv2.THRESH_BINARY)
    print(type(alpha_thresh))
    print(alpha_thresh.dtype)
    cnts, hierarchy = cv2.findContours(alpha_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key = lambda c: cv2.contourArea(c))
    x,y,w,h = cv2.boundingRect(cnt)
    alpha_img = cv2.rectangle(alpha_img,(x,y),(x+w,y+h),(0,255,0),2)
    cnts, hierarchy = cv2.findContours(shape_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key = lambda c: cv2.contourArea(c))
    x,y,w,h = cv2.boundingRect(cnt)
    shape_img = cv2.rectangle(shape_img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Shape", shape_img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    cv2.imshow("Alpha", alpha_img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

alpha = random.sample(ALPHA_OPTIONS,1)[0]
shape = random.sample(SHAPE_OPTIONS,1)[0]
createRandomTarget(alpha, shape)
"""
from shutil import copyfile
for letter in ALPHA_OPTIONS:
    src = '../Data/fontCharData/char_data' + letter + '/PTC75F.PNG'
    dst = 'Alphas/' + letter + '.png'
    copyfile(src, dst)
"""
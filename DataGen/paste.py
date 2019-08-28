import cv2
import numpy as np
from random import randint
import itertools

SHAPE_COLOR_DEFAULT = [0,255,0]
ALPHA_COLOR_DEFAULT = [255,255,255]
def colorify(img, shape, alpha):
    ori = img.copy()
    mask = np.all(ori == SHAPE_COLOR_DEFAULT, axis=-1)
    img[mask] = shape
    mask = np.all(ori == ALPHA_COLOR_DEFAULT, axis=-1)
    img[mask] = alpha
    return img

def brightness(image, alpha, beta):
    #Alpha represents contrast control
    #Beta represents brightness control
    new_image = np.zeros(image.shape, image.dtype)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
    return new_image

def paste(img, background, size, location = None):
    if location is None:
        #Get random location on background
        location = (randint(0, 100), randint(0, 100))

    w, h, _ = img.shape
    shape = [background.shape[0] * size / 100, background.shape[1] * size / 100]
    if shape[0] < shape[1]:
        shape[1] = shape[0] * h / w
    else:
        shape[0] = shape[1] * w / h
    img = cv2.resize(img, (int(shape[0]), int(shape[1])))
    location = (int((background.shape[0] - img.shape[0]) / 100) * location[0], int((background.shape[1] - img.shape[1]) / 100) * location[1])
    new = background.copy()
    mask = np.any(img != [0, 0, 0], axis=-1)
    for a,b in itertools.product(list(range(mask.shape[0])), list(range(mask.shape[1]))):
        if mask[a,b]:
            new[location[0] + a, location[1] + b] = img[a,b]
#    new[location[0]:location[0]+img.shape[0], location[1]:location[1]+img.shape[1]] = img
    return new

#Location based on scale: 0 = leftmost, 100 = rightmost
#Img size is based on scale where size of 100 takes up as much of the img as possible
#Changing the contrast / brightness is used to make the image look darker (as if its cloudy)

COLOR_OPTIONS = []
alpha = .6
beta = -30
shape_color = [0,0,255]
alpha_color = [255,0,0]
size = 7
location = (50,50)

img = cv2.imread("AlphaShapeData/circle_H.png")
background = cv2.imread("Grounds/Green Grass 5.jpg")

img = colorify(img, shape_color, alpha_color)
img = brightness(img, alpha, beta)
pasted = paste(img, background, size, location)
cv2.imshow("Pasted", pasted)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
from pprint import pprint

SHAPE_COLOR_DEFAULT = [0,255,0]
ALPHA_COLOR_DEFAULT = [255,255,255]
shape = [255,0,0]
alpha = [0,0,255]
img = cv2.imread("AlphaShapeData/circle_H.png")
ori = img.copy()
mask = np.all(ori == SHAPE_COLOR_DEFAULT, axis=-1)
img[mask] = shape
mask = np.all(ori == ALPHA_COLOR_DEFAULT, axis=-1)
img[mask] = alpha
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

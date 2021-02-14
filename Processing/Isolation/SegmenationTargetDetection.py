import cv2
import numpy as np
import json
from detection_methods import *
import os

def calculateHeuristic(image):
    height, width, _ = image.shape
    edgeImage = cv2.Canny(image,100,200)
    numNonWhite = cv2.countNonZero(edgeImage)
    percentNonWhite = numNonWhite/(height*width)
    return (percentNonWhite)

imageDir = "yuh.jpg"
image = cv2.imread(imageDir)
imageFolderDir = os.getcwd()+"\\Processing\\Isolation\\Images\\"

grid = splitImage(image, 10, 10)
heuristicArray = [(calculateHeuristic(image), image) for image in grid]
finalHeuristicArray = sorted(heuristicArray, key=lambda tup: tup[0])[::-1][:10]

for block in finalHeuristicArray:
    cv2.imwrite(imageFolderDir+str(block[0]*100)+".jpg", block[1])

cv2.waitKey(0)
cv2.destroyAllWindows()
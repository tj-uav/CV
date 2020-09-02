#To run code you must have a directory at Processing/Image Stitching/Images/ containing images

#Imports
import cv2
import numpy as np
import os

#Reading image files
images = []
imagesDirectory = 'Processing/Image Stitching/Images/'
dim = (1024,768)
for fileName in os.listdir(imagesDirectory):
    fileDir = os.path.join(imagesDirectory, fileName)
    image=cv2.imread(fileDir,cv2.IMREAD_COLOR)
    image=cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
    images.append(image)

#Settings up image stiticer
stitcher = cv2.Stitcher.create()
ret, pano = stitcher.stitch(images)

#Display stitched image
if ret ==cv2.STITCHER_OK:
    cv2.imshow('Panorama', pano)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error during stitching")
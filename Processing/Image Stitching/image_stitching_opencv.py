import cv2
import numpy as np

dim = (1024,768)
left=cv2.imread('Processing/Image Stitching/Left.jpg',cv2.IMREAD_COLOR)
left = cv2.resize(left,dim,interpolation=cv2.INTER_AREA)
right=cv2.imread('Processing/Image Stitching/Right.jpg',cv2.IMREAD_COLOR)
right=cv2.resize(right,dim,interpolation=cv2.INTER_AREA)

images =[]
images.append(left)
images.append(right)

stitcher = cv2.Stitcher.create()
ret, pano = stitcher.stitch(images)

if ret ==cv2.STITCHER_OK:
    cv2.imshow('Panorama', pano)
    cv2.waitKey()
    cv2.destroyAllWindows()
else:
    print("Error during stitching")


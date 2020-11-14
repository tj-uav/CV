#To run code you must have a directory at Processing/Image Stitching/Images/ containing images

#Imports
import cv2
import numpy as np
import os, signal

#Reading image files
images = []
#imagesDirectory = os.getcwd() + '/Processing/Image Stitching/Images/'
imagesDirectory = os.getcwd() + '/Images/'
# dim = (1024, 768)
for fileName in os.listdir(imagesDirectory):
    fileDir = os.path.join(imagesDirectory, fileName)
    image=cv2.imread(fileDir,cv2.IMREAD_COLOR)
    # image=cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
    images.append(image)

print("Image sizes:", images[0].shape, images[1].shape)
#Settings up image stiticer
print("Running image stitcher")
stitcher = cv2.Stitcher.create()
ret, pano = stitcher.stitch(images)

print("Done stitching, saving to file")
#Display stitched image
if ret == cv2.STITCHER_OK:
    cv2.imwrite("panorama.png", pano)
else:
    print("Error during stitching:", ret)
    if ret == cv2.STITCHER_ERR_NEED_MORE_IMGS:
        print("Need more images")
    if ret == cv2.STITCHER_ERR_HOMOGRAPHY_EST_FAIL:
        print("Homography est failed")
    if ret == cv2.STITCHER_ERR_CAMERA_PARAMS_ADJUST_FAIL:
        print("Camera params adjust failed")

os.kill (os.getpid(), signal.SIGTERM)
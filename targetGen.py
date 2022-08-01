import cv2 as cv2
import numpy as np
import os
from PIL import Image, ImageDraw
from sklearn.cluster import MiniBatchKMeans
import random
import imutils

def blurImage(img):
    kernel_size = 7
    kernel_v = np.zeros((kernel_size, kernel_size))
    kernel_h = np.copy(kernel_v)
    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    
    kernel_v /= kernel_size
    kernel_h /= kernel_size
    
    vertical_mb = cv2.filter2D(img, -1, kernel_v)
    horizonal_mb = cv2.filter2D(vertical_mb, -1, kernel_h)
    return horizonal_mb

def getTarget(shap, shapColor, shapeSize, shapRotation, lett, lettColor, lettSize, lettRotation):
    lett = imutils.rotate_bound( lett, lettRotation)  #Rotate images without accidental crop
    shap = imutils.rotate_bound( shap, shapRotation)

    lettsize = lett.shape
    shapsize = shap.shape

    border = abs((lettSize - shapeSize))//2

    lett = cv2.resize( lett, ( lettSize, lettSize ) )           #Resizes as final size
    shap = cv2.resize( shap, ( shapeSize, shapeSize ) )
    lettmask = cv2.inRange( lett, np.array( [ 1, 1, 1 ] ), np.array( [ 255, 255, 255 ] ) )  #Cleans images, converts to mask
    shapmask = cv2.inRange( shap, np.array( [ 1, 1, 1 ] ), np.array( [ 255, 255, 255 ] ) )

    lett = cv2.cvtColor( lettmask, cv2.COLOR_GRAY2BGR ) #Reconvert clean mask to letter
    shap = cv2.cvtColor( shapmask, cv2.COLOR_GRAY2BGR )
    
    lett[ np.where( ( lett == [ 255, 255, 255 ] ).all( axis = 2 ) ) ] = lettColor #Colorize
    shap[ np.where( ( shap == [ 255, 255, 255 ] ).all( axis = 2 ) ) ] = shapColor
    
    lett = cv2.copyMakeBorder( lett, top = border, bottom = border, left = border, right = border, borderType = cv2.BORDER_CONSTANT, value= [ 0, 0, 0 ] )   #Rescale letter for copy
    lettmask = cv2.copyMakeBorder( lettmask, top = border, bottom = border, left = border, right = border, borderType = cv2.BORDER_CONSTANT, value= [ 0 ] )

    lettmask_inv = cv2.bitwise_not( lettmask )                  #Clear the area
    shap = cv2.bitwise_and( shap, shap, mask = lettmask_inv )
    shap = cv2.add( shap, lett )                              #Add colors
    return shap

count = 0
for ix in os.listdir("Shapes/"):
    for jx in os.listdir("Alphas/"):
        letter = cv2.imread("Alphas/" + jx)
        shape = cv2.imread("Shapes/" + ix)
        sizeL = random.randint(30, 40)
        sizeS = random.randint(55, 65)
        if sizeS % 2 != 0: sizeS += 1
        if sizeL % 2 != 0: sizeL += 1

        letterRot = random.randint(0, 360)

        target = getTarget(shape, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), sizeS, random.randint(0, 360), letter, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), sizeL, letterRot)
        #target = blurImage(target)
        cv2.imshow("h", target)
        print(letterRot)
        cv2.waitKey(0)
        cv2.imwrite("GenImages/" + ix.replace(".png", "") + "_" + jx.replace(".png", "") + "_" + str(letterRot) + ".png", target)
        count += 1

print(count)
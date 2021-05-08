import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from random import randint
import random
import numpy as np
import imutils
import time

def rotateImage(img, degree):
    #frame = imutils.rotate_bound(image, degree)
    if degree == 90:
        out=cv2.transpose(img)
        out=cv2.flip(out,flipCode=1)
    elif degree == 180:
        out=cv2.transpose(img)
        out=cv2.flip(out,flipCode=1)
        out=cv2.transpose(out)
        out=cv2.flip(out,flipCode=1)
    elif degree == 270:
        out=cv2.transpose(img)
        out=cv2.flip(out,flipCode=0)
    elif degree == 0: 
        return img
    return out

def adjustBrightness(image, brightness):

    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV) 
   
    hsv[...,2] = hsv[...,2]+brightness
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

def blur(image, ksize):
    return cv2.blur(image, ksize)

def zoom(image, x1, y1, x2, y2):
    oSize = (image.shape[0], image.shape[1])
    crop_img = image[y1:y2, x1:x2]
    #print(oSize)
    return cv2.resize(crop_img, oSize)

def generateRandomTarget(image):
    mu, sigma = 0, 0.1 # mean and standard deviation
    s = np.random.normal(mu, sigma, 1000)
    ranBlur = int((np.random.normal(loc=25.0, scale=10.0, size=None))//1)
    print(ranBlur)
    x = min([ranBlur, image.shape[0]])
    image = blur(image, (max([ranBlur//2, 1]), max([ranBlur//4, 1])))
    image = rotateImage(image, [0, 270, 180, 90][random.randint(0, 3)])
    return image

def generateAllTargets(path, outputPath, totalImages):
    imgInFolder = len(listdir(path))
    count =0 
    for image_path in listdir(path):
        for i in range(0, totalImages//imgInFolder):
            # create the full input path and read the file
            input_path = join(path, image_path)
            image_to_rotate = cv2.imread(input_path)

            # rotate the image
            newTar = generateRandomTarget(image_to_rotate)

            # create full output path, 'example.jpg' 
            # becomes 'rotate_example.jpg', save the file to disk
            fullpath = join(outputPath, str(count) + "_" +image_path)
            cv2.imwrite(fullpath, newTar)
            count = count + 1

#image = cv2.imread("test.jpg")
#image = blur(image, (20, 20))
#image = adjustBrightness(image, 1)
#image = zoom(image, 100, 100, 300, 300)
#cv2.imshow("test1", image)
#cv2.imshow("test1", blur(image, (1, 30)))
#cv2.imshow("test2", blur(image, (30, 1)))
#cv2.imshow("test3", blur(image, (30, 30)))
#image = rotateImage(image, 180)

#cv2.imshow("ssss", generateRandomTarget(image))

#cv2.waitKey(1000)
#generateAllTargets("Images/2019 Competition/submitted", "Images/2019 Competition/newTest", 48)
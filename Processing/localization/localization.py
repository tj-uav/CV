import cv2, numpy as np, time, os
import math
from math import pi

def localize(image, locX, locY, theta, altitude, planeLocX, planeLocY, planeAngle):
    '''
    Arguements: the image that will be processed, x value of the location of the edge on the origional image,
    y value of the location of the edge on the origional image, half of the angle of view from the camera that
    we are using, the height at which the plane is flying, the current location of the plane (longitude), the
    current location of the plane (latitude), current direction the plane is facing (clockwise from north)
    '''
    theta = theta * pi / 180
    planeAngle = - planeAngle * pi / 180 #DONT KNOW IF THIS SHOULD BE NEGATIVE
    height = image.shape[0]
    width = image.shape[1]

    centerX = int(width/2)
    centerY = int(height/2)
    currentPosition = (centerX, centerY)

    widthFeet = 2*altitude*math.tan(theta) #Finds the width of the picture in feet

    conversionPixelToFeet = widthFeet/width #Gives a value that is equal to feet per pixel

    distanceX = int(locX - centerX) #Finds x distance between center and locX
    distanceY = int(locY - centerY) #Find y distance between center and locY

    feetDistX = distanceX * conversionPixelToFeet
    feetDistY = distanceY * conversionPixelToFeet

    feetDistEast = (feetDistX*math.cos(planeAngle)) - (feetDistY*math.sin(planeAngle))
    feetDistNorth = (feetDistY*math.sin(planeAngle)) + (feetDistY*math.cos(planeAngle))

    longDistX = feetDistEast / 280156.8 #Conversion from Feet to Change in degrees longitude (NOT EXACT, NEED TO FIND BETTER OPTION)
    latDistY = feetDistNorth / 364267.2 #Conversion from Feet to Change in degrees latitude (NOT EXACT, NEED TO FIND BETTER OPTION)

    finalLat = latDistY + planeLocY
    finalLong = longDistX + planeLocX

    print (finalLat)
    print (finalLong)
    print ("Distance between center and edge: " + str((feetDistX**2 + feetDistY**2)**.5))
    cv2.circle(image ,(centerX, centerY), 5, (0,0,255), -1)
    cv2.circle(image ,(locX, locY), 5, (255,0,0), -1)
    cv2.imshow("Test", image)


directoryString = r"/Users/daniel/Documents/UAV/"
img = cv2.imread(directoryString + "localizationPic1.jpg")
img = cv2.resize(img, (int(img.shape[1]*.2), int(img.shape[0]*.2)))
localize(img, 275, 235, 30.5792, 11, 0, 0, 30)
cv2.waitKey(0)

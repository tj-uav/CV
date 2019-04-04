import cv2
import sys
import numpy as np
#from sklearn.cluster import KMeans

import time #temp

def main():
    global colorDict
    colorDict = {}
    filename = 'dependencies/color_hexes.txt'
    try:
        file = open(filename, "r")
        lines = file.readlines()
        for line in lines:
            [color, rgb] = line.split(" ")
            r = int(rgb[0:2],16)
            g = int(rgb[2:4],16)
            b = int(rgb[4:6],16)
            colorDict[color.strip()] = [r,g,b]
        file.close()
    except:
        print("File not found")
    print( "targetIsolation.py is being run independently, continuing with default image" )
    try:
        gimg = cv2.imread( "dependencies/generictarget2.jpg" )
    except:
        print( "Error: Dependency missing: generictarget2.png" )
        sys.exit( 0 )
    [fmask,isolated] = isolateTargetUnique( gimg )
    cv2.imwrite( "output.png", isolated )
    print(type(isolated))
    print(dominantColor(isolated,fmask))

def detectColor(pixel):
    global colorDict
    [b,g,r] = pixel
    closestColor = ""
    closestDistance = float("inf")
    for color in colorDict:
        dist = 0
        dist += abs(colorDict[color][0]-r)
        dist += abs(colorDict[color][1]-g)
        dist += abs(colorDict[color][2]-b)
        if dist < closestDistance:
            closestDistance = dist
            closestColor = color
    return closestColor

def dominantColor(image,fmask):
    global colorDict
    colorCount = {}
    for a in range(len(image)):
        for b in range(len(image[0])):
            if fmask[a][b] == 0:
                continue
            pixel = image[a][b]
            color = detectColor(pixel)
            if color in colorCount:
                colorCount[color] += 1
            else:
                colorCount[color] = 1
#            print(color)
    maxcolor = ""
    maxnum = -1
    for color in colorCount:
        if colorCount[color] > maxnum:
            maxnum = colorCount[color]
            maxcolor = color
    return maxcolor

def isolateTargetUnique( croppedimage ):
    #Scales image
    scaledimg = cv2.resize( croppedimage, ( 200, 200 ) )
    #Quantization
    n = 4
    indices = np.arange( 0, 256 )
    divider = np.linspace( 0, 255, n + 1 )[ 1 ]
    quantiz = np.int0( np.linspace( 0, 255, n ) )
    color_levels = np.clip( np.int0( indices / divider ), 0, n - 1 )
    palette = quantiz[ color_levels ]
    im2 = palette[ scaledimg ]
    scaledimg2 = cv2.convertScaleAbs( im2 )
    #Target isolation
    crop1 = scaledimg2[ 0:50, 0:200 ]
    crop2 = scaledimg2[ 150:200, 0:200 ]
    unsortedcolors = np.concatenate( ( crop1, crop2 ) )
    onedunsort = unsortedcolors.reshape( ( unsortedcolors.shape[ 0 ] * unsortedcolors.shape[ 1 ], 3 ) )
    colors = np.unique( onedunsort, axis = 0 )
    targetcrop = scaledimg2[ 50:150, 50:150 ]
    fmask = np.zeros( ( 100, 100 ), dtype = np.uint8 )
    for color in colors:
        fmask = cv2.bitwise_or( cv2.inRange( targetcrop, color, color ), fmask )
    return [cv2.bitwise_not(fmask),cv2.bitwise_and( scaledimg[ 50:150, 50:150 ], scaledimg[ 50:150, 50:150 ], mask = cv2.bitwise_not( fmask ) )]
if( __name__ == "__main__" ):
    main()

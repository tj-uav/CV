import cv2
import sys
import numpy as np
from sklearn.cluster import KMeans

import time #temp

def main():
    print( "targetIsolation.py is being run independently, continuing with default image" )
    try:
        gimg = cv2.imread( "dependencies/generictarget.jpg" )
    except:
        print( "Error: Dependency missing: generictarget.png" )
        sys.exit( 0 )
    #isolateTargetDominant( gimg, 16 )
    cv2.imwrite( "output.png", isolateTargetUnique( gimg ) )

def isolateTargetUnique( croppedimage ):
    scaledimg = cv2.resize( croppedimage, ( 200, 200 ) )
    crop1 = scaledimg[ 0:50, 0:200 ]
    crop2 = scaledimg[ 150:200, 0:200 ]
    unsortedcolors = np.concatenate( ( crop1, crop2 ) )
    onedunsort = unsortedcolors.reshape( ( unsortedcolors.shape[ 0 ] * unsortedcolors.shape[ 1 ], 3 ) )
    colors = np.unique( onedunsort, axis = 0 )
    targetcrop = scaledimg[ 50:150, 50:150 ]
    fmask = np.zeros( ( 100, 100 ), dtype = np.uint8 )
    for color in colors:
        fmask = cv2.bitwise_or( cv2.inRange( targetcrop, color, color ), fmask )
    return cv2.bitwise_and( targetcrop, targetcrop, mask = cv2.bitwise_not( fmask ) )
if( __name__ == "__main__" ):
    main()

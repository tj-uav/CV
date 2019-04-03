import cv2
import sys
import numpy as np
from sklearn.cluster import KMeans

import time #temp

def main():
    print( "targetIsolation.py is being run independently, continuing with default image" )
    try:
        gimg = cv2.imread( "dependencies/generictarget2.jpg" )
    except:
        print( "Error: Dependency missing: generictarget2.png" )
        sys.exit( 0 )
    cv2.imwrite( "output.png", isolateTargetUnique( gimg ) )

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
    return cv2.bitwise_and( scaledimg[ 50:150, 50:150 ], scaledimg[ 50:150, 50:150 ], mask = cv2.bitwise_not( fmask ) )
if( __name__ == "__main__" ):
    main()

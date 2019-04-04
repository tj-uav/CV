#Imports
import cv2
import sys
import numpy as np

#Various setups
kern = np.ones( ( 3, 3 ), np.uint8 )

def main():
    checkDependencies()
    checkMainDependencies()
    print( "targetIsolation.py is being run independently, continuing with default image" )
    isolated, targetmask = isolateTargetUnique( cv2.imread( "dependencies/generictarget2.jpg" ) )
    shapeColorAlpha = hsv2name( cv2.cvtColor( np.array( [ np.array( [ dominant( cv2.blur( cv2.dilate( isolated, kern, iterations = 3 ), ( 5, 5 ) ), targetmask ) ] ) ] ), cv2.COLOR_BGR2HSV ) )   #AAAAAHHHHHHHHHHH
    print( shapeColorAlpha )
#def getProperties( roi ):    #Nonmain method of getting things #Need to complete, this'll be the method used in competition
#    checkDependencies()

#Dependencies Checker
maindependencies = [ "generictarget.jpg", "generictarget2.jpg", "generictarget3.jpg"]   #Dependencies required when running independently
dependencies = [ "color_hexes.txt" ]                            #Independencies always in use
def checkDependencies():
    global dependencies
    for dependent in dependencies:
        try:
            temp = open( "dependencies/" + dependent )
        except FileNotFoundError:
            print( "Dependency not found: " + dependent )
            sys.exit( 1 )
def checkMainDependencies():
    global maindependencies
    for dependent in maindependencies:
        try:
            temp = open( "dependencies/" + dependent )
        except FileNotFoundError:
            print( "Dependency not found: " + dependent )
            sys.exit( 1 )

#Isolation methods
def isolateTargetUnique( croppedimage ):
    #Scales image
    scaledimg = cv2.resize( croppedimage, ( 200, 200 ) )
    #Quantization
    scaledimg2 = quantize( scaledimg, 16 )
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
    return [ cv2.bitwise_and( scaledimg[ 50:150, 50:150 ], scaledimg[ 50:150, 50:150 ], mask = cv2.bitwise_not( fmask ) ), cv2.bitwise_not( fmask ) ]
def quantize( image, n ):
    indices = np.arange( 0, 256 )
    divider = np.linspace( 0, 255, n + 1 )[ 1 ]
    quantiz = np.int0( np.linspace( 0, 255, n ) )
    color_levels = np.clip( np.int0( indices / divider ), 0, n - 1 )
    palette = quantiz[ color_levels ]
    im2 = palette[ cv2.bitwise_and( image, image ) ]
    f = cv2.convertScaleAbs( im2 )
    return f
def dominant( image, mask ):     #Dominant colors, with a mask! (tm)
    pdata = image.reshape( ( image.shape[ 0 ] * image.shape[ 1 ], 3 ) )
    omask = mask.reshape( ( mask.shape[ 0 ] * mask.shape[ 1 ], 1 ) )
    data = []
    for i in range( 0, len( pdata ) ):
        if omask[ i ] == [ 255 ]:
            data.append( pdata[ i ] )
    data = np.array( data )
    f = np.float32( data )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0 )
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness,labels,centers = cv2.kmeans( f, 1, None, criteria, 10, flags)
    return( centers[ 0 ].astype( np.uint8 ) )
def hsv2name( hsv ):    #Opencv h value is 0->180, not 0->360
    h = hsv[ 0 ][ 0 ][ 0 ]
    s = hsv[ 0 ][ 0 ][ 1 ]
    v = hsv[ 0 ][ 0 ][ 2 ]
    if s < 15: #Low Saturation
        if v < 30: #Low Value
            return( "Black" )
        elif v < 200: #Mid Value
            return( "Gray" )
        else:   #High Value
            return( "White" )
    else:   #Anything not low saturation is a valid color
        if 165 < h or h < 8:    #Red
            return( "Red" )
        elif h < 17:            #Orange/Brown
            if v < 220:
                return( "Brown" )
            else:
                return( "Orange" )
        elif h < 35:            #Yellow
            return( "Yellow" )
        elif h < 75:
            return( "Green" )
        elif h < 132:
            return( "Blue" )
        else:
            return( "Purple" )
#Starts main if run independently
if( __name__ == "__main__" ):
    main()

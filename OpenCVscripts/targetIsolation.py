#Imports
import cv2
import sys
import numpy as np
from sklearn.cluster import MiniBatchKMeans

#Various setups
kern = np.ones( ( 3, 3 ), np.uint8 )

def main():
    checkDependencies()
    checkMainDependencies()
    print( "targetIsolation.py is being run independently, continuing with default image" )
    isolated, targetmask = isolateTarget( cv2.imread( "dependencies/generictarget.jpg" ) )
    isolatedLetter = isolateLetter( isolated, targetmask )
    shapeColorAlpha = hsv2name( cv2.cvtColor( np.array( [ np.array( [ dominantKMeans( cv2.blur( cv2.dilate( isolated, kern, iterations = 3 ), ( 5, 5 ) ), targetmask ) ] ) ] ), cv2.COLOR_BGR2HSV ) )   #AAAAAHHHHHHHHHHH
    print( "Target Color: " + shapeColorAlpha )
    letterColorAlpha = hsv2name( cv2.cvtColor( np.array( [ np.array( [ dominantKMeans( cv2.blur( cv2.dilate( isolatedLetter, kern, iterations = 3 ), ( 5, 5 ) ), targetmask ) ] ) ] ), cv2.COLOR_BGR2HSV ) )   #NO GO AWAY FOUL BEAST
    print( "Letter Color: " + letterColorAlpha )
#def getProperties( roi ):    #Nonmain method of getting things #Need to complete, this'll be the method used in competition
#    checkDependencies()

#Dependencies Checker
maindependencies = [ "generictarget.jpg", "generictarget2.jpg", "generictarget3.jpg" ]   #Dependencies required when running independently
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
def isolateTarget( croppedimage ):
    #Scales image
    scaledimg = cv2.resize( croppedimage, ( 400, 400 ) )
    #Quantization
    scaledimg2 = quantize( scaledimg, 16 )
    #Target isolation
    crop1 = scaledimg2[ 0:100, 0:400 ]
    crop2 = scaledimg2[ 300:400, 0:400 ]
    unsortedcolors = np.concatenate( ( crop1, crop2 ) )
    onedunsort = unsortedcolors.reshape( ( unsortedcolors.shape[ 0 ] * unsortedcolors.shape[ 1 ], 3 ) )
    colors = np.unique( onedunsort, axis = 0 )
    targetcrop = scaledimg2[ 100:300, 100:300 ]
    fmask = np.zeros( ( 200, 200 ), dtype = np.uint8 )
    for color in colors:
        fmask = cv2.bitwise_or( cv2.inRange( targetcrop, color, color ), fmask )
    return [ cv2.bitwise_and( scaledimg[ 100:300, 100:300 ], scaledimg[ 100:300, 100:300 ], mask = cv2.bitwise_not( fmask ) ), cv2.bitwise_not( fmask ) ]
def isolateLetter( target, mask ):
    ori = target
    target = kMeansQuantize( target, 3 )    #Black background counts as one
    domcolor = dominantSimple( target, mask )
    target = cv2.bitwise_not( cv2.inRange( target, domcolor, domcolor ), mask = mask )
    target = cv2.bitwise_and( cv2.dilate( cv2.erode( target, kern, iterations = 2 ), kern, iterations = 3 ), target )
    return cv2.bitwise_and( ori, ori, mask = target )
def quantize( image, n ):
    indices = np.arange( 0, 256 )
    divider = np.linspace( 0, 255, n + 1 )[ 1 ]
    quantiz = np.int0( np.linspace( 0, 255, n ) )
    color_levels = np.clip( np.int0( indices / divider ), 0, n - 1 )
    palette = quantiz[ color_levels ]
    im2 = palette[ cv2.bitwise_and( image, image ) ]
    f = cv2.convertScaleAbs( im2 )
    return f
def kMeansQuantize( image, n ):
    image = cv2.cvtColor( image, cv2.COLOR_BGR2LAB )    #Fancy!
    w, h, _ = image.shape
    image = image.reshape( ( image.shape[ 0 ] * image.shape[ 1 ], 3 ) )
    clt = MiniBatchKMeans( n_clusters = n )
    labels = clt.fit_predict( image )
    quant = clt.cluster_centers_.astype( "uint8" )[ labels ]
    quant = quant.reshape( ( h, w, 3 ) )
    image = image.reshape( ( h, w, 3 ) )
    quant = cv2.cvtColor( quant, cv2.COLOR_LAB2BGR )
    image = cv2.cvtColor( image, cv2.COLOR_LAB2BGR )
    return quant
def dominantKMeans( image, mask ):     #Dominant colors, with a mask! (tm)
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
def dominantSimple( image, mask ):
    samples = []
    for x in range( len( image[ 0 ] ) ):
        for y in range( len( image[ 1 ] ) ):
            if mask[ x ][ y ] == 0:
                continue
            samples.append( image[ x ][ y ] )
    colors, count = np.unique( np.array( samples ), axis = 0, return_counts = True )
    return colors[ count.argmax() ]
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

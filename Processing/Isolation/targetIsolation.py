#### TargetIsolation.py. Please, to all that work on this code on in the future, please maintain this chart here, and maintain descriptive commenting that explains the logic behind steps taken. ~Alex Black, 2021
### Structure                                   #
## ⮡ Imports                                    # All imports with descriptions can be found under the
## ⮡ Variables                                  # Any variable that needs to be 'global' per say or should be easy to access
## ⮡ Starter/Primary Methods                    #
##      ⮡ main()                                # Starts if this program is started independently
##      ⮡ isolate( roiCrop )                    # Runs through all steps to find required properties of a 400x400 input, target assumed in center
## ⮡ Helper Methods                             #
##      ⮡ dependency checker methods            #
##          ⮡ checkDependencies()               # Ensures any files required to launch this program exists. Refrain from referencing files in main() that are not verified by checkDependencies()
##          ⮡ checkMainDependencies()           # Ensures any files required to launch this program independently exists. Refrain from referencing files in main() that are not verified by checkMainDependencies()
##     ⮡ isolation methods                      #
##          ⮡ isolateTarget( croppedImage )     # Takes a 400x400 image with target assumed in center and attempts to remove the background to the target
##          ⮡ isolateLetter( target, mask )     # Takes a 200x200 image with target assumed in center and without a background to remove everything but the letter
##          ⮡ quantize( image, n )              # Takes an image and reduces the image to n intensities per color channel (numpy)
##          ⮡ kMeansQuantize( image, n )        # Takes an image and reduces the image to the n most dominant (ie common) colors
##          ⮡ dominantKMeans( image, mask )     # Determines the most common color in the mask area using kMeans
##          ⮡ dominantSimple( image, mask )     # Determines the most common color in the mask area by sampling intensities in each color channel (numpy)
##          ⮡ hsv2name( hsv )                   # Converts an OpenCV HSV array to an appropriate name. This outputs the recommended channel values, and is prefered over bgr2name.
##          ⮡ bgr2name( bgr )                   # Converts an OpenCV BGR array to an appropriate name. May output different results than hsv2name

#Imports
import cv2                                          # The OpenCV library itself. For Computer Vision.
import sys                                          # Allows advanced storage disk functions
import numpy as np                                  # Library to call C++ array functions. Vital for advanced OpenCV
from sklearn.cluster import MiniBatchKMeans         # Easier to use version of opencv's kmeans system

#Various setups
kern = np.ones( ( 3, 3 ), np.uint8 )                # Quick variable to modify noise reduction steps taken in isolateLetter
#relativeFilePath = "/media/data/Projects/UAV/CV/OpenCVscripts/" # A variable used to fix relative file paths if a system does not support it (VS Code debugger for example). Set to "" if running from command line.
relativeFilePath = ""


def main():                                         # Prints the colors of the shape, the two most difficult properties to get correct.
    checkMainDependencies()
    image,shape,targetmask,letter,lettermask,shapecolor,lettercolor = isolate( cv2.imread( relativeFilePath + "dependencies/generictarget.jpg" ) ) #waifu2xgenerictargets/generictarget5.png" ) )
    print( "Target Color: " + shapecolor )
    print( "Letter Color: " + lettercolor )
    #cv2.imwrite( "d.png", shape );


def isolate( roiCrop ):                             # Runs through the entire process of reducing a region-of-interest
    shapewithoutmask, shapewithmask, targetmask = isolateTarget( roiCrop )
    isolatedLetter, maskLetter = isolateLetter( shapewithmask, targetmask )
    shapecolor = hsv2name( cv2.cvtColor( np.array( [ np.array( [ dominantSimple( shapewithmask, targetmask ) ] ) ] ), cv2.COLOR_BGR2HSV ) )     # Grabs a color, coverts it to a 1x1 OpenCV image, converts it to HSV, then passes said data to get color name
    letterColor = hsv2name( cv2.cvtColor( np.array( [ np.array( [ dominantSimple( isolatedLetter, maskLetter ) ] ) ] ), cv2.COLOR_BGR2HSV ) )
    return shapewithoutmask, shapewithmask, targetmask, isolatedLetter, maskLetter, shapecolor, letterColor


#Dependencies Checker
maindependencies = [ "generictarget.jpg", "generictarget2.jpg", "generictarget3.jpg", "generictarget4.jpg", "generictarget5.jpg" ]   #Dependencies required when running independently
dependencies = [ "color_hexes.txt" ]                            #Dependencies always in use


def checkDependencies():                            # Checks to ensure all required files in dependencies[] exist in ./dependencies
    global dependencies
    for dependent in dependencies:
        try:                                        # Goes to except if open() fails.
            temp = open( relativeFilePath + "dependencies/" + dependent )
        except FileNotFoundError:
            print( "Dependency not found: " + dependent )
            sys.exit( 1 )                           # Exits if a file is missing, prining the offending data


def checkMainDependencies():                        # Checks to ensure all required files in maindependencies[] exist in ./dependencies
    global maindependencies
    for dependent in maindependencies:
        try:                                        # Goes to except if open() fails.
            temp = open( relativeFilePath + "dependencies/" + dependent )
        except FileNotFoundError:
            print( "Dependency not found: " + dependent )
            sys.exit( 1 )                           # Exits if a file is missing, printing the offending data


def isolateTarget( croppedimage ):                  # Samples the border 100 pixels on each side from a reduced color set then removes those from all parts of the image, leaving the target.
    ori = croppedimage
    #scaledimg = cv2.blur( cv2.resize( croppedimage, ( 400, 400 ), cv2.INTER_CUBIC ), ( 3, 3 ) )     # Ensures the image is 400x400 pixels, then removes noise using blur()
    scaledimg = cv2.bilateralFilter( cv2.resize( croppedimage, ( 400, 400 ), cv2.INTER_CUBIC ), 7, 75, 75 )
    scaledimg2 = quantize( scaledimg, 16 )          # Reduces the number of colors in the image by allowing 16 intensities per channel
    crop1 = scaledimg2[ 0:100, 0:400 ]              # Makes two crops as the area to sample colors from
    crop2 = scaledimg2[ 300:400, 0:400 ]
    unsortedcolors = np.concatenate( ( crop1, crop2 ) )     # Merges the two images side-by-sideW
    onedunsort = unsortedcolors.reshape( ( unsortedcolors.shape[ 0 ] * unsortedcolors.shape[ 1 ], 3 ) )     # Puts all pixels into a one dimensional array, invalid as a OpenCV image. Required for np.unique()
    colors = np.unique( onedunsort, axis = 0 )      # Finds all reoccuring pixel colrs, and removes them fom the array, thus creating a 'list' of colors to remove
    targetcrop = scaledimg2[ 100:300, 100:300 ]     # Crops the center area that was not sampled for colors. Assumes target in here
    fmask = np.zeros( ( 200, 200 ), dtype = np.uint8 )  # Creates a new image used to layer areas below.
    for color in colors:                            # Adds any area that matches the unique color currently loaded in the variable color
        fmask = cv2.bitwise_or( cv2.inRange( targetcrop, color, color ), fmask )    # Matches a color to the variable color using inRange(), then combines it with preexisting data from previous steps with bitwise_or()
    fmask = cv2.bitwise_not( fmask )                # Inverts fmask to change it from a map of where the background is to where it isn't (thus the location of the target)
    fscaledimg = scaledimg[ 100:300, 100:300 ]      # Crops the unmodified image so that the sampled area is not included
    return [ ori, cv2.bitwise_and( fscaledimg, fscaledimg, mask = fmask ), fmask ]  # Returns the input, a crop of the original image with the background removed, then a mask of the target


def isolateLetter( target, mask ):                  # Takes an input of the cropped (200x200 target-centered) original image with or without the background, as well as the map of where it is.
    ori = target                                    # Makes a copy of the original image
    target = kMeansQuantize( target, 3 )            # Reduces the image to the three most common colors in it (background black, outer shape, inner letter). This step is often slightly inconsistent between runs.
    domcolor = dominantSimple( target, mask )       # Finds the more common color in the shape (thanks to the mask)
    mask = cv2.erode( cv2.dilate( mask, kern, iterations = 1 ), kern, iterations = 5 )  # Cleans up any little imperfections in the mask. Works by making tiny imperfections smaller then rescaling the shape back up.
    target = cv2.bitwise_not( cv2.inRange( target, domcolor, domcolor ), mask = mask )  # Gets the location of the more common color in the target, then inverts it and puts the mask back on to get the letter's location
    return cv2.bitwise_and( ori, ori, mask = target ), target   # Returns the original image with only the letter shown, as well as the letter's mask


def quantize( image, n ):                           # Uses numpy slicing to quantize the most common intensities in each channel.
    indices = np.arange( 0, 256 )
    divider = np.linspace( 0, 255, n + 1 )[ 1 ]
    quantiz = np.int0( np.linspace( 0, 255, n ) )
    color_levels = np.clip( np.int0( indices / divider ), 0, n - 1 )
    palette = quantiz[ color_levels ]
    im2 = palette[ cv2.bitwise_and( image, image ) ]
    f = cv2.convertScaleAbs( im2 )
    return f


def kMeansQuantize( image, n ):                     # Uses k-Means to quantize an image into n colors.
    image = cv2.cvtColor( image, cv2.COLOR_BGR2LAB )# The LAB color space is especially useful for this type of system, as it's similar to how a human eye works
    w, h, _ = image.shape                           # Gets the shape of the image for the next step
    image = image.reshape( ( image.shape[ 0 ] * image.shape[ 1 ], 3 ) ) #k-Means required an image to be one pixel tall for some reason
    clt = MiniBatchKMeans( n_clusters = n )         # Creates a new kMeans system. Cluster count tells how many regions of best fit should be made.
    labels = clt.fit_predict( image )               # This takes a aet of inputted colors and tries to find clusters of data. Similar to how one might move a '3d cursor' to the center of data on -a 3d mqp
    quant = clt.cluster_centers_.astype( "uint8" )[ labels ]    #Replaces every pixel with the closewt color k-Means found
    quant = quant.reshape( ( h, w, 3 ) )            # Rescales the quantized image back into its original shape. Simply imagine the pixels line wrqpping similar to a wore processor
    quant = cv2.cvtColor( quant, cv2.COLOR_LAB2BGR )# Reverts the color as well
    return quant


def dominantKMeans( image, mask ):     # Uses k-Means to obtain the most common color in the image
    pdata = image.reshape( ( image.shape[ 0 ] * image.shape[ 1 ], 3 ) ) # k-Means requires the image to be one pixel tall
    omask = mask.reshape( ( mask.shape[ 0 ] * mask.shape[ 1 ], 1 ) )    # Do the same for the mask
    #data = []                                       # Something something that's really slow
    #for i in range( 0, len( pdata ) ):
    #    if omask[ i ] == [ 255 ]:
    #        data.append( pdata[ i ] )
    #data = np.array( data )
    f = cv2.bitwise_and( pdata, pdata, mask = omask )   # Masks off the non-important regions
    f = np.float32( pdata )                          # Adds precision to the numbers when calculating
    criteria = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0 )  # Black magic, sets up OpenCV Parameters
    flags = cv2.KMEANS_RANDOM_CENTERS               # Makes the next step use random points, also is black magic
    compactness,labels,centers = cv2.kmeans( f, 1, None, criteria, 10, flags)   # Moves those random points until a point is basically in a cluster of similarly-colored pixels
    return( centers[ 0 ].astype( np.uint8 ) )       # Removes the precision needed for k-means


def dominantSimple( image, mask ):
    samples = []                                    # Setup for this has to use non-array setups
    for x in range( len( image[ 0 ] ) ):            # For every x row
        for y in range( len( image[ 1 ] ) ):        # And every pixel on said row
            if mask[ x ][ y ] == 0:                 # Check if a point should be in a mask
                continue
            samples.append( image[ x ][ y ] )       # If it should, it's added to the pool of colors
    colors, count = np.unique( np.array( samples ), axis = 0, return_counts = True )    # Uses numpy to locate the unique colors. Is black magic
    return colors[ count.argmax() ]                 # Returns the most dominant colors


def hsv2name( hsv ):    # Approximates the color of a 1x1 image into a valid name
    h = hsv[ 0 ][ 0 ][ 0 ]  # Imagine an image of size 1*1
    s = hsv[ 0 ][ 0 ][ 1 ]  # Each of these is a channel
    v = hsv[ 0 ][ 0 ][ 2 ]  # Representing Hue, Saturation, and Value. OpenCV uses non-standard value
    if s < 25: #Low Saturation
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
            if v < 180:
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


def bgr2name( bgr ):    # Not recommended for use, many colors have not been verified
    [b,g,r] = bgr[0][0]
    if max(abs(b-r),abs(b-g),abs(g-r)) < 50: #Distance between r,g,b values is less than 50
        avg = (b+r+g)/3
        if avg < 80:
            return ( "Black" )
        elif avg < 160:
            return ( "Gray" )
        else:
            return ( "White" )
    if r > 175: #High red value:
        if g < 60:
            if b < 70:
                return ( "Red" )
            else:
                return ( "Purple" )
        elif g > 200:
             return ( "Yellow" )
        else:
             return ( "Orange" )
    elif r > 100:
        if g < 80 and b < 80:
            return ( "Brown" )
    if b > 200 and r > 200 and g > 200:
        return( "White" )
    if b > 200 and r > 200 and g > 200:
        return( "White" )
    if b > 200 and r > 200 and g > 200:
        return( "White" )
    if b > 200 and r > 200 and g > 200:
        return( "White" )
    if b > 200 and r > 200 and g > 200:
        return( "White" )


if( __name__ == "__main__" ):                       # Checks execution of program, runs main() if nessesary. Must stay at end
    main()

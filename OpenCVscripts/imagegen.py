##Ensure your CV_DATA folder exists with relative path ../../CV_DATA

#Start variables
imagesToGenerate = 10

#Imports
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from random import randint
import imutils
import time

#Find and load files used for generation into memory
samples = [ cv2.imread( "../../CV_DATA/groundtextures/" + str( i ) ) for i in listdir( "../../CV_DATA/groundtextures/" ) if isfile( join( "../../CV_DATA/groundtextures/", i ) ) ]
letters = [ cv2.imread( "../../CV_DATA/letterpatterns/" + str( i ) ) for i in listdir( "../../CV_DATA/letterpatterns/" ) if isfile( join( "../../CV_DATA/letterpatterns/", i ) ) ]
shapes = [ cv2.imread( "../../CV_DATA/shapepatterns/" + str( i ) ) for i in listdir( "../../CV_DATA/shapepatterns/" ) if isfile( join( "../../CV_DATA/shapepatterns/", i ) ) ]

#Generate Images
for i in range( 0, imagesToGenerate ):
    samplesnum = randint( 0, len( samples ) - 1 )   #Pick image out of lists
    lettersnum = randint( 0, len( letters ) - 1 )
    shapesnum = randint( 0, len( shapes ) - 1 )

    back = samples[ samplesnum ]                    #Rereference
    lett = letters[ lettersnum ]
    shap = shapes[ shapesnum ]

    lett = imutils.rotate_bound( lett, randint( 0, 360 ) )  #Rotate images without accidental crop
    shap = imutils.rotate_bound( shap, randint( 0, 360 ) )

    lett = cv2.resize( lett, ( 26, 26 ) )           #Resizes as final size
    shap = cv2.resize( shap, ( 66, 66 ) )

    lettmask = cv2.inRange( lett, np.array( [ 1, 1, 1 ] ), np.array( [ 255, 255, 255 ] ) )  #Cleans images, converts to mask
    shapmask = cv2.inRange( shap, np.array( [ 1, 1, 1 ] ), np.array( [ 255, 255, 255 ] ) )

    lett = cv2.cvtColor( lettmask, cv2.COLOR_GRAY2BGR ) #Reconvert clean mask to letter
    shap = cv2.cvtColor( shapmask, cv2.COLOR_GRAY2BGR )

    lett[ np.where( ( lett == [ 255, 255, 255 ] ).all( axis = 2 ) ) ] = [ randint( 0, 255 ), randint( 0, 255 ), randint( 0, 255 ) ] #Colorize
    shap[ np.where( ( shap == [ 255, 255, 255 ] ).all( axis = 2 ) ) ] = [ randint( 0, 255 ), randint( 0, 255 ), randint( 0, 255 ) ]

    lett = cv2.copyMakeBorder( lett, top = 20, bottom = 20, left = 20, right = 20, borderType = cv2.BORDER_CONSTANT, value= [ 0, 0, 0 ] )   #Rescale letter for copy
    lettmask = cv2.copyMakeBorder( lettmask, top = 20, bottom = 20, left = 20, right = 20, borderType = cv2.BORDER_CONSTANT, value= [ 0 ] )

    lettmask_inv = cv2.bitwise_not( lettmask )                  #Clear the area
    shap = cv2.bitwise_and( shap, shap, mask = lettmask_inv )
    shap = cv2.add( shap, lett )                              #Add colors

    height, width = back.shape[:2]
    randpos = [ randint( 0, height - 66 ), randint( 0, width - 66 ) ]

    shap = cv2.copyMakeBorder( shap, top = randpos[ 0 ], bottom = height - randpos[ 0 ] - 66, left = randpos[ 1 ], right = width - randpos[ 1 ] - 66, borderType = cv2.BORDER_CONSTANT, value= [ 0, 0, 0 ] )
    shapmask = cv2.copyMakeBorder( shapmask, top = randpos[ 0 ], bottom = height - randpos[ 0 ] - 66, left = randpos[ 1 ], right = width - randpos[ 1 ] - 66, borderType = cv2.BORDER_CONSTANT, value= [ 0 ] )

    shapmask_inv = cv2.bitwise_not( shapmask )
    back = cv2.bitwise_and( back, back, mask = shapmask_inv )
    back = cv2.add( back, shap )

    cv2.imwrite( "data/" + str( i ) + ".png", back )

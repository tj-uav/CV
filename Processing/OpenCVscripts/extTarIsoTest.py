import cv2
import numpy as np
import targetIsolation

img = cv2.imread( "/home/alexander/Desktop/i4.JPG" )
shapewithoutmask, shapewithmask, targetmask, isolatedLetter, maskLetter, shapecolor, letterColor = targetIsolation.isolate( ( targetIsolation.crop( img, ( 1944, 1260 ) ) ) )
print( shapecolor, letterColor )

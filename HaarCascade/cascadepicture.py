#uses the cascade
#classifies a single image

import sys
import cv2
import numpy as np
import imutils
import time

def quantize( image, n ):
    indices = np.arange( 0, 256 )
    divider = np.linspace( 0, 255, n + 1 )[ 1 ]
    quantiz = np.int0( np.linspace( 0, 255, n ) )
    color_levels = np.clip( np.int0( indices / divider ), 0, n - 1 )
    palette = quantiz[ color_levels ]
    im2 = palette[ cv2.bitwise_and( image, image ) ]
    f = cv2.convertScaleAbs( im2 )
    return f

frame = cv2.imread( "C:/Users/zz198/Desktop/tj-uav/CV/OpenCVscripts/dependencies/generictarget.jpg")
kern = np.ones( ( 3, 3 ), np.uint8 )
frame = quantize( frame, 8 )
frame = cv2.erode( cv2.dilate( frame, kern, iterations = 3 ), kern, iterations = 3 )
triangle_cascade = cv2.CascadeClassifier("triangledata/cascade10.xml")

detected = 0


#frame = imutils.resize(frame,width=1067,height=600)
frame = imutils.resize(frame,width = 146,height = 86)						#raw is 364,216
#frame = imutils.resize(frame,width = 80,height = 50)						#for pentagon
#frame = imutils.resize(frame,width = 364,height=85)
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#gray = cv2.GaussianBlur(gray, (5,5), 0)
ret, gray = cv2.threshold(gray, 250,255,cv2.THRESH_BINARY)
triangles = triangle_cascade.detectMultiScale(gray,1.3,5)					#(gray,50,1) works best with whiteTriangle_Trim and redtriangle
																			#(gray,1.3,5) works well for differing sizes

for(x,y,w,h) in triangles:
	print((x,y,w,h),detected)
	cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)

	#for drawing larger crops
	cv2.rectangle(frame,(x+w//2-200,y+h//2-200),(x+w//2+200,y+h//2+200),(0,255,255),2)
	detected+=1

cv2.imshow("frame",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

#For triangledata/cascade10.xml
#Arguments		white_triangle detected		redsquare detected
#Goal			163							0
#(gray,50,1)	163
#(gray,50,2)								133
#(gray,50,3)	45							98
#(gray,50,1)	163
#(gray,20,1)	163							171

import cv2
import math
import numpy as np

# Alex Black
# Set of methods to simulate expected lens issues.
# Sensor artifacts, using camera properties
def noise( img, iso = 100, signalToNoise = 70 ): # signalToNoise (gaussian) is defined at ISO100 - higher numbers are better. Typically this'll be equal to a camera's pixel pitch time 18, but research on a per-camera basis.
	var = math.log( iso ** 2 ) / math.log( signalToNoise ) # More of an estimate equation, the actual math behind this seems to be a nightmare
	print( var )
	# Starting here, credit to https://stackoverflow.com/a/30609854
	rows, cols, ch = img.shape 
	mean = 2 * var
	sigma = var ** 2
	g = np.random.normal( mean, sigma, ( rows, cols, ch ) )
	g = g.reshape( rows, cols, ch )
	return img + g

# Near-field distortions. These methods will only simulate distortions at point, and will not apply the distortion to the entire image.
# Other common abberations are ignored simply due to neglectable effects with modern equiptment
def defocus( img, mask, blur = 0 ):	# Standard ol' blur. To simulate the effects of a lens the blur is log-based.
	img = cv2.bitwise_and( img, mask )
	v = int( math.log( blur, 2 ) )
	if v % 2 == 0:
		v += 1
	return cv2.GaussianBlur( img, ( v, v ), 0 )
def axialChromaticAbberation( img, mask, strength = 0 ):	# Blurs channels selectively, strength increases by sqrt of input. Also technically a full distortion - see below.
	b, g, r = cv2.split ( cv2.bitwise_and( img, mask ) )
	adjstrength = int( abs( strength ) / strength ) * math.ceil( abs( strength ) ** .5 / 2. ) * 2 + 1
	g = cv2.GaussianBlur( g, ( abs( adjstrength ), abs( adjstrength ) ), 0 )
	if strength > 0:	# Red channel stays focused
		b = cv2.GaussianBlur( b, ( 2 * adjstrength + 1, 2 * adjstrength + 1 ), 0 )
	else:	# Blue channel stays focused
		r = cv2.GaussianBlur( r, ( -2 * adjstrength + 1, -2 * adjstrength + 1 ), 0 )
	return cv2.merge( ( b, g, r ) )
def transverseChromaticAbberation( img, mask, theta = 0, strength = 0 ):	# Scales channels selectively - strength is linear, theta in radians. This is the classic 'rainbow glitch look' with red as the base.
	v = ( strength * math.cos( theta ), -1 * strength * math.sin( theta ) )	# Reverse y to account for inverse y in images
	img = cv2.bitwise_and( img, mask )
	rows, cols, _ = img.shape
	b, g, r = cv2.split( img )
	Mg = np.float32( [ [1, 0, v[ 0 ] ], [ 0, 1, v[ 1 ] ] ] )
	Mb = np.float32( [ [1, 0, 2 * v[ 0 ] ], [ 0, 1, 2 * v[ 1 ] ] ] )
	g = cv2.warpAffine( g, Mg, ( cols, rows ) )
	b = cv2.warpAffine( b, Mb, ( cols, rows ) )
	return cv2.merge( ( b, g, r ) )
	
# Full distortions - these tend to be significantly more difficult but the most common are implimented here
def imageTransverseChromaticAbberation( img, strength = 0 ):	# This is a full-image transverse chromatic abberation. Slow because of multiple scalings.
	b, g, r = cv2.split( img )
	rows, cols, _ = img.shape
	g = cv2.resize( g, ( cols + 2 * abs( strength ) , rows + 2 * abs( strength ) ), interpolation = cv2.INTER_AREA )
	g = g[ abs( strength ):( abs( strength ) + rows ), abs( strength ):(abs( strength ) + cols ) ]
	if strength > 0:	# Red treated as base.
		b = cv2.resize( b, ( cols + 4 * strength , rows + 4 * strength ), interpolation = cv2.INTER_AREA )
		b = b[ ( 2 * strength ):( 2 * strength + rows ), ( 2 * strength ):( 2 * strength + cols ) ]
	else:	# strength < 0, blue treated as base.
		strength *= -1	# Invert strength value
		r = cv2.resize( r, ( cols + 4 * strength , rows + 4 * strength ), interpolation = cv2.INTER_AREA )
		r = r[ ( 2 * strength ):( 2 * strength + rows ), ( 2 * strength ):( 2 * strength + cols ) ]
	return cv2.merge( ( b, g, r ) )

import os
def main(): # main() method to aid debugging
	img = cv2.imread( 'b.png' )
	img = defocus( img, img, 4 )
	img = axialChromaticAbberation( img, img, 10000 )
	img = transverseChromaticAbberation( img, img, 3, 5 )	# No mask, just use same image as mask
	img = imageTransverseChromaticAbberation( img, strength = 20 )
	img = noise( img, iso = 16000 )
	img = cv2.blur( img, ( 2, 2 ) )	# Simulate de-noise algorithm

	cv2.imwrite( 'b.png', img )

if __name__ == "__main__":
	main()
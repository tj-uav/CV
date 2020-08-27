# Alex Black
# Set of methods to simulate expected lens issues.
# Sensor artifacts, using camera properties
def noise( img, iso = 100, exposure = .005, signalToNoise = 70 ): # signalToNoise (gaussian) is defined at ISO100 - higher numbers are better. Typically this'll be equal to a camera's pixel pitch time 18, but research on a per-camera basis.
	pass

# Near-field distortions. These methods will only simulate distortions at point, and will not apply the distortion to the entire image.
# Other common abberations are ignored simply due to neglectable effects with modern equiptment
def defocus( img, blur = 0 ):
	pass
def tangentialAstigmatism( img, radius = 0, theta = 0, strength = 0 ):	# Basically the blur direction is tangential to the radius at a given angle 
	pass
def sagittalAstigmatism( img, radius = 0, theta = 0, strength = 0 ):	# Blur direction is perpendicular to the radius at a given angle. DOES NOT RETURN A ROTATED tangentialAstigmatism!
	pass
def fieldCurvature( img, radius = 0, strength = 0 ):	# Basically defocus but with arguments similar to other complicated methods
	pass
def axialChromaticAbberation( img, radius = 0, theta = 0, strength = 0 ):	# Blurs channels selectively
	pass
def transverseChromaticAbberation( img, radius = 0, theta = 0, strength = 0 ):	# Scales channels selectively. This is the classic 'rainbow glitch look.'
	pass

def main(): # main() method to aid debugging
	pass

if __name__ == "__main__":
	main()
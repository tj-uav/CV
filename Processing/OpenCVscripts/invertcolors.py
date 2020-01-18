#inputs image and outputs image with inverted colors
#two command line arguments: path to input image, and path of output image

import sys
import cv2


img = cv2.imread(sys.argv[1])

img2 = cv2.bitwise_not(img)

cv2.imwrite(sys.argv[2],img2)

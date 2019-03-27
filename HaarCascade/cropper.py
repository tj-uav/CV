#crops images from file in format: filename numrects x y w h
#python cropper.py inputtxt outputfolder outputdescriptionfile
#python cropper.py bgneg.txt negatives negatives/bg.txt

import sys
import cv2

#make negative description file
bg = open(sys.argv[3],"w")
#for i in range(1,1136):
#	bg.write("neg" + str(i) + ".png\n")
#quit()



f = open(sys.argv[1],"r")
#out = open(sys.argv[3])

count = 1

for line in f.readlines():
	s = line.split(" ")

	y = int(s[2])
	x = int(s[3])
	h = int(s[4])
	w = int(s[5])

	img = cv2.imread(s[0])

	crop = img[x:x+w,y:y+h]

	cv2.imwrite(sys.argv[2]+"/neg"+str(count)+".png",crop)

	bg.write(sys.argv[2]+"/neg"+str(count)+".png\n")

	count+=1

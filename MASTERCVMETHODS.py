#cv methods to extract images from each step

import cv2
import sys
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from keras.models import model_from_json
import imutils
from OpenCVscripts import targetIsolation

#cascade = cv2.CascadeClassifier("HaarCascade/triangledata/cascade10.xml")
def HaarCascade(cascade,frame):
	frame = imutils.resize(frame,width = 146,height = 86)						#raw is 364,216
	#frame = imutils.resize(frame,width = 80,height = 50)						#for pentagon
	#frame = imutils.resize(frame,width = 364,height=85)
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	#gray = cv2.GaussianBlur(gray, (5,5), 0)
	ret, gray = cv2.threshold(gray, 250,255,cv2.THRESH_BINARY)
	rects = cascade.detectMultiScale(gray,1.3,5)					#(gray,50,1) works best with whiteTriangle_Trim and redtriangle

	return rects

#isolatedletter is result of targetisolation.getLetter()
def classifyLetter(isolatedLetter):
	_, threshLetter = cv2.threshold(cv2.cvtColor(isolatedLetter, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
	contours, _ = cv2.findContours(threshLetter, 1, 1)
	cnt = max(contours,key = lambda c: cv2.contourArea(c))
	x, y, w, h = cv2.boundingRect(cnt)
	cv2.rectangle(isolatedLetter,(x,y),(x+w,y+h),(0,255,0),2)
	cropped = threshLetter[y - int(h*bufferFraction): y + h + int(h*bufferFraction), x - int(w*bufferFraction) :x + w + int(w*bufferFraction)]
	# load json and create model
	json_file = open('CNN/alphanumeric/models/alphanumeric_modelBalanced.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights("CNN/alphanumeric/models/alphanumeric_modelBalanced.h5")
	print("Loaded model from disk")

	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	def predict_output(img):
	    print(img.shape)
	    output = model.predict(img.reshape(1, 28, 28, 1) / 255)
	    prediction,confidence,output = test_CNN_EMNIST.actualValue(output[0], output_map)
	    return prediction,confidence,output

	cropped = imutils.rotate(cropped,0)
	#cropped = imutils.cropped(cropped,width=28,height=28)
	cropped = cv2.resize(cropped,(28,28))
	for i in range(0,360,20):
	    print(predict_output(imutils.rotate(cropped,i)))

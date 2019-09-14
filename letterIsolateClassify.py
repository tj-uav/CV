import cv2
import numpy as np
#import sys
from OpenCVscripts import targetIsolation
from CNN.alphanumeric import test_CNN_EMNIST
from keras.models import model_from_json
import imutils
#sys.path.append('OpenCVscripts')
#sys.path.append('CNN/alphanumeric')
#import sys
#sys.path.append('OpenCVscripts')
#sys.path.append('dependencies')

digits_map = [i for i in range(10)]
alpha_map = [chr(i+65) for i in range(26)]
output_map = []
output_map.extend(digits_map)
output_map.extend(alpha_map)

bufferFraction = 0.25
isolatedLetter = targetIsolation.getLetter("OpenCVscripts/dependencies/generictarget3.jpg")
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
    cv2.imshow("Cropped",imutils.rotate(cropped,i))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
cv2.imshow("image", cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()

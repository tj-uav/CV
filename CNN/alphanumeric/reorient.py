import cv2
import numpy as np
import imutils
from keras.models import model_from_json

# load json and create model
json_file = open('models/reorient_newFonts_longer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("models/reorient_newFonts_longer.h5")
print("Loaded model from disk")

def correct_orientation(filename, tHold):
    image = cv2.imread(filename)
#    image = cv2.erode(image, np.ones((2,2),np.uint8))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, tHold, 255, cv2.THRESH_BINARY)

    cnts, ret = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts,key = lambda c: cv2.contourArea(c))
    x,y,w,h = cv2.boundingRect(cnt)
    border = image[y:y+h,x:x+w]
    bigger = max(w, h)
    bigger *= 1.5
    tb = int((bigger - h) / 2)
    rl = int((bigger - w) / 2)
    print(border.shape)
    border = cv2.copyMakeBorder(border, top=tb, bottom=tb, left=rl, right=rl, borderType=cv2.BORDER_CONSTANT,
                                value=[0, 0, 0])
    print(border.shape)
    border = cv2.resize(border, (50, 50))
    cv2.imshow("Cropped", border)
    print(border.shape)
    temp = cv2.cvtColor(border,cv2.COLOR_BGR2GRAY)
    temp = temp / 255
    o = model.predict(temp.reshape(1,50,50,1))
    angle = np.argmax(o, axis=1)
    rotated = imutils.rotate(image, -angle * 5)
    return rotated

filename = 'E.png'
rotated = correct_orientation(filename,100)
cv2.imshow("Rotated",rotated)
cv2.imshow("Original",cv2.imread(filename))
cv2.waitKey(0)
cv2.destroyAllWindows()
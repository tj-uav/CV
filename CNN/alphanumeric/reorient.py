import cv2
import numpy as np
import imutils
from keras.models import model_from_json

# load json and create model
json_file = open('models/alpha_reorient.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("models/alpha_reorient.h5")
print("Loaded model from disk")

def correct_orientation(filename, tHold):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
    ret, thresh = cv2.threshold(gray, tHold, 255, cv2.THRESH_BINARY)

    cnts, ret = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    assert(len(cnts) == 1)
    cnt = cnts[0]
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
    border = cv2.resize(border, (28, 28))
    print(border.shape)
    temp = cv2.cvtColor(border,cv2.COLOR_BGR2GRAY)
    temp = temp / 255
    o = model.predict(temp.reshape(1,28,28,1))
    angle = np.argmax(o, axis=1)
    rotated = imutils.rotate(image, -angle)
    return rotated


filename = 'W.png'
rotated = correct_orientation(filename,100)
cv2.imshow("Rotated",rotated)
cv2.imshow("Original",cv2.imread(filename))
cv2.waitKey(0)
cv2.destroyAllWindows()
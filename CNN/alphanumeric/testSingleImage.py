from keras.models import  model_from_json
import cv2
import imutils
import numpy as np

def global_vars():
    global output_map
    global height,width
    #Dicts for classes
    digits_map = [i for i in range(10)]
    letters_map = [chr(i+65) for i in range(26)]
    output_map  = []
    output_map.extend(digits_map)
    output_map.extend(letters_map)

    height = 28
    width = 28

def main():
    global output_map
    global_vars()
    # load json and create model
    json_file = open('models/alphanumeric_modelBalanced.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("models/alphanumeric_modelBalanced.h5")
    print("Loaded model from disk")

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print('Starting')

    bufferFraction = 0.25
    #Choose Image file here
    img = cv2.imread("C:/Users/Srikar/Downloads/w_new.png")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,img = cv2.threshold(img,120,255,cv2.THRESH_BINARY)
    contours,ret = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key = lambda c: cv2.contourArea(c))
    x,y,w,h = cv2.boundingRect(cnt)
    img = img[y - int(h * bufferFraction): y + h + int(h * bufferFraction), x - int(w * bufferFraction):x + w + int(w * bufferFraction)]

    img = cv2.resize(img,(height,width))
#    img = cv2.erode(img, np.ones((3,3),np.uint8))
    for angle in range(0,360,20):
        rotated = imutils.rotate(img,angle)
        temp = rotated / 255
        output = model.predict(temp.reshape(1,height,width,1))
        print(output)
        prediction,confidence = actualValue(output[0],output_map)
        print("Prediction: " + str(prediction))
        print("Confidence: " + str(confidence))
#        print(output)
        cv2.imshow("Image",rotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def actualValue(output,output_map):
#    print(output)
    maxindex = 0
    for i in range(len(output)):
        if output[i] > output[maxindex]:
            maxindex = i
    return output_map[maxindex],output[maxindex]

if __name__ == '__main__':
    main()
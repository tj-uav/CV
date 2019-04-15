import keras
from keras.models import Sequential, model_from_json
from keras.layers import *
from keras.utils import np_utils
from keras.optimizers import *
import numpy as np
import cv2
from scipy.io import loadmat

#Dicts for classes
classes_dict = {}
classes_dict["digits"] = 10
classes_dict["letters"] = 26
digits_map = [i for i in range(10)]
letters_map = [chr(i+65) for i in range(26)]
outputs_dict = {}
outputs_dict["digits"] = digits_map
outputs_dict["letters"] = letters_map

matfiles = []
matfiles.append((loadmat("C:/Users/Srikar/Documents/NN_Databases/EMNIST/matlab/emnist-digits.mat"),"digits"))
matfiles.append((loadmat("C:/Users/Srikar/Documents/NN_Databases/EMNIST/matlab/emnist-letters.mat"),"letters"))

output_map = []

# Local functions
def rotate(img):
    # Used to rotate images (for some reason they are transposed on read-in)
    flipped = np.fliplr(img)
    return np.rot90(flipped)


height = 28
width = 28

def get_data(matfiles):
    num_classes = 0
    thresh = 120
    total_x_train = np.array([])
    total_y_train = np.array([])
    total_x_test = np.array([])
    total_y_test = np.array([])
    for count,(matfile,name) in enumerate(matfiles):
        print(name)
        nbclasses = classes_dict[name]
        output_map.extend(outputs_dict[name])
        data = matfile['dataset']

        x_test = data['test'][0,0]['images'][0,0]
        y_test = data['test'][0,0]['labels'][0,0]

        diff = num_classes
        if name == "letters":
            diff -= 1

        if diff != 0:
            for i in range(len(y_test)):
                j = y_test[i][0]
                j += diff
                y_test[i] = np.array([j],dtype='uint8')

        num_classes += nbclasses

        if count == 0:
            total_x_test = x_test
            total_y_test = y_test
            continue

        total_x_test = np.concatenate((total_x_test,x_test))
        total_y_test = np.concatenate((total_y_test,y_test))

    print('Thresholding data sets')

    total_x_test[total_x_test <= thresh] = 0
    total_x_test[total_x_test > thresh] = 1
    total_x_test = total_x_test.reshape(len(total_x_test), height, width, 1)

    print("Rotating images")

    for i in range(len(total_x_test)):
        total_x_test[i] = rotate(total_x_test[i])

    print("Shuffling data sets")

    indices_test = np.arange(total_x_test.shape[0])
    np.random.shuffle(indices_test)
    total_x_test = total_x_test[indices_test]
    total_y_test = total_y_test[indices_test]

    total_y_test = np_utils.to_categorical(total_y_test, num_classes)

    return total_x_test, total_y_test


x_test, y_test = get_data(matfiles)
print(x_test.shape)
# load json and create model
json_file = open('alphanumeric_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("alphanumeric_model.h5")
print("Loaded model from disk")

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print('Starting')

def actualValue(output):
#    print(output)
    maxindex = 0
    for i in range(len(output)):
        if output[i] > output[maxindex]:
            maxindex = i
    return output_map[maxindex]

for i in range(len(x_test)):  
    img = x_test[i]
    output = model.predict(img.reshape(1,28,28,1))
    prediction = actualValue(output[0])
    label = actualValue(y_test[i])
    print("Prediction: " + str(prediction) + "\t Actual: " + str(label))
    temp = img.reshape(28,28)
    temp = temp*255
    temp = temp.astype('uint8')
    cv2.imshow("Prediction",temp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#score = model.evaluate(x_test, y_test, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

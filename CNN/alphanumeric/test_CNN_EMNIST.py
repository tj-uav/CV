import keras
from keras.models import Sequential, model_from_json
from keras.layers import *
from keras.utils import np_utils
from keras.optimizers import *
import numpy as np
import cv2
from scipy.io import loadmat

def global_vars():
    global classes_dict,outputs_dict,matfiles,output_map
    global height,width
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
    matfiles.append((loadmat("C:/Users/Srikar/Documents/NN_Databases/EMNIST/matlab/emnist-balanced.mat")))
    output_map = [i for i in range(10)]
    output_map.extend([chr(i+65) for i in range(26)])

    height = 28
    width = 28

# Local functions
def rotate(img):
    # Used to rotate images (for some reason they are transposed on read-in)
    flipped = np.fliplr(img)
    return np.rot90(flipped)

def get_data(matfile):
    num_classes = 36
    thresh = 80
    total_x_train = np.array([])
    total_y_train = np.array([])
    total_x_test = np.array([])
    total_y_test = np.array([])
    for count,matfile, in enumerate(matfiles):
#        print(name)
        data = matfile['dataset']

        x_test = data['test'][0,0]['images'][0,0]
        y_test = data['test'][0,0]['labels'][0,0]

#        y_test = y_test

        if count == 0:
            total_x_test = x_test
            total_y_test = y_test
            continue

        total_x_test = np.concatenate((total_x_test,x_test))
        total_y_test = np.concatenate((total_y_test,y_test))

    total_x_test[total_x_test <= thresh] = 0
    total_x_test[total_x_test > thresh] = 1

    total_x_test = total_x_test.reshape(len(total_x_test), height, width, 1)

    print(x_test.shape)
    print(y_test.shape)

    print("Rotating images")

    toRemove = []
    for i in range(len(total_x_test)):
        if total_y_test[i][0] >= 36:
            toRemove.append(i)
        total_x_test[i] = rotate(total_x_test[i])

    total_x_test = np.delete(total_x_test,toRemove,0)
    total_y_test = np.delete(total_y_test,toRemove,0)

    total_y_test = np_utils.to_categorical(total_y_test, num_classes)

    return total_x_test, total_y_test


def run_test_data():
    global_vars()
    x_test, y_test = get_data(matfiles)
    print(x_test.shape)
    # load json and create model
    json_file = open('models/alphanumeric_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("models/alphanumeric_model.h5")
    print("Loaded model from disk")

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print('Starting')

    for i in range(len(x_test)):
        img = x_test[i]
        output = model.predict(img.reshape(1,28,28,1))
        prediction = actualValue(output[0],output_map)
        label = actualValue(y_test[i],output_map)
        print(output)
        print("Prediction: " + str(prediction) + "\t Actual: " + str(label))
        temp = img.reshape(28,28)
        temp = temp*255
        temp = temp.astype('uint8')
        cv2.imshow("Prediction",temp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def actualValue(output,output_map):
#    print(output)
    maxindex = 0
    for i in range(len(output)):
        if output[i] > output[maxindex]:
            maxindex = i
    return output_map[maxindex],output[maxindex],output

if __name__ == '__main__':
    run_test_data()
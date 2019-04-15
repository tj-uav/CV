import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Convolution2D
from keras.utils import np_utils
from scipy.io import loadmat
import numpy as np
from random import shuffle
import cv2

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

        x_train = data['train'][0,0]['images'][0,0]
        y_train = data['train'][0,0]['labels'][0,0]
        x_test = data['test'][0,0]['images'][0,0]
        y_test = data['test'][0,0]['labels'][0,0]

        print("Removing validation set")


        val_start = x_train.shape[0] - x_test.shape[0]
        x_val = x_train[val_start:x_train.shape[0],:]
        y_val = y_train[val_start:x_train.shape[0]]
        x_train = x_train[0:val_start,:]
        y_train = y_train[0:val_start]

        diff = num_classes
        if name == "letters":
            diff -= 1

        if diff != 0:
            for i in range(len(y_train)):
                j = y_train[i][0]
                j += diff
                y_train[i] = np.array([j],dtype='uint8')

            for i in range(len(y_test)):
                j = y_test[i][0]
                j += diff
                y_test[i] = np.array([j],dtype='uint8')

        num_classes += nbclasses

        if count == 0:
            total_x_train = x_train
            total_y_train = y_train
            total_x_test = x_test
            total_y_test = y_test
            continue

        total_x_train = np.concatenate((total_x_train,x_train))
        total_y_train = np.concatenate((total_y_train,y_train))
        total_x_test = np.concatenate((total_x_test,x_test))
        total_y_test = np.concatenate((total_y_test,y_test))

    print('Thresholding data sets')
    total_x_train[total_x_train <= thresh] = 0
    total_x_train[total_x_train > thresh] = 1

    total_x_test[total_x_test <= thresh] = 0
    total_x_test[total_x_test > thresh] = 1

    total_x_train = total_x_train.reshape(len(total_x_train), height, width, 1)
    total_x_test = total_x_test.reshape(len(total_x_test), height, width, 1)

    print("Rotating images")

    for i in range(len(total_x_train)):
        total_x_train[i] = rotate(total_x_train[i])

    for i in range(len(total_x_test)):
        total_x_test[i] = rotate(total_x_test[i])

    print("Shuffling data sets")

    indices_train = np.arange(total_x_train.shape[0])
    np.random.shuffle(indices_train)
    total_x_train = total_x_train[indices_train]
    total_y_train = total_y_train[indices_train]

    indices_test = np.arange(total_x_test.shape[0])
    np.random.shuffle(indices_test)
    total_x_test = total_x_test[indices_test]
    total_y_test = total_y_test[indices_test]

    total_y_train = np_utils.to_categorical(total_y_train, num_classes)
    total_y_test = np_utils.to_categorical(total_y_test, num_classes)

    return total_x_train, total_y_train, total_x_test, total_y_test, num_classes


def make_model(num_classes):

    nb_filters = 32 # number of convolutional filters to use
    pool_size = (2, 2) # size of pooling area for max pooling
    kernel_size = (3, 3) # convolution kernel size
    input_shape = (height,width,1)
    model = Sequential()
    model.add(Convolution2D(nb_filters,
                            kernel_size,
                            padding='valid',
                            input_shape=input_shape,
                            activation='relu'))
    model.add(Convolution2D(nb_filters,
                            kernel_size,
                            activation='relu'))

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    print(model.summary())
    return model


x_train, y_train, x_test, y_test, num_classes = get_data(matfiles)
print("Generating model")
print("Classes: " + str(num_classes))
model = make_model(num_classes)
epochs = 2
batch_size = 256

model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

model.summary()
# serialize model to JSON
model_json = model.to_json()
with open("alphanumeric_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("alphanumeric_model.h5")
print("Saved model to disk")

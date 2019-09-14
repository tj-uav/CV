import cv2
import numpy as np
import imutils
from keras.models import Sequential
from keras.layers import MaxPooling2D, Flatten, Dropout, Dense, Convolution2D
from keras.utils import np_utils

height = 40
width = 40
def make_training_data():
    x_train = []
    y_train = []
    for i in range(0,36):
        count = 0
        for j in range(-20,21):
            img = cv2.imread("images/"+str(i)+"/img"+str(count)+".png")
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret,thresh = cv2.threshold(gray,120,5255,cv2.THRESH_BINARY)
            binary = thresh / 255
            binary = binary.astype('uint8')
            x_train.append(binary)
            y_train.append(i)
            count += 1

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = x_train.reshape(len(x_train),40,40,1)
    indices_train = np.arange(x_train.shape[0])
    np.random.shuffle(indices_train)
    x_train = x_train[indices_train]
    y_train = y_train[indices_train]
    x_test = x_train[:int(len(x_train)/15)]
    y_test = y_train[:int(len(y_train)/15)]
    x_train = x_train[int(len(x_train)/15):]
    y_train = y_train[int(len(y_train)/15):]

    y_train = np_utils.to_categorical(y_train, 36)
    y_test = np_utils.to_categorical(y_test, 36)

    return (x_train,y_train,x_test,y_test)

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
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())
    return model


x_train,y_train,x_test,y_test = make_training_data()
print("Generating model")
model = make_model(36)
epochs = 8
batch_size = 100
model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

model.summary()
# serialize model to JSON
model_json = model.to_json()
with open("models/block_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/block_model.h5")
print("Saved model to disk")
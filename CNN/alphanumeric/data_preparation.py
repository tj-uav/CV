from scipy.io import loadmat
import numpy as np
from random import shuffle
import cv2

matfile = loadmat("C:/Users/Srikar/Documents/NN_Databases/EMNIST/matlab/emnist-balanced.mat")

output_map = [str(i) for i in range(10)]
temp = [chr(65+i) for i in range(26)]
output_map.extend(temp)

# Local functions
def rotate(img):
    # Used to rotate images (for some reason they are transposed on read-in)
    flipped = np.fliplr(img)
    return np.rot90(flipped)

height = 28
width = 28

def get_data(matfile):
    thresh = 120
    data = matfile['dataset']

    x_train = data['train'][0,0]['images'][0,0]
    y_train = data['train'][0,0]['labels'][0,0]

    print('Thresholding data sets')
    x_train[x_train <= thresh] = 0
    x_train[x_train > thresh] = 1

    print("Rotating images")

    toRemove = []
    x_train = x_train.reshape(x_train.shape[0],28, 28, 1)
    for i in range(len(x_train)):
        x_train[i] = rotate(x_train[i])
        if y_train[i][0] >= 36:
            toRemove.append(i)
            x_train[i] = rotate(x_train[i])

    print(len(toRemove))

    x_train = np.delete(x_train,toRemove,0)
    y_train = np.delete(y_train,toRemove,0)

    print("Shuffling data sets")

    indices_train = np.arange(x_train.shape[0])
    np.random.shuffle(indices_train)
    x_train = x_train[indices_train]
    y_train = y_train[indices_train]

    # Show first 100 images and their labels
    for i in range(100):
        print(output_map[y_train[i][0]])
        cv2.imshow("Example",x_train[i].reshape(28,28) * 255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

get_data(matfile)
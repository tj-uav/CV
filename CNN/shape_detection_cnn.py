import os
import tensorflow as tf
from tensorflow.python.keras import Model
import numpy as np
import matplotlib.pyplot as plt
import os, sys, time
import cv2

# Add to Python path temporarily
sys.path.insert(1, '/home/jasonc/windows/Jason/UAV/CV/')
img_path = '/home/jasonc/windows/Jason/UAV/CV/DataGen/AlphaShapeData/'

# Hyperparamters
EPOCHS = 1000
LEARNING_RATE = 1e-04
NUM_SAMPLES = 10001
BATCH_SIZE = 64
STEP_SIZE = 10
SAVE_MODEL = True

# https://www.tensorflow.org/tutorials/estimators/cnn
class ConvolutionModel(Model):
    def __init__(self, *args, **kwargs):
        super(ConvolutionModel, self).__init__(name='convolution_model')
        self.input_layer = tf.reshape(5, [-1, 28, 28, 1])  # 5 is the label
        self.conv2d_1 = tf.keras.layers.Conv2D()

"""
image : ndarray
    Input image data. Will be converted to float.
mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 255.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
"""


def generate_noised_img(img):
    img = cv2.imread(img_path + 'circle_5.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(threshold, (5, 5), 1)
    noise = noisy('s&p', blur)
    noise = np.expand_dims(noise, axis=2)
    return noise

def noisy(noise_type, image):
    if noise_type == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy

    elif noise_type == "s&p":
        row, col = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 255
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out

    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    elif noise_type =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

    else:
        print('Invalid noise type.')
        return None

def main():
    # Preprocessing
    

    img = cv2.imread(img_path + 'circle_5.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(threshold, (5, 5), 1)
    noise = noisy('s&p', blur)
    noise = np.expand_dims(noise, axis=2)   
    """
    cv2.imshow('img', img)
    cv2.imshow('gray', gray)
    cv2.imshow('threshold', threshold)
    cv2.imshow('blur', blur)
    cv2.imshow('noise', noise)
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
    """

if __name__ == "__main__":
    main()


import cv2
import numpy as np
import fastai
from fastai.vision import *
from sklearn.cluster import MiniBatchKMeans

display_dim = (3840 // 8, 2160 // 8)

## General Utility functions

def bgr_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def hsv_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

def bgr_to_lab(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

def resize(img, dim):
    return cv2.resize(img, dim)

def crop_img(image, bounds, scales=[1,1]):
    minX, minY, maxX, maxY = bounds
    scaleX, scaleY = scales
    scaledMinY = int(minY * scaleY)
    scaledMaxY = int(maxY * scaleY)
    scaledMinX = int(minX * scaleX)
    scaledMaxX = int(maxX * scaleX)
    crop = image[scaledMinY: scaledMaxY, scaledMinX: scaledMaxX]
#    display("Image", image)
#    display("Crop", crop)
    cv2.waitKey(0)
    return crop

## Preprocessing functions

# Image should be BGR
def bilateral(image, thresh, sigma):
    return cv2.bilateralFilter(image, thresh, sigma, sigma)

# Image should be BGR
def threshold(image, threshold_value, is_gray = False):
    if not is_gray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY) #Figure out appropriate threshold on image-to-image basis
    return thresh

#Sets the brightness component of the hsv to some average value
def remove_shadows(img):
    hsv = bgr_to_hsv(img)
    avg_brightness = 0
    for a in hsv:
        for b in a:
            assert(len(b) == 3)
            avg_brightness += b[2]
    avg_brightness = avg_brightness // (hsv.shape[0] * hsv.shape[1])
    print("Average brightness", avg_brightness)
    for a in hsv:
        for b in a:
            b[2] = avg_brightness
    return hsv_to_bgr(hsv)

#Sets the saturation component of the hsv to some average value
def remove_saturation(img):
    hsv = bgr_to_hsv(img)
    avg_saturation = 0
    for a in hsv:
        for b in a:
            avg_saturation += b[1]
    avg_saturation = avg_saturation // (hsv.shape[0] * hsv.shape[1])
    print("Average saturation", avg_saturation)
    for a in hsv:
        for b in a:
            assert(len(b) == 3)
            b[1] = avg_saturation
    return hsv_to_bgr(hsv)

## Detection functions

# Image should be thresholded
def get_contours(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours found:", len(contours))
    return contours

def blob_detection(image, is_gray = False):
    thresh = threshold(image, 150, is_gray)
    thresh = cv2.bitwise_not(thresh)
    display("Thresh", thresh)
    thresh = (255 - thresh)
    detector = cv2.SimpleBlobDetector_create()
    keypoints = detector.detect(thresh)
    return keypoints

#https://www.pyimagesearch.com/2014/07/07/color-quantization-opencv-using-k-means-clustering/
def kmeans(image, dim, n_clusters):
    h,w = dim
    image = image.reshape((image.shape[0] * image.shape[1]), 3)
    clt = MiniBatchKMeans(n_clusters=n_clusters)
    labels = clt.fit_predict(image)
    print(len(labels))
    print(max(labels))
    quant = clt.cluster_centers_.astype("uint8")[labels]

    quant = quant.reshape((h,w,3))
    image = image.reshape((h,w,3))

    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    
    return labels, quant

def labels_to_dict(labels):
    unique, counts = np.unique(labels, return_counts=True)
    occurs = dict(zip(unique, counts))
    occurs = sorted(occurs.items(), key= lambda kv: kv[1])
    map = {}
    i = 0
    for key in occurs:
        map[key[0]] = i
        i += 1
    return occurs, map

def quantize_img(dim, labels, n_clusters):
    img = np.zeros(dim)
    h,w = dim
    scale = 255 // n_clusters

    occurs, map = labels_to_dict(labels)
    for i in range(len(labels)):
        img[i // w, i % w] = map[labels[i]] * scale

    return img

def image_threshs(dim, labels, n_clusters):
    h,w = dim
    scale = 255 // n_clusters
    threshs = []
    occurs, map = labels_to_dict(labels)
    # Only look at the 4 least occuring clusters
    for j in range(5):
        img = np.zeros(dim)
        for i in range(len(labels)):
            img[i // w, i % w] = 255 if map[labels[i]] == j else 0
        threshs.append(img.copy())
    display_dim = (3840 // 8, 2160 // 8)
    for i in range(len(threshs)):
        display("Thresh " + str(i), threshs[i])

## Display functions

# Beware: this edits the image that is sent as an argument.
def draw_contours(image, contours):
     for cnt in contours:
          approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
          cv2.drawContours(image, [approx], 0, (0), 5)
     return image

# Dim is the width/height of the image
def contour_bounding(cnt, dim, buffer=0):
    x,y,w,h = cv2.boundingRect(cnt)
    dimX, dimY = dim
    x1 = max(0, x-buffer)
    y1 = max(0, y-buffer)
    x2 = min(dimX, x+buffer)
    y2 = min(dimY, y+buffer)
    return x1, y1, x2, y2

def draw_keypoints(image, keypoints, color=(0,0,255)):
    copy = image.copy()
    for point in keypoints:
        cv2.circle(copy, (int(point.pt[0]), int(point.pt[1])), 30, (255,0,0), 5)
    return copy

#    return cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def display(name, img, dim=display_dim):
    cv2.imshow(name, resize(img, dim))

## ML Functions

def load_model(config):
    return load_learner(config['learnerLocation'])

def predict(model, imgloc):
    img = open_image(imgloc).resize(256)
    cat, _, _ = model.predict(img)
    return cat

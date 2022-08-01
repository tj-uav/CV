from turtle import back
import cv2 as cv2
import numpy as np
import os
from PIL import Image, ImageDraw
from sklearn.cluster import MiniBatchKMeans
import random
import imutils
from torch import full

def kMeansQuantize2( image, n ):                     # Uses k-Means to quantize an image into n colors.
    image = cv2.cvtColor( image, cv2.COLOR_BGR2LAB )# The LAB color space is especially useful for this type of system, as it's similar to how a human eye works
    w, h, _ = image.shape                           # Gets the shape of the image for the next step
    image = image.reshape( ( image.shape[ 0 ] * image.shape[ 1 ], 3 ) ) #k-Means required an image to be one pixel tall for some reason
    clt = MiniBatchKMeans( n_clusters = n )         # Creates a new kMeans system. Cluster count tells how many regions of best fit should be made.
    labels = clt.fit_predict( image )               # This takes a aet of inputted colors and tries to find clusters of data. Similar to how one might move a '3d cursor' to the center of data on -a 3d mqp
    quant = clt.cluster_centers_.astype( "uint8" )[ labels ]    #Replaces every pixel with the closewt color k-Means found
    quant = quant.reshape( ( h, w, 3 ) )            # Rescales the quantized image back into its original shape. Simply imagine the pixels line wrqpping similar to a wore processor
    quant = cv2.cvtColor( quant, cv2.COLOR_LAB2BGR )# Reverts the color as well
    #for i in labels: print(labels)
    return labels, quant


# def kMeansQuantize( image, n , labels):                     # Uses k-Means to quantize an image into n colors.
#     image = cv2.cvtColor( image, cv2.COLOR_BGR2LAB )# The LAB color space is especially useful for this type of system, as it's similar to how a human eye works
#     w, h, _ = image.shape                           # Gets the shape of the image for the next step
#     image = image.reshape( ( image.shape[ 0 ] * image.shape[ 1 ], 3 ) ) #k-Means required an image to be one pixel tall for some reason
#     clt = MiniBatchKMeans( n_clusters = n )         # Creates a new kMeans system. Cluster count tells how many regions of best fit should be made.
#     #labels = clt.fit_predict( image )               # This takes a aet of inputted colors and tries to find clusters of data. Similar to how one might move a '3d cursor' to the center of data on -a 3d mqp
#     quant = clt.cluster_centers_.astype( "uint8" )[ labels ]    #Replaces every pixel with the closewt color k-Means found
#     quant = quant.reshape( ( h, w, 3 ) )            # Rescales the quantized image back into its original shape. Simply imagine the pixels line wrqpping similar to a wore processor
#     quant = cv2.cvtColor( quant, cv2.COLOR_LAB2BGR )# Reverts the color as well
#     #for i in labels: print(labels)
#     return labels, quant

def blurImage(img):
    kernel_size = 10
    kernel_v = np.zeros((kernel_size, kernel_size))
    kernel_h = np.copy(kernel_v)
    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    
    kernel_v /= kernel_size
    kernel_h /= kernel_size
    
    vertical_mb = cv2.filter2D(img, -1, kernel_v)
    horizonal_mb = cv2.filter2D(vertical_mb, -1, kernel_h)
    return horizonal_mb

def getFinImage(s_img, l_img, sShape1, sShape2, lShape1, lShape2):
    s_img = cv2.resize( s_img, ( sShape1, sShape2 ) )           #Resizes as final size
    l_img = cv2.resize( l_img, ( lShape1, lShape2 ) )
    
    y_offset = random.randrange(0, l_img.shape[0] - s_img.shape[0])
    x_offset = random.randrange(0, l_img.shape[1] - s_img.shape[1])
    y1, y2 = y_offset, y_offset + s_img.shape[0]
    x1, x2 = x_offset, x_offset + s_img.shape[1]

    for i in range(y1, y2):
        for j in range(x1, x2):
            if not(s_img[i-y1, j-x1][0] == 0 and s_img[i-y1, j-x1][1] == 0 and s_img[i-y1, j-x1][2] == 0):
                l_img[i,j] = s_img[i-y1, j-x1]
    return l_img, 500 - y_offset, x_offset, x1, x2, y1, y2


def getTarget(shap, shapeSize, lett, lettSize):
    lettsize = lett.shape
    shapsize = shap.shape

    border = abs((lettSize - shapeSize))//2

    lett = cv2.resize( lett, ( lettSize, lettSize ) )           #Resizes as final size
    shap = cv2.resize( shap, ( shapeSize, shapeSize ) )
    #lettmask = cv2.inRange( lett, np.array( [ 1, 1, 1 ] ), np.array( [ 255, 255, 255 ] ) )  #Cleans images, converts to mask
    #lett = cv2.cvtColor( lettmask, cv2.COLOR_GRAY2BGR ) #Reconvert clean mask to letter
    #shap = cv2.cvtColor( shapmask, cv2.COLOR_GRAY2BGR )
    
    #lett[ np.where( ( lett == [ 255, 255, 255 ] ).all( axis = 2 ) ) ] = lettColor #Colorize
    #shap[ np.where( ( shap == [ 255, 255, 255 ] ).all( axis = 2 ) ) ] = shapColor
    
    lett = cv2.copyMakeBorder( lett, top = border, bottom = border, left = border, right = border, borderType = cv2.BORDER_CONSTANT, value= [ 0, 0, 0 ] )   #Rescale letter for copy
    #lettmask = cv2.copyMakeBorder( lettmask, top = border, bottom = border, left = border, right = border, borderType = cv2.BORDER_CONSTANT, value= [ 0 ] )

    #lettmask_inv = cv2.bitwise_not( lettmask )                  #Clear the area
    shap = cv2.add( shap, lett )                              #Add colors
    return shap

def backgroundRemove(full_road):
    full_road = cv2.resize(full_road, (512, 512))
    fr = cv2.resize(full_road, (200, 200))
    fr = kMeansQuantize2(fr, 4)
    
    distinct = set()
    for i in range(0, len(fr)):
        for j in range(0, len(fr[i])):
            distinct.add((fr[i][j][0],fr[i][j][1],fr[i][j][2]))

    #fr = cv2.resize(fr, (512, 512))
    # newimg = full_road
    # for i in range(0, len(newimg)):
    #     for j in range(0, len(newimg[i])):
    #         if 10<30:#math.sqrt(dist(newimg[i][j], distinct)) < 30:
    #             newimg[i][j] = [0, 0, 0]
    num = 50
    oldfr = full_road.copy()
    for i in distinct:
        mask = cv2.inRange(full_road, np.array([i[0]- num,i[1]- num,i[2]-num]), np.array([i[0]+ num,i[1]+ num,i[2]+num]))
        #masking the image
        newmask = cv2.bitwise_not(mask)
        output = cv2.bitwise_and(full_road, full_road, mask = newmask)
        full_road = output   

    return cv2.resize(full_road, (512, 512))

count = 0

backs = []
genIm = []
for ifile in os.listdir("Backgrounds/"):
    backs.append(ifile)
for jfile in os.listdir("GenImages/"):
    genIm.append(jfile)

#for ix in os.listdir("Backgrounds/"):
    #for jx in os.listdir("GenImages/"):
filenamesPNG = set()
shap = 0
b = 0
file1 = open("vals2.txt", "w")
while count <= 100:
    if shap >= len(genIm): shap = 0
    if b >= len(backs): b = 0
    
    jx = genIm[shap]#random.randint(0, len(genIm) - 1)]
    ix = backs[b]#random.randint(0, len(backs) - 1)]
    b += 1
    shap += 1
    letter = cv2.imread("GenImages/" + jx)
    shape = cv2.imread("Backgrounds/" + ix)
    sizeL1 = random.randint(18, 18)
    sizeL2 = random.randint(18, 18) 

    sizeS = random.randint(512, 512)
    if sizeS % 2 != 0: sizeS += 1
    if sizeL1 % 2 != 0: sizeL1 += 1
    if sizeL2 % 2 != 0: sizeL2 += 1
    

    target, y_offset, x_offset, x1, x2, y1, y2 = getFinImage(letter, shape, sizeL1, sizeL2, 3840, 2160)
    target = blurImage(target)

    filename = "FinalGeneratedImages2/" + ix.replace(".png", "").replace(".jpg", "") + "_" + jx.replace(".png", "").replace(".jpg", "") + ".png"

    if filename in filenamesPNG:
        filename = filename.replace("png", "") + "2.png"
    filenamesPNG.add(filename)
    #sx = sp[0] + "," + sp[1] + "," + sp[2] + "," + str(x_offset) + "," + str(y_offset + sizeL) + "," + str(x_offset + sizeL) + "," + str(y_offset)
    sx = str(x1) + "," + str((y1)) + "," + str(x2) + "," + str((y2)) + ",0"
    finStr = filename + " " + sx
    
    file1.write(finStr + "\n")

    print(count, finStr)
    print(x1, x2, y1, y2)
    print()
    
    full_road = cv2.resize(target, (3840, 2160))
    #cv2.imshow("fff", cv2.resize(full_road, (512, 512)))
    fr = cv2.resize(full_road, (200, 200))
    if True:
        labels, fr = kMeansQuantize2(fr, 4)
    else:
        fr = kMeansQuantize(fr, 4, labels)
    
    distinct = set()
    for i in range(0, len(fr)):
        for j in range(0, len(fr[i])):
            distinct.add((fr[i][j][0],fr[i][j][1],fr[i][j][2]))

    cv2.imwrite(filename, full_road)

    count += 1
#file1.close()
print(count)
print(len(filenamesPNG))

import cv2 
import numpy as np 
import os 
import glob 
  
#filenames is a list of the filenames of images of checkerboards. Returns the image distortion values  
def getDistortionVals(filenames):
    CHECKERBOARD = (3,3) 
    
    criteria = (cv2.TERM_CRITERIA_EPS + 
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
    
    
    threedpoints = [] 
    
    twodpoints = [] 
    
    
    objectp3d = np.zeros((1, CHECKERBOARD[0]  
                        * CHECKERBOARD[1],  
                        3), np.float32) 
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 
                                0:CHECKERBOARD[1]].T.reshape(-1, 2) 
    #print(objectp3d)
    prev_img_shape = None
    

    
    for filename in filenames: 
        image = cv2.imread(filename) 
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

        ret, corners = cv2.findChessboardCorners( 
                        grayColor, CHECKERBOARD,  
                        cv2.CALIB_CB_ADAPTIVE_THRESH  
                        + cv2.CALIB_CB_FAST_CHECK + 
                        cv2.CALIB_CB_NORMALIZE_IMAGE) 

        if ret == True: 
            threedpoints.append(objectp3d) 

            corners2 = cv2.cornerSubPix( 
                grayColor, corners, (11, 11), (-1, -1), criteria) 
    
            twodpoints.append(corners2) 
    
            image = cv2.drawChessboardCorners(image,  
                                            CHECKERBOARD,  
                                            corners2, ret) 
        image = cv2.resize(image, (500, 500))
        cv2.imshow(filename, image) 
        cv2.waitKey(500) 
    
    cv2.destroyAllWindows() 
    
    h, w = image.shape[:2] 
    

    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera( 
        threedpoints, twodpoints, grayColor.shape[::-1], None, None) 
    
    print(" Camera matrix:") 
    print(matrix) 
    
    print("\n Distortion coefficient:") 
    print(distortion) 
    
    print("\n Rotation Vectors:") 
    print(r_vecs) 
    
    print("\n Translation Vectors:") 
    print(t_vecs) 
    return matrix, distortion, r_vecs, t_vecs

def adjustBrightness(filename1, filename2):
    image1 = cv2.imread(filename1)
    image2 = cv2.imread(filename2)
    image1 = cv2.resize(image1, (500, 500))
    image2 = cv2.resize(image2, (500, 500))
    hsv1 = cv2.cvtColor(image1,cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(image2,cv2.COLOR_BGR2HSV)
    sum1 = 0
    sum2 = 0
    for z in hsv1[...,2]:
        for i in z:
            sum1 = sum1 + i 
     
    for z in hsv2[...,2]:
        for i in z:
            sum2 = sum2 + i
    
    hsv2[...,2] = hsv2[...,2] * (sum1/sum2)
    return cv2.cvtColor(hsv2,cv2.COLOR_HSV2RGB)

image = adjustBrightness('20210110_115451.jpg', '20210110_115443.jpg')
cv2.imshow('f', image)
cv2.waitKey(5000)
#matrix, distortion, r_vecs, t_vecs = getDistortionVals(['20210110_115443.jpg'])
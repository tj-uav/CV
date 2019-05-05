import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
vidNum = 0

while(cap.isOpened()):

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    currOut = cv2.VideoWriter('output' + str(vidNum) + '.avi',fourcc, 20.0, (640,480))
    startTime = time.time()
    print(vidNum)

    while time.time() - startTime < 5:
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.flip(frame,0)
            frame = cv2.flip(frame, 1)

            # write the flipped frame
            currOut.write(frame)
        else:
            break
    
    vidNum += 1
    currOut.release()

# Release everything if job is finished
cap.release()
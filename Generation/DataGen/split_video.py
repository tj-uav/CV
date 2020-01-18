import cv2
import os

videoPath = "C:/Users/Srikar/Downloads/DJI_0167.MP4"
imgPath = "C:/Users/Srikar/Documents/UAV/Data/Images/DJI_0167/image"
imgEnding = ".png"

save_rate = 20

cap = cv2.VideoCapture(videoPath)
count = 0
while cap.isOpened():
    if count % 100 == 0:
        print(count)
    ret, frame = cap.read()
    if not ret: continue
    count += 1
    if count % save_rate == 0:
        cv2.imwrite(imgPath + str(count // save_rate) + imgEnding, frame)

print(count)
cap.release()
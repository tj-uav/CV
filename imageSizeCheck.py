import cv2

image = cv2.imread("C:/Users/Ron/UAV/GCS/ManualClassification/assets/img/Target17.png")
print(image.shape)
image = cv2.resize(image, (16, 16))
cv2.imshow("image", image)
cv2.waitKey(0)
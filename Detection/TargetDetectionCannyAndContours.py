import cv2, numpy as np, time, os
from matplotlib import pyplot as plt
import fastai
from fastai.vision import *

start = time.time()
directoryString = "C:/Users/Ron/Desktop/Files/UAV Fly Pics/Flight2/"
imageName = "Frame78"
imageExtension = ".jpg"
maxContours = 100
minContours = 2
fileSaveString = "C:/Users/Ron/UAV/GCS/ManualClassification/assets/img/testTargets"

image = cv2.imread(directoryString + imageName + imageExtension)
og = image.copy()

canny = cv2.Canny(image, 200, 250)
contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
i =0
while i < len(contours):
   if len(contours[i]) < 0:
      contours.pop(i)
      i -= 1
   i += 1

cv2.imshow("Test Contours", cv2.drawContours(cv2.cvtColor(np.zeros((700, 1200, 1), dtype = "uint8"), cv2.COLOR_GRAY2BGR), contours, -1, (255, 255, 255), 1))
cv2.waitKey(0)


print("Preprocessing")
originalY, originalX, _ = og.shape
currY, currX = 700, 1200
image = cv2.resize(image, (1200, 700))
imageX = image.copy()
image2 = image.copy()
ogResized = image.copy()
# cv2.imshow("Original Image (M2)", cv2.resize(image, (1200, 700)))

print("Thresholding + Contours")
iter = 1
num = 30
while True:
   blurred = cv2.bilateralFilter(image, 30, num, num)
   canny = cv2.Canny(blurred, 200, 255)
   # cv2.imshow("Canny", canny)
   # cv2.imshow("Blurred", blurred)
   image2 = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
   # cv2.imshow("Grayscale", image2)
   _, threshold = cv2.threshold(image2, 200, 255, cv2.THRESH_BINARY) #Figure out appropriate threshold on image-to-image basis
   contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   print("Number of contours found:", len(contours))
   print("Iteration:", iter, "\nValue of n:", num)

   if 2 <= len(contours) <= maxContours:
      break
   elif len(contours) > maxContours:
      num += 10
      iter += 1
   elif len(contours) < minContours:
      if num < 10:
         break
      num -= 10
      iter += 1

contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contourDict = {i:contours[i] for i in range(len(contours))}

contourImage = np.zeros((700, 1200, 1), dtype = "uint8")

for cnt in contourDict:
    cnt = contourDict[cnt]
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    cv2.drawContours(imageX, [approx], 0, 0, 1)
    x = approx.ravel()[0]
    y = approx.ravel()[1]
contourImage = cv2.drawContours(cv2.cvtColor(contourImage, cv2.COLOR_GRAY2BGR), contours, -1, (255, 255, 255), 1)
cv2.imshow("Image with contours", imageX)
cv2.imshow("Canny", canny)
cv2.imshow("Contours", contourImage)
cv2.imshow("Combo", cv2.bitwise_and(cv2.cvtColor(contourImage, cv2.COLOR_BGR2GRAY), canny))
cv2.waitKey(0)

os.chdir(fileSaveString)
# cv2.imwrite("1list.jpg", image)
for i in range(0, len(contours)):
   windowName = "Contour " + str(i)
   minX, minY = 10000000, 10000000
   maxX, maxY = 0, 0
   for point in contours[i]:
      point = point[0]
      if point[0] < minX:
         minX = point[0]
      if point[1] < minY:
         minY = point[1]
      if point[0] > maxX:
         maxX = point[0]
      if point[1] > maxY:
         maxY = point[1]
   if minX < 10:
      minX = 0
   else:
      minX -= 10
   if minY < 10:
      minY = 0
   else:
      minY -= 10
   if maxX > len(image[0]) - 10:
      maxX = len(image[0])
   else:
      maxX += 10
   if maxY > len(image) - 10:
      maxY = len(image)
   else:
      maxY += 10
   scaledMinY = int(minY * originalY / currY)
   scaledMaxY = int(maxY * originalY / currY)
   scaledMinX = int(minX * originalX / currX)
   scaledMaxX = int(maxX * originalX / currX)
   crop = og[scaledMinY: scaledMaxY, scaledMinX: scaledMaxX]
   # crop = cv2.resize(crop, (len(crop[0])*10, len(crop)*10))
   crop = cv2.resize(crop, (50, 50))

   fileName = "Target" + str(i) + ".png"
   cv2.imwrite(fileName, crop)
print("Target Detection and Save Time:", time.time() - start)

classTime = time.time()
learn = load_learner('C:/Users/Ron/Downloads/')
count = 0
for pic in os.listdir("C:/Users/Ron/UAV/GCS/ManualClassification/assets/img/testTargets"):
   picture = "C:/Users/Ron/UAV/GCS/ManualClassification/assets/img/testTargets/" + pic
   # cv2.imshow("pic", cv2.imread(pic))
   img = open_image(picture).resize(256)
   cat, _, _ = learn.predict(img)
   # print(str(cat))
   if str(cat) == "bad":
      os.remove(picture)
      contourDict.pop(int(pic[6:pic.find(".")]))
   # print(learn.predict(img))
   cv2.waitKey(0)
print("Classification Time:", time.time() - classTime)
for cnt in contourDict:
    cnt = contourDict[cnt]
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    cv2.drawContours(image, [approx], 0, (0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1]
cv2.imshow("Contours Before and After", np.concatenate((cv2.resize(imageX, (600, 350)), cv2.resize(image, (600, 350)))))
cv2.imshow("Combo", np.concatenate((canny, )))
cv2.waitKey(0)
# cv2.imshow("Drawn Contours (M2)", image)
# cv2.imshow("Final Image (M2)", threshold)
# cv2.imwrite(imageName + "Concat.jpg", np.concatenate((np.concatenate((ogResized, blurred)), np.concatenate((image, cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR))))))
# cv2.waitKey(0)
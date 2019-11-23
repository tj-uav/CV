import cv2, numpy as np, time, os
import json
from matplotlib import pyplot as plt
from detection_methods import *

config = json.load(open("../config.json"))
start = time.time()
directoryString = config["directoryString"]
imageName = "1560599871.22"
imageExtension = ".jpg"
maxContours = 100
minContours = 2
fileSaveString = config["fileSaveString"]

image = cv2.imread(directoryString + imageName + imageExtension)
print(image.shape)
og = image.copy()

process_dim = (1200,700)
display_dim = (600,350)

print("Preprocessing")
originalY, originalX, _ = og.shape
image = resize(image, process_dim)
currY, currX, _ = image.shape
imageX = image.copy()
image2 = image.copy()
ogResized = image.copy()
# cv2.imshow("Original Image (M2)", cv2.resize(image, (1200, 700)))
#cv2.imshow("Image", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


print("Thresholding + Contours")
iter = 1
num = 30
while True:
   blurred = bilateral(image, 30, num)
   # cv2.imshow("Blurred", blurred)
   thresh = threshold(blurred, 200)
   contours = get_contours(thresh)
   print("Iteration:"
   , iter, "\nValue of n:", num)

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

contourDict = {i:contours[i] for i in range(len(contours))}

draw_contours(imageX, contours)
cv2.imshow("Contours", imageX)
cv2.waitKey(0)

"""
# cv2.imwrite("1list.jpg", image)
maxX = 0
for i,cnt in enumerate(contours):
   bounds = contour_bounding(cnt, process_dim, buffer=10)
   crop = crop_img(og, bounds, [originalX / currX, originalY / currY])
   crop = resize(crop, (50, 50))

   fileName = "Target" + str(i) + ".png"
   cv2.imwrite(fileSaveString + fileName, crop)

print(time.time() - start)

start = time.time()
learn = load_learner(config['learnerLocation'])
for pic in os.listdir(fileSaveString):
   picture = fileSaveString + "/" + pic
   cat = predict(learn, picture)
   if str(cat) == "bad":
      os.remove(picture)
      contourDict.pop(int(pic[6:pic.find(".")]))

print("Classification Time:", time.time() - start)

contours = [contourDict[k] for k in contourDict]
image = draw_contours(image, contours)
imageXRe = resize(imageX, display_dim)
imageRe = resize(image, display_dim)
cv2.imshow("Contours Before and After", np.concatenate((imageXRe, imageRe)))
cv2.waitKey(0)
"""
# cv2.imshow("Drawn Contours (M2)", image)
# cv2.imshow("Final Image (M2)", threshold)
# cv2.imwrite(imageName + "Concat.jpg", np.concatenate((np.concatenate((ogResized, blurred)), np.concatenate((image, cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR))))))
# cv2.waitKey(0)
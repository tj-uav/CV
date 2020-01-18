import cv2
import numpy as np

img = cv2.imread("C:/Users/Ron/Desktop/Files/UAV Fly Pics/Flight2/Frame91.jpg")
og = img.copy()
pixels = np.float32(cv2.resize(img, (600 // 5, 350 // 5)).reshape(-1, 3))

num_contours = 1000000
n_colors = 25
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
flags = cv2.KMEANS_RANDOM_CENTERS

maxContours = 100
minContours = 2
minimumContourPoints = 3

_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
_, counts = np.unique(labels, return_counts=True)
dominant = palette[np.argmax(counts)]
average = img.mean(axis=0).mean(axis=0)
counts = [(counts[i],i) for i in range(len(counts))]
counts = sorted(counts, reverse=True)
print(counts)
print("Dominants")
patches = []
for val,ind in counts:
    print(palette[ind])
    patches.append(palette[ind])

for color in patches:
    patch = np.ones(shape=img.shape, dtype=np.uint8)*np.uint8(color)
    lowerMask = np.array(patch - 100)
    upperMask = np.array(patch + 100)
    mask = cv2.inRange(img, lowerMask, upperMask)
    mask = 255 - mask
    img = cv2.bitwise_and(img, img, mask=mask)

kernel = np.ones((2,2),np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
kernel = np.ones((2,2),np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Image", cv2.resize(img, (600, 350)))
res = img
threshold = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours found:", len(contours))
count = 0
newContours = []
for cnt in contours:
    if len(cnt) >= minimumContourPoints:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        cv2.drawContours(og, [approx], 0, (0), 5)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        count += 1
        newContours.append(cnt)
# cv2.imwrite("1list.jpg", image)
print("Number of Contours Drawn and Saved:", count)
cv2.imshow("Drawn Contours (M2)", cv2.resize(og, (600,350)))
cv2.waitKey(0)
cv2.destroyAllWindows()
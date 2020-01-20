import cv2
from detection_methods import *

# BLob detection method
def method1(img):
    display("Image", img)
    blurred = bilateral(img, 30)
    keypoints = blob_detection(blurred)
    print(len(keypoints))
#    points = [point.pt for point in keypoints]
#    for point in points:
#        cv2.circle(img, (int(point[0]), int(point[1])), 50, (0,0,0), 5)
    drawn = draw_keypoints(img, keypoints)
    display("Drawn", img)

# Threshold and contour detection method
def method2(image):
    og = image.copy()
    originalY, originalX, _ = og.shape
    currY, currX = 700, 1200
    process_dim = 1200, 700
    image = resize(image, process_dim)
    display("Original Image (M2)", image)
    iter = 1
    num = 30
    while True:
        blurred = bilateral(image, num)
        display("Blurred", blurred)
        thresh = threshold(blurred, 200)
        contours = get_contours(thresh)
        print("Iteration:", iter, "\nValue of n:", num)
        if 2 <= len(contours) <= 100:
            break
        elif len(contours) > 100:
            num += 10
            iter += 1
        elif len(contours) < 2:
            if num < 10:
                break
            num -= 10
            iter += 1

    image = draw_contours(image, contours)
    display("Contours", image)
#    cv2.imshow("Contours", resize(image, display_dim))
#    for i in range(0, len(contours)):
#        windowName = "Contour " + str(i)
#        cnt = contours[i]
#        minX, minY, maxX, maxY = contour_bounding(cnt, image.shape, 10)
#        crop = crop_img(og, (minX, minY, maxX, maxY), scales=[originalX / currX, originalY / currY])
#        crop = resize(crop, (len(crop[0])*10, len(crop)*10)
#        cv2.imshow("Target " + str(i), crop)

# Quantization method
def method3(image):
    num_clusters = 20

    blurred = bilateral(image, 30)
    shadowless = remove_shadows(remove_saturation(blurred))
    display("Shadowless", shadowless)

    labels, quant = kmeans(shadowless, image.shape[:2], num_clusters)

    splitup = quantize_img(image.shape[:2], labels, num_clusters)
    display("Image", image)
    display("Blurred", blurred)
    display("Lab", lab)
    display("Quant", quant)
    display("Splitup", splitup)


# HSV method
def method4(image):
    num_clusters = 20

    blurred = bilateral(image, 30)
#    shadowless = remove_shadows(remove_saturation(blurred))
#    display("Shadowless", shadowless)

    hsv = bgr_to_hsv(blurred)
    hsv_in_range = cv2.inRange(hsv, (70,0,0), (255,255,255))
    kernel = np.ones((3,3),np.uint8)
    hsv_in_range = cv2.morphologyEx(hsv_in_range, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((10,10),np.uint8)
    hsv_in_range = cv2.morphologyEx(hsv_in_range, cv2.MORPH_CLOSE, kernel)
    print(hsv.shape)
    # Index of yellow cross of image 22
    print(hsv[330][692])

    contours = get_contours(hsv_in_range)
    contours = list(filter(lambda c: cv2.contourArea(c) < 200, contours))
    draw_contours(image, contours)
    print(len(contours))

    display("Image", image)
    display("Blurred", blurred)
    display("HSV", hsv)
    display("HSV in range", hsv_in_range)

imgPath = "C:/Users/Srikar/Documents/UAV/Data/Images/GP010231/image%s.png"
imgNames = ["2", "6", "16", "22", "33", "34", "35", "39", "48", "49", "50", "51", "52", "53", "63", "64", "65", "71", "83", "84", "85", "93", "102", "103"]
imgNames = ["2"]
display_dim = (3840 // 8, 2160 // 2)
process_dim = (3840 // 4, 2160 // 4)

for imgName in imgNames:
    img = cv2.imread(imgPath % imgName)
    method4(resize(img, process_dim))
    key = cv2.waitKey(0)
    if key == ord("q"):
        break

cv2.destroyAllWindows()

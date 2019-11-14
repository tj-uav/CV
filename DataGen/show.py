cursor = [0,0]
for i in range(250,10):
    img = cv2.imread("Hue " + str(i) + ".png")
    cv2.circle(img, cursor, , (0,0,0), 3)
    cv2.imshow()
    cv2.waitKey(0)

cv2.destroyAllWindows()


import cv2

def border(img, goal_width, goal_height):
    border_left = int((goal_width - img.shape[1])/2)
    border_right = border_left if (goal_width - img.shape[1]) % 2 == 0 else border_left + 1
    if border_left < 0 or border_right < 0:
        border_left -= 2
        border_left *= -1
        border_right *= -1
    border_top = int((goal_height - img.shape[0])/2)
    border_bottom = border_top if (goal_width - img.shape[0]) % 2 == 0 else border_top + 1
    if border_top < 0 or border_bottom < 0:
        border_top -= 2
        border_top *= -1
        border_bottom *= -1
    return cv2.copyMakeBorder(img, top=border_top, bottom=border_bottom, left=border_left, right=border_right, borderType=cv2.BORDER_CONSTANT, value=[0])

img = cv2.imread('Alphas/8.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

new = border(thresh, 200, 100)
print(new.shape)

cv2.imshow("Thresh", thresh)
cv2.imshow("New", new)
cv2.waitKey(1000)
cv2.destroyAllWindows()
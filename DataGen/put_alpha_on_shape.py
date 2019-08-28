import random
import cv2
import numpy as np
import imutils

rand = np.random.randint(2)
SHAPE_OPTIONS = ["circle", "semicircle", "quartercircle", "triangle", "square",
                 "rectangle", "trapezoid", "pentagon", "hexagon", "heptagon", "octagon", "star", "cross"]
#COLOR_OPTIONS = ["Black", "Gray", "White", "Red", "Blue", "Green", "Brown", "Orange", "Yellow", "Purple"]
ALPHA_OPTIONS = [chr(i) for i in range(48, 58)]
ALPHA_OPTIONS.extend([chr(i) for i in range(65, 65+26)])


def thresh(img):
    return cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]

def resize(img, dim):
    resized = cv2.resize(img, dim)
    return thresh(resized)

def border(img, goal_width, goal_height):
    border_right = int((goal_width - img.shape[1])/2)
    border_left = goal_width - img.shape[1] - border_right
    if border_left > border_right and random.random() < .5:
        border_left -= 1
        border_right += 1
    if border_left < 0 or border_right < 0:
        border_left = 0
        border_right = 0
    border_bottom = int((goal_height - img.shape[0])/2)
    border_top = goal_height - img.shape[0] - border_bottom
    if border_top < 0 or border_bottom < 0:
        border_top = 0
        border_bottom = 0
    if border_top > border_bottom and random.random() < .5:
        border_top -= 1
        border_bottom += 1
    return cv2.copyMakeBorder(img, top=border_top, bottom=border_bottom, left=border_left, right=border_right, borderType=cv2.BORDER_CONSTANT, value=[0])


def processes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_img = thresh(gray)
    _, cnts, hierarchy = cv2.findContours(
        thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=lambda c: cv2.contourArea(c))
    x, y, w, h = cv2.boundingRect(cnt)
    return thresh_img[y:y+h, x:x+w]


def makeSmaller(img, amount):
    original = img.shape
    img = imutils.resize(img, width=img.shape[1]-amount)
    img = thresh(img)
    if amount >= 0:
        img = border(img, original[1], original[0])
    else:
        h, w = img.shape
        dX, dY = int((w-original[1])/2), int((h-original[0])/2)
        img = img[dY:dY+original[0], dX:dX+original[1]]
    return img


def moveFromCenter(thresh, center):
    temp = thresh.copy()
    if center:
        dX, dY = int(thresh.shape[1]/2) - \
            center[0], int(thresh.shape[0]/2) - center[1]
    else:
        return thresh
    bL, bR, bT, bB = 0, 0, 0, 0
    if int(dX/2) >= 0:
        temp = temp[:, int(dX/2):]
        bL = int(dX/2)
    else:
        temp = temp[:, :int(dX/2)]
        bR = int(-dX/2)
    if int(dY/2) >= 0:
        temp = temp[int(dY/2):, :]
        bT = int(dY/2)
    else:
        temp = temp[:int(dY/2), :]
        bB = int(-dY/2)
    temp = cv2.copyMakeBorder(temp, top=bB, bottom=bT, left=bR,
                              right=bL, borderType=cv2.BORDER_CONSTANT, value=[0])
    return temp


def stack(alpha, shape, angle):
    alpha_img = cv2.imread('Alphas/' + alpha + '.png')
    alpha_thresh = processes(alpha_img)
    shape_img = cv2.imread('Shapes/' + shape + '.png')
    shape_thresh = processes(shape_img)
    if alpha_thresh.shape[1] > alpha_thresh.shape[0]:
        alpha_thresh = imutils.resize(
            alpha_thresh, width=shape_thresh.shape[1])
    else:
        alpha_thresh = imutils.resize(
            alpha_thresh, height=shape_thresh.shape[0])
    alpha_thresh = border(
        alpha_thresh, shape_thresh.shape[1], shape_thresh.shape[0])
    alpha_thresh = resize(
        alpha_thresh, (shape_thresh.shape[1], shape_thresh.shape[0]))

#    cv2.imshow("Shape", shape_img)
#    cv2.imshow("Alpha", alpha_img)

    alpha = alpha_thresh
    shape = shape_thresh
    alpha = border(alpha, shape.shape[1], shape.shape[0])
    shape = border(shape, alpha.shape[1], alpha.shape[0])
    alpha, shape, ok = alpha_on_shape(alpha, shape, None)
#    ok = True
    while ok:
        shape = makeSmaller(shape, 10)
        alpha, shape, ok = alpha_on_shape(alpha, shape, None)

    cX, cY = centroid(shape_img, shape)

    while not ok:
        alpha = makeSmaller(alpha, 10)
        alpha, shape, ok = alpha_on_shape(alpha, shape, (cX, cY))

    alpha = makeSmaller(alpha, 45)
    stacked = stack_img(shape, alpha, (cX, cY))
#    cv2.imshow("Stacked", stacked)
    rotated = imutils.rotate_bound(stacked, angle)
#    cv2.imshow("Rotated", rotated)
    return rotated

def alpha_on_shape(alpha, shape, center):
    return alpha, shape, checkOk(alpha, shape, center)


def checkOk(alpha, shape, center):
    temp = moveFromCenter(alpha, center)
    for index, a in np.ndenumerate(temp):
        b = shape[index]
        if a == 255 and b == 0:
            return False
    return True


def stack_img(shape, alpha, center):
    background = np.zeros((shape.shape[0], shape.shape[1], 3), np.uint8)
    background[:] = (0, 255, 0)
    background = cv2.bitwise_and(background, background, mask=shape)
    overlay = np.zeros((alpha.shape[0], alpha.shape[1], 3), np.uint8)
    overlay[:] = (255, 255, 255)
    temp = moveFromCenter(alpha, center)
    overlay = cv2.bitwise_and(overlay, overlay, mask=temp)
    stacked = cv2.bitwise_or(background, overlay)
    return stacked
#    cv2.imshow("Stacked", stacked)
#    cv2.waitKey(1500)
#    cv2.destroyAllWindows()


def centroid(img, thresh):
    M = cv2.moments(thresh)
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # put text and highlight the center
    cv2.circle(img, (cX, cY), 5, (255, 0, 0), -1)
    cv2.putText(img, "centroid", (cX - 25, cY - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # display the image
#    cv2.imshow("Image", img)
#    cv2.waitKey(0)
    return (cX, cY)

#Generate AlphaShapeData folder
if __name__ == '__main__':
#    for angle in range(0, 360, 10):
    angle = 0
    for shape in SHAPE_OPTIONS:
        print(shape)
        for alpha in ALPHA_OPTIONS:
            #    alpha = random.sample(ALPHA_OPTIONS,1)[0]
            #    shape = random.sample(SHAPE_OPTIONS,1)[0]
            stacked = stack(alpha, shape, angle)
            cv2.imwrite('AlphaShapeData/' + shape +
                        '_' + alpha + '.png', stacked)

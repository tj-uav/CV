import cv2, random
from paste import *

def set_saturation(img, sat):
    hue = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hue[:,:,2] = sat
    return cv2.cvtColor(hue, cv2.COLOR_HLS2BGR)

def set_lighting(img, light):
    mask = np.all(img == [0,0,0], axis=-1)
    hue = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hue[:,:,1] = light
    new_img = cv2.cvtColor(hue, cv2.COLOR_HLS2BGR)
    new_img[mask] = [0,0,0]
    return new_img


def make(target, background, shape_color, alpha_color, size, location):
    alpha = .6
    beta = -30

    target = colorify(target, shape_color, alpha_color)
    target = cv2.cvtColor(target, cv2.COLOR_HLS2BGR)
#    print({(j[0], j[1], j[2]) for i in target for j in i})
#    target = set_saturation(target, sat)
#    target = set_lighting(target, light)
#    target = brightness(target, alpha, beta)
    pasted = paste(target, background, size, location)
    return pasted
#    cv2.imshow("Pasted", cv2.resize(pasted, (pasted.shape[1] // 4, pasted.shape[0] // 4)))
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

def gen():
    for k in range(50):
        background = cv2.imread("Grounds/Frame7.jpg")
        hue = k * 5
        for i in range(13):
            sat = i * 20
            for j in range(13):
                scale = 100 // 13
                light = j * 20
                shape_color = [hue, light, sat]
                target = cv2.imread("Shapes/circle.png")
                background = make(target, background, shape_color, alpha_color, size, (i*scale, j*scale))
        cv2.imwrite("Hue " + str(hue) + ".png", cv2.resize(background, (background.shape[1] // 4, background.shape[0] // 4)))

def crop(img, target_shape, loc, size):
    xmid, ymid, w, h = get_info(img, target_shape, loc, size)
    x1, x2 = xmid - w // 2 - buffer // 2, xmid + w // 2 + buffer // 2
    y1, y2 = ymid - h // 2 - buffer // 2, ymid + h // 2 + buffer // 2
    return img[y1:y2, x1:x2]

def get_info(img, target_shape, loc, size):
    buffer = size * 5
    w, h, _ = target_shape
    shape = [img.shape[0] * size / 100, img.shape[1] * size / 100]
    if shape[0] < shape[1]:
        shape[1] = shape[0] * h / w
    else:
        shape[0] = shape[1] * w / h
    location = (int((img.shape[0] - shape[0]) / 100) * loc[0], int((img.shape[1] - shape[1]) / 100) * loc[1])
#    location = (int((img.shape[0] - target_shape[0]) / 100) * loc[0], int((img.shape[1] - target_shape[1]) / 100) * location[1])
    y1 = location[0]
    y2 = location[0] + int(shape[0])
    x1 = location[1]
    x2 = location[1] + int(shape[1])
    y1 = max(y1, 0)
    y2 = min(y2, img.shape[0])
    x1 = max(x1, 0)
    x2 = min(x2, img.shape[1])
    w = x2 - x1
    h = y2 - y1
    xmid = (x1 + x2) // 2
    ymid = (y1 + y2) // 2

    return xmid, ymid, w, h


#target = cv2.imread("AlphaShapeData/circle_H.png")
#grounds = ['Frame1.jpg', 'Frame2.jpg', 'Frame3.jpg', 'Frame4.jpg', 'Frame5.jpg', 'Frame6.jpg', 'Frame7.jpg', 'Frame8.jpg', 'Frame9.jpg', 'Frame10.jpg', 'Frame11.jpg', 'Frame12.jpg', 'Frame13.jpg', 'Frame14.jpg', 'Frame15.jpg', 'Grass1.jpg', 'Grass2.jpg', 'Grass3.png', 'Grass4.png']
shapes = ['circle', 'cross', 'heptagon', 'hexagon', 'octagon', 'pentagon', 'quartercircle', 'rectangle', 'semicircle', 'square', 'star', 'trapezoid', 'triangle']
grounds = ['Frame1.jpg', 'Frame2.jpg', 'Frame3.jpg', 'Frame4.jpg', 'Frame5.jpg', 'Frame6.jpg', 'Frame7.jpg', 'Frame8.jpg', 'Frame9.jpg', 'Frame10.jpg', 'Frame11.jpg', 'Frame12.jpg', 'Frame13.jpg', 'Frame14.jpg', 'Frame15.jpg']
#Restriction format: (xMin, xMax), (yMin, yMax)
#Frame 6 is all dead grass, Frame 10 is weird, 
restrictions = [[(20,60),(0,60), 1], [(0,25), (0,100), 3], [(25,90), (0,100), 2], [(60,100), (20,100), 2], [(5,75), (0,100), 3], [(0,0),(0,0), 2], [(0,70), (0,60), 2], [(0,90), (0,70), 4], [(0,50), (30,100), 3], [(0,0),(0,0), 2], [(30,60), (10,100), 1], [(0,0),(0,0),1], [(25,90),(0,100),2], [(0,100),(0,100),3], [(15,100), (0,100), 2]]
def cascade_data(num, positive):
    start = 0
    i = 0
    while i < num:
        if i % 10 == 0:
            print(start + i)
        shape_idx = random.randint(0,len(shapes) - 1)
        target = cv2.imread("Shapes/" + shapes[shape_idx] + ".png")
#        print(shapes[shape_idx])
#        target = cv2.imread("Shapes/circle.png")
        target_shape = target.shape
        hue = random.randint(0,255)
        light = random.randint(100,255)
        sat = random.randint(light, 255)
        shape_color = [hue, light, sat]
        alpha_color = [0,0,0]
        idx = random.randint(0, len(grounds) - 1)
        restX, restY, size = restrictions[idx]
        size = size * 2
        xMin, xMax = restX
        yMin, yMax = restY
        if xMin == 0 and xMax == 0 and yMin == 0 and yMax == 0: continue
        xMin, xMax, yMin, yMax = 0,100,0,100
        background = cv2.imread("Trimmed/" + grounds[idx])
#        print(grounds[idx])
#        print(shape_color)
        x = random.randint(xMin, xMax)
        y = random.randint(yMin, yMax)
        location = (y,x)
        if positive:
            img = make(target, background, shape_color, alpha_color, size, location)
            filename = "Positive2/img"
        else:
            img = background
            filename = "Negative/neg"
#        cv2.imshow("Image", cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4)))
#        cv2.waitKey(0)
#        data = crop(img, target_shape, location, size)
#        cv2.imwrite("CascadeData/" + filename + str(start + i) + ".png", data)
        xmid, ymid, w, h = get_info(img, target_shape, location, size)
        xmid = xmid / img.shape[1]
        ymid = ymid / img.shape[0]
        w = w / img.shape[1]
        h = h / img.shape[0]
        data = ["0", str(xmid), str(ymid), str(w), str(h)]
        cv2.imwrite("CascadeData/" + filename + str(start + i) + ".png", img)
        file = open("CascadeData/" + filename + str(start + i) + ".txt", "w")
        file.write(' '.join(data))
        file.close()
        i += 1

def test():
    target = cv2.imread("Shapes/pentagon.png")
    background = cv2.imread("Grounds/Frame1.jpg")
    size = 3
    location = (80,50)
    hue = random.randint(0,255)
    light = random.randint(100,255)
    sat = random.randint(light, 255)
    shape_color = [hue, sat, light]
    shape_color = [21, 171, 220]
    alpha_color = [100,150,200]
    data = make(target.copy(), background, shape_color, alpha_color, size, location)
    cropped = crop(data, target.shape, location, size)
    print(cropped[110,90])
    print(cropped[50,90])
#    print(cropped[50,40])
    cv2.imshow("Data", cv2.resize(data, (data.shape[1] // 4, data.shape[0] // 4)))
    cv2.imshow("Cropped.png", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def trim_grounds():
    for idx in range(len(grounds)):
        ground = cv2.imread("Grounds/" + grounds[idx])
        restX, restY, size = restrictions[idx]
        xMin, xMax = restX
        yMin, yMax = restY
        xMin = int(xMin * len(ground[0]) / 100)
        xMax = int(xMax * len(ground[0]) / 100)
        yMin = int(yMin * len(ground) / 100)
        yMax = int(yMax * len(ground) / 100)
        img = ground[yMin:yMax, xMin:xMax]
        cv2.imwrite("Trimmed/" + grounds[idx], img)

#test()
cascade_data(100, True)
#trim_grounds()

"""
Color schemes that look realistic:
BGR
White: Base=[255,255,255], Sat=200 +- 10, Light = 180 +- 10    
Black: Base=[0,0,0], Sat=200 +- 10, Light = 180 +- 10          
Gray: Base=[255,255,255], Sat=200 +- 10, Light = 180 +- 10     
Red:
    Hue = 0-5:
        Light: 100, Sat: 200-255
        Light: 120-140, Sat: 180-255
        Light: 
    Hue = 5-10:




Blue: Base=[255,255,255], Sat=200 +- 10, Light = 180 +- 10     
Green: Base=[255,255,255], Sat=200 +- 10, Light = 180 +- 10    
Yellow: Base=[255,255,255], Sat=200 +- 10, Light = 180 +- 10   
Purple: Base=[255,255,255], Sat=200 +- 10, Light = 180 +- 10   
Brown: Base=[255,255,255], Sat=200 +- 10, Light = 180 +- 10    
Orange: Base=[255,255,255], Sat=200 +- 10, Light = 180 +- 10   
"""
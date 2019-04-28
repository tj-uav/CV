from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np
import os
import time

def getBoxPoints(image, tHold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, tHold, 255, cv2.THRESH_BINARY)

    cnts, ret = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #Combine all contours into one big contour
    p1 = [-1,-1]
    p2 = [-1,-1]
    for cnt in cnts:
        x,y,w,h = cv2.boundingRect(cnt)
        if p1[0] == -1:
            p1 = [x,p1[1]]
        if p1[1] == -1:
            p1 = [p1[0],y]
        p1 = [min(p1[0],x),min(p1[1],y)]
        p2 = [max(p2[0],x+w),max(p2[1],y+h)]
    return p1,p2


def addImage(dir, char, font, filename, pos, toWrite):
    (x,y) = pos
    pil_im = Image.fromarray(cv2_im_rgb)
    text_to_show = i
    draw = ImageDraw.Draw(pil_im)

    # Draw the text
    w, h = draw.textsize(text_to_show, font=font)
    W, H = image.shape[0], image.shape[1]
    draw.text(((W - w) / 2, (H - h) / 2), text_to_show, font=font)

    # Get back the image to OpenCV
    cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

    p1,p2 = getBoxPoints(cv2_im_processed,100)
    p1 = (p1[0],p1[1])
    p2 = (p2[0],p2[1])
    crop = cv2_im_processed[p1[1]:p2[1],p1[0]:p2[0]]
    w = p2[0] - p1[0]
    h = p2[1] - p1[1]
    bigger = max(w, h)
    bigger *= 1.5
    tb = int((bigger - h) / 2)
    rl = int((bigger - w) / 2)
    border = cv2.copyMakeBorder(crop, top=tb, bottom=tb, left=rl, right=rl, borderType=cv2.BORDER_CONSTANT,
                                value=[0, 0, 0])
    gray = cv2.cvtColor(border,cv2.COLOR_RGB2GRAY)
    ret,thresh = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)
    resized = cv2.resize(thresh,(50,50))
    winname = char
    cv2.namedWindow(winname)  # Create a named window
    cv2.moveWindow(winname, x, y)  # Move it to (40,30)
    cv2.imshow(winname, resized)
    path = dir + '/' + filename
    if toWrite:
        cv2.imwrite(path,resized)

#Constan, constanb, ROCCs, impact, TCCB, TCCEB, TCCM, truebuc, trubucbd
# Load image in OpenCV
image = cv2.imread("blank.PNG")
image = cv2.resize(image,(120,120))

# Convert the image to RGB (OpenCV uses BGR)
cv2_im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Pass the image to PIL
fDir = '../../../Fonts/'
chars = [str(i) if i < 10 else chr(i+55) for i in range(36)]
for char in chars:
    os.mkdir('../../../Data/fontCharData/char_data' + char)
keep = []
fonts = []
for file in os.listdir(fDir):
    fonts.append(file)
for f in fonts:
    # use a truetype font
    print(f)
    font = ImageFont.truetype(fDir + f, 80)
    x = 20
    y = 20
    for i in chars:
        addImage('C:/Users/Srikar/Documents/UAV/Data/fontCharData/char_data' + i, i, font, f.split('.')[0] + '.PNG', (x,y), False)
        x += 140
        if x > 1300:
            y += 150
            x = 20
    key = cv2.waitKey(0)
    if key == 113:
        keep.append(f)
    print(key)

time.sleep(5)

for f in keep:
    # use a truetype font
    print(f)
    font = ImageFont.truetype(fDir + f, 80)
    x = 20
    y = 20
    for i in chars:
        addImage('C:/Users/Srikar/Documents/UAV/Data/fontCharData/char_data' + i, i, font, f.split('.')[0] + '.PNG', (x,y), True)
        x += 140
        if x > 1300:
            y += 150
            x = 20
    key = cv2.waitKey(0)
    if key == 113:
        keep.append(fDir + f)
    print(key)


cv2.destroyAllWindows()
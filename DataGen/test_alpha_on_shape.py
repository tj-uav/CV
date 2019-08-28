import cv2
from put_alpha_on_shape import stack_img, processes, border, makeSmaller
import imutils, itertools

"""
0 circle (88, 73) 60
1 semicircle (162, 87) 75
2 quartercircle (42, 103) 65
3 triangle (112, 151) 105
4 square (105, 105) 75
"""
SHAPE_OPTIONS = ["circle", "semicircle", "quartercircle", "triangle", "square", "rectangle", "trapezoid", "pentagon", "hexagon", "heptagon", "octagon", "star", "cross"]
ALPHA_OPTIONS = [chr(i) for i in range(48,58)]
ALPHA_OPTIONS.extend([chr(i) for i in range(65,65+26)])

def register_key(center, key):
    cX, cY = center
    if key == ord('w'):
        return cX,cY-2,0
    elif key == ord('a'):
        return cX-2,cY,0
    elif key == ord('s'):
        return cX,cY+2,0
    elif key == ord('d'):
        return cX+2,cY,0
    elif key == ord('k'):
        return cX,cY,5
    elif key == ord('l'):
        return cX,cY,-5
    else:
        cv2.destroyAllWindows()
        return None

def put_all_letters(shape_thresh):
    oris = []
    threshs = []
    for alpha in ALPHA_OPTIONS:
        alpha_img = cv2.imread('Alphas/' + alpha + '.png')
        alpha_thresh = processes(alpha_img)
        if shape_thresh.shape[1]/alpha_thresh.shape[1] < shape_thresh.shape[0]/alpha_thresh.shape[0]:
            alpha_thresh = imutils.resize(alpha_thresh, width=shape_thresh.shape[1])
        else:
            alpha_thresh = imutils.resize(alpha_thresh, height=shape_thresh.shape[0])

        alpha_thresh = border(alpha_thresh, shape_thresh.shape[1], shape_thresh.shape[0])
        alpha_ori = alpha_thresh.copy()  
        oris.append(alpha_ori)
        threshs.append(alpha_thresh)
    return oris,threshs
 
def display(imgs):
    imgs = [imutils.resize(img, width=int(img.shape[1]/2)) for img in imgs]
    height,width,rando = imgs[0].shape
    buffer = 50
    it = itertools.product([i for i in range(9)], repeat=2)
    for i,img in enumerate(imgs):
        winname = ALPHA_OPTIONS[i]
        cv2.namedWindow(winname) 
        y,x = next(it)
        cv2.moveWindow(winname, x * (width + buffer), y * (height + buffer)) 
        cv2.imshow(winname, img)
 
for i in range(0, len(SHAPE_OPTIONS)):
    shape = SHAPE_OPTIONS[i]
    shape_img = cv2.imread('Shapes/' + shape + '.png')
    shape_thresh = processes(shape_img)

    oris,threshs = put_all_letters(shape_thresh)

    size = 0
    center = (int(shape_thresh.shape[1]/2), int(shape_thresh.shape[0]/2))
    while True:
        display([stack_img(shape_thresh, thresh, center) for thresh in threshs])
        key = cv2.waitKey(0)
        ret = register_key(center, key)
        if not ret:
            break
        cX,cY,dSize = ret
        center = cX,cY
        size += dSize
        threshs = [makeSmaller(oris[j],size) for j in range(len(oris))]
    print(i,shape,center,size)
    cv2.destroyAllWindows()

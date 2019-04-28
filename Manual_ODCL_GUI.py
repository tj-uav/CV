from tkinter import *
from PIL import Image, ImageTk
import cv2
import json
import imutils

root = Tk()      
root.title('Manual ODCL')
width = 600
height = 500
image_width = 265
image_height = 265
#root.geometry('{}x{}'.format(width, height))

root.grid_rowconfigure(2, weight=1)
root.grid_columnconfigure(2, weight=1)

cv_img = cv2.imread("OpenCVscripts/dependencies/generictarget.jpg")
cv_img = cv2.resize(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB), (image_width, image_height))
height, width, no_channels = cv_img.shape
photo = ImageTk.PhotoImage(image = Image.fromarray(cv_img))
canvas = Canvas(root, width = width, height = height)
canvas.grid(row=0,column=0,columnspan=3)
canvas_picture = canvas.create_image(0, 0, image=photo, anchor=NW)

userFrame = Frame(root, bg='cyan', padx=10, pady=10)
userFrame.grid(row=0, column=3, rowspan=1)
userFrame.grid_rowconfigure(2, weight=1)
userFrame.grid_columnconfigure(2, weight=1)

COLOR_OPTIONS = [
        "BLACK",
        "GRAY",
        "WHITE",
        "RED",
        "BLUE",
        "GREEN",
        "BROWN",
        "ORANGE",
        "YELLOW",
        "PURPLE"
        ]

SHAPE_CLASS_OPTIONS = [
        "CIRCLE",
        "SEMICIRCLE",
        "QUARTER_CIRCLE",
        "TRIANGLE",
        "SQUARE",
        "RECTANGLE",
        "TRAPEZOID",
        "PENTAGON",
        "HEXAGON",
        "HEPTAGON",
        "OCTAGON",
        "STAR",
        "CROSS"
        ]

ALPHA_CLASS_OPTIONS = [str(i) for i in range(10)]
ALPHA_CLASS_OPTIONS.extend([chr(i+65) for i in range(26)])

#userFrame.grid_rowconfigure(1, weight=1)
#userFrame.columnconfigure(0, weight=1)

shapeFrame = Frame(userFrame, height=int(height/3))
shapeFrame.grid(row=0, column=0, rowspan=1, columnspan=2, sticky='n')
shapeFrame.rowconfigure(0, weight=1)
shapeFrame.columnconfigure(0, weight=1)

alphaFrame = Frame(userFrame, height=int(height/3))
alphaFrame.grid(row=1, column=0, rowspan=1, columnspan=2, sticky='s', pady=20)
alphaFrame.rowconfigure(0, weight=1)
alphaFrame.columnconfigure(0, weight=1)


shapeColorLabel = Label(shapeFrame, text="Shape Color:")
shapeColor = StringVar(shapeFrame)
shapeColor.set(COLOR_OPTIONS[0])
shapeColorDropdown = OptionMenu(shapeFrame, shapeColor, *COLOR_OPTIONS)
shapeColorLabel.grid(in_=shapeFrame, row=0, column=0, padx=10, pady=10)
shapeColorDropdown.grid(in_=shapeFrame, row=0, column=1, padx=10, pady=10)

shapeClassLabel = Label(shapeFrame, text="Shape Classification:")
shapeClass = StringVar(shapeFrame)
shapeClass.set(SHAPE_CLASS_OPTIONS[0])
shapeClassDropdown = OptionMenu(shapeFrame, shapeClass, *SHAPE_CLASS_OPTIONS)
shapeClassLabel.grid(in_=shapeFrame, row=1, column=0, padx=10, pady=10)
shapeClassDropdown.grid(in_=shapeFrame, row=1, column=1, padx=10, pady=10)

alphaColorLabel = Label(alphaFrame, text="Alphanumeric Color:")
alphaColor = StringVar(alphaFrame)
alphaColor.set(COLOR_OPTIONS[0])
alphaColorDropdown = OptionMenu(alphaFrame, alphaColor, *COLOR_OPTIONS)
alphaColorLabel.grid(in_=alphaFrame, row=0, column=0, padx=10, pady=10)
alphaColorDropdown.grid(in_=alphaFrame, row=0, column=1, padx=10, pady=10)

alphaClassLabel = Label(alphaFrame, text="Alpha Classification:")
alphaClass = StringVar(alphaFrame)
alphaClass.set(ALPHA_CLASS_OPTIONS[0])
alphaClassDropdown = OptionMenu(alphaFrame, alphaClass, *ALPHA_CLASS_OPTIONS)
alphaClassLabel.grid(in_=alphaFrame, row=1, column=0, padx=10, pady=10)
alphaClassDropdown.grid(in_=alphaFrame, row=1, column=1, padx=10, pady=10)

def updateValue(event):
        global photo
        val = alphaOrientationScale.get()
        print(val)
        rotated = imutils.rotate(cv_img, angle=int(val))
        photo = ImageTk.PhotoImage(image=Image.fromarray(rotated))
        canvas.itemconfigure(canvas_picture, image=photo, anchor=NW)


alphaOrientationScale = Scale(root, from_=0, to_=359, length=500, orient=HORIZONTAL, command=updateValue)
#alphaOrientationScale.bind("<ButtonRelease-1>", updateValue)
alphaOrientationScale.grid(row=1, column=0, columnspan=4)

def num_to_orientation(val):
        dirs = ["n", "ne", "e", "se", "s", "sw", "w", "nw"]
        closestRotation = float("inf")
        closestIndex = 0
        for index in range(len(dirs)):
                dir = dirs[index]
                dist = min(abs(index * 45 - val), abs(index * 45 + 360 - val))
                if dist < closestRotation:
                        closestRotation = dist
                        closestIndex = index
        return dirs[closestIndex]

def make_json():
        plane_rotation = 0
        img_rotation = alphaOrientationScale.get()
        total_rotation = plane_rotation + img_rotation
        info_dict = {}
        info_dict["id"] = 1
        info_dict["type"] = "STANDARD"
        info_dict["latitude"] = 0 #Need GPS Data
        info_dict["longitude"] = 0 #Need GPS Data
        info_dict["orientation"] = num_to_orientation(total_rotation)
        info_dict["shape"] = shapeClass.get()
        info_dict["shapeColor"] = shapeColor.get()
        info_dict["alphanumeric"] = alphaClass.get()
        info_dict["alphanumeric_color"] = alphaColor.get()
        info_dict["autonomous"] = False
        return json.dumps(info_dict)


def submit_file():
        json_file = make_json()
        print(json_file)
        #Ask for confirmation
        #Send file to Communications Computer

submitButton = Button(root, text='Submit JSON File', command=submit_file, bg="GREEN", fg="RED", width=70)
submitButton.grid(row=2, column=0, columnspan=4)

mainloop()

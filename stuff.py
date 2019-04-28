from tkinter import *
from PIL import Image, ImageTk
import cv2
import imutils

def stuff(event):
    global picture3
    cv_im = cv2.imread('OpenCVscripts/dependencies/generictarget2.jpg')
    cv_im = cv2.cvtColor(cv_im, cv2.COLOR_BGR2RGB)
    rotated = imutils.rotate(cv_im, int(alphaOrientationScale.get()))
    picture3 = ImageTk.PhotoImage(Image.fromarray(rotated))
    c.itemconfigure(picture2, image = picture3)

main = Tk()
c = Canvas(main, width=300, height=300)
c.pack()

picture = ImageTk.PhotoImage(Image.open('OpenCVscripts/dependencies/generictarget2.jpg'))
picture2 = c.create_image(150,150,image=picture)

alphaOrientationScale = Scale(main, from_=0, to_=359, length=500, orient=HORIZONTAL, command=stuff)
#alphaOrientationScale.bind("<ButtonRelease-1>", stuff)
alphaOrientationScale.pack()

c.bind("<Button-1>", stuff)

main.mainloop()
from tkinter import *

root = Tk()
root.configure(background='#000000')
buttons = []
dict = {
        1: "Black",
        2: "Gray",
        3: "White",
        4: "Red",
        5: "Blue",
        6: "Green",
        7: "Brown",
        8: "Orange",
        9: "Yellow",
        10: "Purple"
        }
color = ""
def add(str):
    global color
    print('hi')
    color = str
    root.quit()

for key in dict:
    button = Button(master = root, text = dict[key], command = lambda x1 = key: add(dict[x1]))
    button.pack()
    buttons.append(button)
#T = Label(root, height=10, width=30, text = "1: Black\n2: Gray\n3: White\n4: Red\n5: Blue\n6: Green\n7: Brown\n8: Orange\n9: Yellow\n10: Purple")
#L = Label(root, height = 5, width = 30, text = "")
#T.pack()
#L.pack()

output = open("RGB_Names.txt","a")
inputFile = open("RGB_Names.txt","r")
length = len(inputFile.readlines())
currR = 0
currG = 0
currB = 0
count = 0
print('hi')
for r in range(0,255,20):
    for b in range(0,255,20):
        for g in range(0,255,20):
            if count < length:
                count += 1
                continue
            print(r,g,b)
            currR = r
            currB = b
            currG = g
            root["bg"] = '#%02x%02x%02x' % (r, g, b)
            root.mainloop()
            output.write(str([r,g,b]) + " " + color + "\n")
            print(color)

output.close()

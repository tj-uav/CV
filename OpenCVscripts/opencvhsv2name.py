import cv2
h = 180
s = 27
v = 54
if s < 75: #Low Saturation
    if v < 50: #Low Value
        print( "Black" )
    elif v < 200: #Mid Value
        print( "Gray" )
    else:   #High Value
        print( "White" )
else:   #Anything not low saturation is a valid color
    if 165 < h or h < 8:    #Red
        print( "Red" )
    elif h < 21:            #Orange/Brown
        if v < 220:
            print( "Brown" )
        else:
            print( "Orange" )
    elif h < 34:            #Yellow
        print( "Yellow" )
    elif h < 85:
        print( "Green" )
    elif h < 130:
        print( "Blue" )
    else:
        print( "Purple" ) 

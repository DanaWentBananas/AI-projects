import numpy as n
import cv2 as c

canvas = n.ones([500,500,3],'uint8')*255

color = (0,255,0)
lineWidth = 3
radius = 5
point=(0,0)
pressed = False


#click functions
def click(event,x,y, flags, param):
    global canvas,point,pressed
    
    if event == c.EVENT_LBUTTONDOWN:
        pressed = True
    if event == c.EVENT_MOUSEMOVE and pressed == True:
        point = (x,y)
    elif event == c.EVENT_LBUTTONUP:
        pressed = False
    
c.namedWindow("canvas")
c.setMouseCallback("canvas", click)

while True:
    
    c.imshow("canvas",canvas)
    
    c.circle(canvas, point, radius, color, lineWidth)
    
    if c.waitKey(1) & 0xFF == ord('q'):
        break
    
c.destroyAllWindows()
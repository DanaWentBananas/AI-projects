import cv2 as c
import numpy as n
#identify the perimeter and area of suquares
#print on blank page
#print perimeter and area
#each should have its own color

orgimg = c.imread("fuzzy.png",1)
grayimg = c.imread("fuzzy.png",0)

#step1: bilateral blur
blur1 = c.bilateralFilter(grayimg, 5, 100, 100)

#step2: binary threshold
ret, binaryThresh = c.threshold(blur1,165,255,c.THRESH_BINARY)

#step3: median blur on binary threshold
blur2 = c.medianBlur(binaryThresh, 5)

c.imshow('threshold',blur2)

#step4: contour
imgg = orgimg.copy()
index=-1
thickness = 4
color = (0,0,0)

contours, heirarchy = c.findContours(blur2, c.RETR_TREE, c.CHAIN_APPROX_SIMPLE)
c.drawContours(imgg,contours,index,color,thickness)

c.imshow("final",imgg)

#step5: draw shapes on another window

canvas = n.ones([300,400,3],'uint8')*255

#loop through number of the contours that were already made 
for counter,t in enumerate(contours):
    
    #skip the unwanted contour of the whole image
    if counter==0:
        continue
    
    area = c.contourArea(t)
    perimeter = c.arcLength(t, True)
    
    print(f'area: {area}, perimeter: {perimeter}')
    
    if area<10000:
        color=(255,0,0)
    elif area<20000:
        color=(0,0,255)
    elif area>20000:
        color=(0,255,0)
    
    #why is t passed as a list?
    c.drawContours(canvas, [t], -1, color, -1)
    
    
c.imshow('redraw',canvas)


c.waitKey(0)
c.destroyAllWindows()
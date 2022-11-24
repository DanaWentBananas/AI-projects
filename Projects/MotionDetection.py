import cv2 as c
import numpy as n

cap = c.VideoCapture('stuff/walking.mp4')

ret, frame1 = cap.read()
ret, frame2 = cap.read()

while True:
    
    diff = c.absdiff(frame1,frame2)
    gray = c.cvtColor(diff, c.COLOR_BGR2GRAY)
    
    #blur
    blur = c.GaussianBlur(gray, (5,5), 0)
    
    #threshold
    ret,thresh = c.threshold(blur,20,255,c.THRESH_BINARY)
    
    #dilate to fill holes
    dilated = c.dilate(thresh,None, iterations = 3)
    
    #contour
    contours,heirarchy = c.findContours(dilated, c.RETR_TREE, c.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        (x,y,w,h) = c.boundingRect(contour)
        
        if c.contourArea(contour) <500:
            continue
        
        c.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
    
    
    c.imshow("feed",frame1)
    
    #go to next frame
    frame1 = frame2
    ret,frame2 = cap.read()
    
    if c.waitKey(3) & 0xFF == ord('q'):
        break
    
c.destroyAllWindows()
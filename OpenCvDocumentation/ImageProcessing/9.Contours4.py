import cv2 as c
import numpy as n

img = c.imread('blobs.png')
img2 = img.copy()
img3 = img.copy()
img4 = img.copy()
img5 = img.copy()
img6 = img.copy()
gray = c.cvtColor(img, c.COLOR_BGR2GRAY)

ret, thresh = c.threshold(gray,10,255,0)

c.imshow('thresh',thresh)

#finding contours
contours,heirarchy = c.findContours(thresh, c.RETR_TREE, c.CHAIN_APPROX_NONE)

#drawing contours
c.drawContours(img,contours,-1,(0,0,255),3)
c.imshow('normal',img)

#Bounding straight rectangle
for i in contours:
    x,y,w,h = c.boundingRect(i)
    c.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),2)
 
c.imshow("straightBoundingRectangle",img2)

#Bounding rotated rectangle
for i in contours:
    rect = c.minAreaRect(i)
    box = c.boxPoints(rect)
    box = n.int0(box)
    c.drawContours(img3,[box],0,(0,255,0),2)
    
c.imshow("rotatedBoundingRectangle",img3)

#Enclosing circle
for i in contours:
    (x,y),radius = c.minEnclosingCircle(i)
    center = (int(x),int(y))
    radius = int(radius)
    c.circle(img4,center,radius,(0,255,0),2)
    
c.imshow('EnclosingCircle',img4)

#Ellipse
for i in contours:
    ellipse = c.fitEllipse(i)
    c.ellipse(img5,ellipse,(0,255,0),2)
    
c.imshow("FitEllipse",img5)

#Line ?
#rows,cols = thresh.shape[:2]
# for i in contours:
#     [vx,vy,x,y] = c.fitLine(i, c.DIST_L2,0,0.01,0.01)
#     lefty = int((-x*vy/vx) + y)
#     righty = int(((cols-x)*vy/vx)+y)
#     c.line(img6,(cols-1,righty),(0,lefty),(0,255,0),2)
#     
# c.imshow("FitLine",img6)

c.waitKey(0)
c.destroyAllWindows()
import cv2 as c
import numpy as n

img = c.imread("coins2.jpg")
gray = c.cvtColor(img,c.COLOR_BGR2GRAY)
ret,thresh = c.threshold(gray,0,255,c.THRESH_BINARY_INV+c.THRESH_OTSU)

kernel = n.ones((3,3),n.uint8)
#remove noise
opening = c.morphologyEx(thresh,c.MORPH_OPEN,kernel,iterations=2)

background = c.dilate(opening,kernel,iterations=3)
distance = c.distanceTransform(opening,c.DIST_L2,5)
ret,foreground = c.threshold(distance,0.7*distance.max(),255,0)
unknown = c.subtract(background,n.uint8(foreground))

ret,markers = c.connectedComponents(n.uint8(foreground))
markers = markers+1
markers[unknown==255]=0

markers = c.watershed(img,markers)
img[markers == -1] = [255,0,0]

c.imshow('f',img)
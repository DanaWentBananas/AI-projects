import cv2 as c
import numpy as n

img = c.imread("coin1.jpg")
img = c.resize(img, None, fx=0.6, fy=0.6, interpolation = c.INTER_CUBIC)
gray = c.cvtColor(img, c.COLOR_BGR2GRAY)

#threshold
#ret, thresh = c.threshold(gray,10,255,c.THRESH_BINARY + c.THRESH_OTSU)
ret, thresh = c.threshold(gray,10,255,c.THRESH_BINARY_INV + c.THRESH_TRIANGLE)

#what otsu/triangle decided is the best threshold
print(ret)

#remove inside noise
kernel = n.ones((3,3),n.uint8)
opening = c.morphologyEx(thresh, c.MORPH_OPEN, kernel)

dilate = c.dilate(opening,kernel,iterations=3)
c.imshow("erosion",dilate)

transform = c.distanceTransform(opening,c.DIST_L2,5)
ret, sure = c.threshold(transform,0.7*transform.max(),255,0)
# Finding unknown region
sure = n.uint8(sure)
unknown = c.subtract(dilate,sure)

c.imshow('opening',opening)
c.imshow('erosion',dilate)
c.imshow("orig",unknown)
c.imshow("thresh",thresh)

c.waitKey(0)
c.destroyAllWindows()

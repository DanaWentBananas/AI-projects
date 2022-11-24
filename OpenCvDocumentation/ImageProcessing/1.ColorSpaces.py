import cv2 as c
import numpy as n

#________________seeing all conversions
#flags = [i for i in dir(c) if i.startswith('COLOR_')]
#print(flags)

#FIND a color range
#1) http://color.lukas-stratmann.com/color-systems/hsv.html
#2)
#green = n.uint8([[[0,255,0]]])
#hsvGreen = c.cvtColor(green, c.COLOR_BGR2HSV)
#print(hsvGreen)

#HSV range: (0:179,0:255,0:255)

img = c.imread('flower.jpg',1)

#change to hsv
hsv = c.cvtColor(img, c.COLOR_BGR2HSV)

#range of colors
lower = n.array([0,50,50])
upper = n.array([150,255,255])

#threshold image based on range
mask = c.inRange(hsv, lower, upper)

#AND them
final = c.bitwise_and(img,img,mask=mask)

c.imshow("img",img)
c.imshow("mask",mask)
c.imshow("final",final)

c.waitKey(0)
c.destroyAllWindows()
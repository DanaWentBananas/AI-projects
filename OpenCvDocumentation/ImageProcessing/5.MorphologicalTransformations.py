import cv2 as c
import numpy as n

#Normally, in cases like noise removal, erosion is followed by dilation.
#Because, erosion removes white noises, but it also shrinks our object.
#So we dilate it. Since noise is gone, they won't come back, but our object area increases.
#It is also useful in joining broken parts of an object.

img = c.imread("j.png",0)
c.imshow("org",img)

#will use in all trnasformations
kernel = n.ones((5,5),n.uint8)

#________EROSION
#try to keep foreground as white
#A pixel in the original image (either 1 or 0) will be considered 1 only if all the pixels under the kernel is 1, otherwise it is eroded (made to zero).
#pixels near boundary will be discarded

erosion = c.erode(img,kernel,iterations=1)
c.imshow("erosion",erosion)

#________DIALATION
#a pixel element is '1' if atleast one pixel under the kernel is '1'.
#it increases the white region in the image or size of foreground object increases.

dilation = c.dilate(img,kernel,iterations=1)
c.imshow("dilation",dilation)

#________OPENING

opening = c.morphologyEx(img, c.MORPH_OPEN, kernel)
c.imshow('opening',opening)

#________CLOSING

closing = c.morphologyEx(img, c.MORPH_CLOSE, kernel)
c.imshow('closing',closing)

#________GRADIANT

gradient = c.morphologyEx(img, c.MORPH_CLOSE, kernel)
c.imshow('gradient',gradient)

#________TOP HAT

tophat = c.morphologyEx(img, c.MORPH_TOPHAT, kernel)
c.imshow('tophat',gradient)

#________BLACK HAT

blackhat = c.morphologyEx(img, c.MORPH_BLACKHAT, kernel)
c.imshow('blackhat',blackhat)


#__________GETTING OTHER SHAPED KERNELS
rect = c.getStructuringElement(c.MORPH_RECT,(5,5))

ellipse = c.getStructuringElement(c.MORPH_ELLIPSE,(5,5))

cross =  c.getStructuringElement(c.MORPH_CROSS,(5,5))

c.waitKey(0)
c.destroyAllWindows()
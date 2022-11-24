import cv2 as c
import numpy as n

#c.findContours(source,arrayOfContour,heirarchy,mode,method,offset)

#Retrievel modes:
#RETR_TREE > retrieves all of the contours, full family heirarchy
#RETR_EXTERNAL > retrieves only the extreme outer contours. child contours are left behind
#RETR_LIST > retrieves all of the contours without establishing any hierarchical relationships.
#RETR_CCOMP > a two level heirarchy
#RETR_FLOODFILL > ?

#Approximation methods:
#CHAIN_APPROX_NONE > stores all contour points
#CHAIN_APPROX_SIMPLE > compresses horizontal, vertical, and diagonal segments and leaves only their end points.
#CHAIN_APPROX_TC89_L1 > Teh-Chin chain approximation algorithm
#CHAIN_APPROX_TC89_KCOS > Teh-Chin chain approximation algorithm

#Optional offset by which every contour point is shifted.
#This is useful if the contours are extracted from the image
#ROI and then they should be analyzed in the whole image context


#________________________________________________


#c.drawContours(destination,contours,contourldx,color,thickness,linetype,heirarchy,maxlevel,offset)

#contourldx: which contour to draw (-1 draws all)

#linetypes: c.FILLED, c.LINE_4, c.LINE_8, c.LINE_AA

#heirarchy: only needed if only some contouors are needed to be drawn


#Maximal level for drawn contours. If it is 0, only the specified contour is
#drawn. If it is 1, the function draws the contour(s) and all the nested contours.
#If it is 2, the function draws the contours, all the nested contours,
#all the nested-to-nested contours, and so on. This parameter is only taken
#into account when there is hierarchy available.

#offset (x,y) an optional shift



img = c.imread('opencv.png',1)
gray= c.cvtColor(img, c.COLOR_BGR2GRAY)

#we gotta do thresh/canny/inrange
#binarythresh
ret, thresh = c.threshold(gray,10,255,0)

c.imshow('thresh',thresh)

#___________________finding contours
contours,heirarchy = c.findContours(thresh, c.RETR_TREE, c.CHAIN_APPROX_NONE)

#___________________drawing contours
c.drawContours(img,contours,-1,(0,0,255),3)

c.imshow('m',img)



c.waitKey(0)
c.destroyAllWindows()
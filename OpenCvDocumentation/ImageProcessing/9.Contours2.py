import cv2 as c
import numpy as n

font = c.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color = (255, 255, 255)
thickness = 2

img = c.imread('blobs.png')
img2 = img.copy()
gray = c.cvtColor(img,c.COLOR_BGR2GRAY)

ret, thresh = c.threshold(gray,10,255,0)

c.imshow('thresh',thresh)

#finding contours
contours,heirarchy = c.findContours(thresh, c.RETR_TREE, c.CHAIN_APPROX_NONE)


#___________________contour features

#number of contours
print(str(len(contours)))

for i,contour in enumerate(contours, start=1):
    #moment of contours
    M = c.moments(contour)

    #centroid
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    #show which shape
    origin = cx,cy
    img2 = c.putText(img2,f'{i}',origin,font,fontScale,color,thickness,c.LINE_AA)
    

    #area
    area = c.contourArea(contour)

    #perimeter
    #flag is weather curve is closed or not
    per = c.arcLength(contour, True)
    
    print(f'shape {i} has area: {area} and per: {per}')
    
#_________contour propertes
    
    
for i,contour in enumerate(contours,start=1):
    #AspectRatio = Width/Height
    x,y,w,h = c.boundingRect(contour)
    aspectRatio = float(w)/h
    
    #Extent = ObjectArea / BoundingRectArea
    area = c.contourArea(contour)
    rectArea=w*h
    extent = float(area)/rectArea
    
    #Solidity = ContourArea/ConvexHullArea
    hull = c.convexHull(contour)
    hullArea = c.contourArea(hull)
    solidity = float(area)/hullArea
    
    #Equivalent diametr
    equiDia = n.sqrt(4*area/n.pi)
    
    #Orientation
    #origin, major and minor axis, angle of rotation
    (x,y),(MA,ma),angle = c.fitEllipse(contour)
    
    #Mask and Pixel points
    mask = n.zeros(gray.shape,n.uint8)
    #answer in row,colum format
    pixelpoints = n.transpose(n.nonzero(mask))
    #answer in x,y format
    pixelpoints2 = c.findNonZero(mask)
    
    #max and min values
    minVal, maxVal, minLoc, maxLoc = c.minMaxLoc(gray,mask = mask)
    
    #Mean Color
    mean = c.mean(img,mask=mask)
    
    #Extreme points (upper,lower,right,left)
    leftmost = tuple(contour[contour[:,:,0].argmin()][0])
    rightmost = tuple(contour[contour[:,:,0].argmax()][0])
    topmost = tuple(contour[contour[:,:,1].argmin()][0])
    bottommost = tuple(contour[contour[:,:,1].argmax()][0])
    
c.imshow('d',img2)

#___________MATCH CONTOURS

#ret = c.matchShapes(contour1,contour2,method,0.0)

#methods:
#CONTOURS_MATCH_I1
#CONTOURS_MATCH_I2
#CONTOURS_MATCH_I3


c.waitKey(0)
c.destroyAllWindows()
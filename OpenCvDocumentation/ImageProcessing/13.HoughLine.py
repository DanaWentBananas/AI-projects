import cv2 as c
import numpy as n

img = c.imread("sudoku.jfif")
img2 = img.copy()
gray = c.cvtColor(img, c.COLOR_BGR2GRAY)

#apply edge detection
edges = c.Canny(gray, 50, 100 , apertureSize = 3)

#houghlines
#HoughLines(img, rho, theta, threshold)
lines = c.HoughLines(edges,1,n.pi/180,200)

for line in lines:
    rho,theta = line[0]
    
    a = n.cos(theta)
    b = n.sin(theta)
    
    x0 = a*rho
    y0 = b*rho
    
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
       
    #Draw line
    c.line(img, (x1,y1), (x2,y2), (255,0,0), 2)
    
#HoughLines probabilistic
linesP = c.HoughLinesP(edges, 1, n.pi/180,100,minLineLength=100,maxLineGap=10)

for line in linesP:
    x1,y1,x2,y2 = line[0]
    
    c.line(img2, (x1,y1), (x2,y2), (255,0,0), 2)

c.imshow('HoughLines',img)
c.imshow('HoughLinesP',img2)

c.waitKey(0)
c.destroyAllWindows()
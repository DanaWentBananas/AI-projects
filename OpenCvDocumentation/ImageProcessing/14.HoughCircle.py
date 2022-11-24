import cv2 as c
import numpy as n

img = c.imread("mm.png")
gray = c.cvtColor(img, c.COLOR_BGR2GRAY)

#blur
median = c.medianBlur(gray,31)

c.imshow("d",median)

#Hough circle
#HoughCircles(img, HoughMode, dp,min distance, param1,param2,minradius,maxradius)

#HOUGH_GRADIENT
circles = c.HoughCircles(median, c.HOUGH_GRADIENT, 1, 50, param1 = 50, param2 = 30, minRadius=100, maxRadius=160)
detectedCircles = n.uint16(n.around(circles))

#HOUGH_GRADIENT_ALT
#circles = c.HoughCircles(median, c.HOUGH_GRADIENT_ALT, 1, 50, param1 = 50, param2 = 30, minRadius=100, maxRadius=160)
#detectedCircles = n.uint16(n.around(circles))

#HOUGH_STANDARD
#circles = c.HoughCircles(median, c.HOUGH_STANDARD, 1, 50, param1 = 50, param2 = 30, minRadius=100, maxRadius=160)
#detectedCircles = n.uint16(n.around(circles))

#HOUGH_PROBABILISTIC
#circles = c.HoughCircles(median, c.HOUGH_PROBABILISTIC, 1, 50, param1 = 50, param2 = 30, minRadius=100, maxRadius=160)
#detectedCircles = n.uint16(n.around(circles))

#HOUGH_MULTI_SCALE
#circles = c.HoughCircles(median, c.HOUGH_MULTI_SCALE, 1, 50, param1 = 50, param2 = 30, minRadius=100, maxRadius=160)
#detectedCircles = n.uint16(n.around(circles))

for circle in detectedCircles[0,:]:
    
    x = circle[0]
    y = circle[1]
    r = circle[2]
    
    c.circle(img, (x,y), r, (255,0,0), 3)
    c.circle(img, (x,y), 2, (0,255,0), 3)
    
c.imshow("HoughCircle",img)
                         
c.waitKey(0)
c.destroyAllWindows()
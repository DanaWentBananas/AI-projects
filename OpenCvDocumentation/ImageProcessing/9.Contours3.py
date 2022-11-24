import cv2 as c

img = c.imread('hand.jpg')
img2 = img.copy()
gray = c.cvtColor(img, c.COLOR_BGR2GRAY)

ret, thresh = c.threshold(gray,250,255, c.THRESH_BINARY_INV)

contours,heirarchy = c.findContours(thresh, c.RETR_TREE, c.CHAIN_APPROX_NONE)

c.drawContours(img,contours,-1,(0,0,255),3)

c.imshow('normal',img)


#________________CONVEX HULL


hull = [c.convexHull(con) for con in contours]
final = c.drawContours(img2, hull, -1, (0,0,255))

c.imshow('hull',img2)

#_____________________Contour approximation

#epsilon = 0.1*c.arcLength(cnt,True)
#approx = c.approxPolyDP(cnt,epsilon,True)

#____________________Checking convexity
#k = cv.isContourConvex(cnt)

c.waitKey(0)
c.destroyAllWindows()
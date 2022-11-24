import cv2 as c
import numpy as n

img = c.imread("sudoku.jfif",0)
#c.imshow('org',orgimg)



#_________LAPLACIAN
laplacian = c.Laplacian(img, c.CV_64F,ksize=3)
laplacian = n.uint8(n.absolute(laplacian))
c.imshow("laplacian",laplacian)

#_________SOBEL
sobelx = c.Sobel(img, c.CV_64F,1,0)
sobelx = n.uint8(n.absolute(sobelx))
c.imshow("sobelx",sobelx)

sobely = c.Sobel(img, c.CV_64F,0,1)
sobely = n.uint8(n.absolute(sobely))
c.imshow("sobely",sobely)



c.waitKey(0)
c.destroyAllWindows()
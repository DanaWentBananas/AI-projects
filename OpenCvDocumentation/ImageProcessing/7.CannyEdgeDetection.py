import cv2 as c

img = c.imread("sudoku.jfif",0)

#_________CANNY EDGE DETECTION
#1) Gaussian filter to remove noise
#2) Sobelx and Sobel y
#3) suppresses unwanted edges
#4) Hysteresis thresholding
#Canny(img, threshold1, threshold2, apareture size)
canny = c.Canny(img,100,20)
c.imshow("canny",canny)

c.waitKey(0)
c.destroyAllWindows()
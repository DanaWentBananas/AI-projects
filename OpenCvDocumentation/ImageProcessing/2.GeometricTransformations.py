import cv2 as c
import numpy as n

img = c.imread("bird.jpeg",0)
c.imshow("org",img)

#_________SCALING
#INTER_CUBIC(slow) and INTER_LINEAR for zooming
#INTER_AREA for shrinking
#INTER_LINEAR for all resizing purposes
#INTER_LINEAR_EXACT
#INTER_NEAREST similar to inter_area
#INTER_NEAREST_EXACT
#INTER_LANCZOS4
#INTER_MAX
#INTER_MAX
#WARP_FILL_OUTLIERS
#WARP_INVERSE_MAP
scale = c.resize(img, None, fx=0.5, fy=0.5, interpolation = c.INTER_CUBIC)
c.imshow("scale",scale)



#_________TRANSLATION

width,height = img.shape

#create transformation matrix
M = n.float32([[1,0,100],[0,1,50]])

#translate
trans = c.warpAffine(img, M, (height,width))
c.imshow("warpAffine/translate",trans)

#_________ROTATION

#transformation matrix
getM = c.getRotationMatrix2D(((height-1)/2.0, (width-1)/2.0),90,1)

#rotate
rotate = c.warpAffine(img, getM, (height,width))
c.imshow("warpAffine/rotate",rotate)


c.waitKey(0)
c.destroyAllWindows()
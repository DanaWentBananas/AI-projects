import numpy as n
import cv2 as c
from matplotlib import pyplot as plt

img = c.imread("flower.jpg",0)
colored = c.imread("flower.jpg",1)

#histogram can show light in an image

#________Histogram using matplotlib

#grey
#plt.hist(img.ravel(), 256, [0,256])
#plt.show()

#bgr
# b,g,r = c.split(colored)
# plt.hist(b.ravel(), 256, [0,256])
# plt.hist(g.ravel(), 256, [0,256])
#plt.hist(r.ravel(), 256, [0,256])
#plt.show()

#________Histogram using opencv

hist = c.calcHist([img], [0], None, [256], [0,256])
plt.plot(hist)
plt.show()

c.waitKey(0)
c.destroyAllWindows()
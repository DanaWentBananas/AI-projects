import cv2 as c

img = c.imread("bird.jpeg",1)

#pyrDown
lower = c.pyrDown(img)
lower2 = c.pyrDown(lower)

#pyrUp
upper =c.pyrUp(img)

#range
layer = img.copy()

gp = [layer]

for i in range(5):
    layer = c.pyrDown(layer)
    gp.append(layer)
    c.imshow(str(i), layer)
    
uppergp = gp[5]

#laplacian pyramids

for i in range(5,0,-1):
    gaussianExtended = c.pyrUp(gp[i])
    laplacian = c.subtract(gp[i-1], gaussianExtended)
    c.imshow(str(i),laplacian)


c.waitKey(0)
c.destroyAllWindows()

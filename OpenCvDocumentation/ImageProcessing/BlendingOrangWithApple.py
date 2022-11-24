import cv2 as c
import numpy as n

apple = c.imread("apple.png")
pop = c.imread("orange.png")
orange = c.resize(pop,(300,300))


#create half + half
#AppleOrange = n.hstack((apple[:,:140],orange[:,160:]))

#Gaussian pyramid for apple
apple2 = apple.copy()
gpA = [apple2]

for i in range(6):
    apple2 = c.pyrDown(apple2)
    gpA.append(apple2)
    
#Gaussian pyramid for orange
orange2 = orange.copy()
gpO = [orange2]

for i in range(6):
    orange2 = c.pyrDown(orange2)
    gpO.append(orange2)

#Laplacian pyramid for apple
lpA = [gpA[5]]

for i in range(5,0,-1):
    x = c.pyrUp(gpA[i])
    laplacian = c.subtract(gpA[i-1], x)
    lpA.append(l)
    
#Laplacian pyramid for orange
lpO = [gpO[5]]

for i in range(5,0,-1):
    x = c.pyrUp(gpO[i])
    l = c.subtract(gpO[i-1], x)
    lpO.append(l)
    
#Add left and right halves in each level
AppleOrangePyr = []
n=0

for la,lo in zip(lpA,lpO):
    rows,cols,ch = la.shape
    laplacian = n.hstack((la[:,:int(cols/2)],lb[:,int(cols/2):]))
    c.imshow('d',ls)
    AppleOrangePyr.append(ls)
    n+=1
    
   
#reconstruct

    

c.waitKey(0)
c.destroyAllWindows()

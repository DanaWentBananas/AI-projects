import cv2 as c
import numpy as n


img1 =  c.imread("A.png",c.IMREAD_GRAYSCALE)
img2 =  c.imread("B.png",c.IMREAD_GRAYSCALE)
img3 =  c.imread("C.png",c.IMREAD_GRAYSCALE)

f1 = n.fft.fft2(img1)
fshift1 = n.fft.fftshift(f1)
mag1 = 20*n.log(n.abs(fshift1))
mag1 = n.asarray(mag1,dtype=n.uint8)

A = n.concatenate((img1,mag1),axis=1)
c.imshow("A",A)

f2 = n.fft.fft2(img2)
fshift2 = n.fft.fftshift(f2)
mag2 = 20*n.log(n.abs(fshift2))
mag2 = n.asarray(mag2,dtype=n.uint8)

B = n.concatenate((img2,mag2),axis=1)
c.imshow("B",B)

f3 = n.fft.fft2(img3)
fshift3 = n.fft.fftshift(f3)
mag3 = 20*n.log(n.abs(fshift3))
mag3 = n.asarray(mag3,dtype=n.uint8)

C = n.concatenate((img3,mag3),axis=1)
c.imshow("C",C)

c.waitKey(0)
c.destroyAllWindows()
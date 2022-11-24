import cv2 as c

img = c.imread('gradient.jpg')
gray = c.cvtColor(img, c.COLOR_BGR2GRAY)
c.imshow('org',img)

#Binary thresholdg
#black if less than threshold, maxvalue if more
#c.threshold(source,threshold,maxvalue,type)
ret, binary = c.threshold(img,85,200,c.THRESH_BINARY)
c.imshow('binary',binary)

#Binary inv threshold
#black if more than threshold, maxvalue if less
#c.threshold(source,threshold,maxvalue,type)
ret, binaryInv = c.threshold(img,85,200,c.THRESH_BINARY_INV)
c.imshow('binaryInv',binaryInv)

#Trunc threshold
#threshold value if more than threshold, source img if less
#c.threshold(source,threshold,?,type)
ret, trunc = c.threshold(img,85,0,c.THRESH_TRUNC)
c.imshow('trunc',trunc)

#ToZero threshold
#source img if more than threshold, 0 if less
ret,tozero = c.threshold(img,127,255,c.THRESH_TOZERO)
c.imshow('tozero',tozero)

#ToZero inv threshold
#source img if less than threshold,0 if more
ret,tozeroInv = c.threshold(img,127,255,c.THRESH_TOZERO_INV)
c.imshow('tozeroinv',tozeroInv)

#Mean adaptive threshold
#max value if more than calculated thresh, 0 if less
#requires grayscale img
#c.adaptiveThreshold(source,maxvalue,method,thresh type,blocksize,constant subtracted)
meanAdaptive = c.adaptiveThreshold(gray,255,c.ADAPTIVE_THRESH_MEAN_C,c.THRESH_BINARY,11,2)
c.imshow('meanAdaptive',meanAdaptive)

#Gaussian adaptive threshold
#max value if more than calculated thresh, 0 if less
#requires grayscale img
#c.adaptiveThreshold(source,maxvalue,method,thresh type,blocksize,constant subtracted)
gaussianAdaptive = c.adaptiveThreshold(gray,255,c.ADAPTIVE_THRESH_GAUSSIAN_C,c.THRESH_BINARY,11,2)
c.imshow('gaussianAdaptive',gaussianAdaptive)
         
#Otsu's thresholding ??
#ret, otsu = c.threshold(img,0,255,c.THRESH_BINARY+c.THRESH_OTSU)
#c.imshow('otsu',otsu)


c.waitKey(0)
c.destroyAllWindows()
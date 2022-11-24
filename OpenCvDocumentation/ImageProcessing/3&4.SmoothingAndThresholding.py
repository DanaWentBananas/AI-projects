import cv2 as c

orgimg = c.imread("fuzzy.png",1)
grayimg = c.imread("fuzzy.png",0)

#blurs
blurGaus = c.GaussianBlur(grayimg,(7,7),20)
blurMean = c.blur(grayimg, (3,3))
blurMedian = c.medianBlur(grayimg, 5)
blurBilateral = c.bilateralFilter(grayimg, 5, 75, 75)

#_____________________________________________________________________________

#binary thresholds
ret, binaryGaus = c.threshold(blurGaus,85,255,c.THRESH_BINARY)
ret, binaryMean = c.threshold(blurMean,133,255,c.THRESH_BINARY)
ret, binaryMedian = c.threshold(blurMedian,133,255,c.THRESH_BINARY)
ret, binaryBilateral = c.threshold(blurBilateral,133,255,c.THRESH_BINARY)

#_____________________________________________________________________________

#trunc thresholds
ret, truncGaus = c.threshold(blurGaus,85,200,c.THRESH_TRUNC)
ret, truncMean = c.threshold(blurMean,85,200,c.THRESH_TRUNC)
ret, truncMedian = c.threshold(blurMedian,85,200,c.THRESH_TRUNC)
ret, truncBilateral = c.threshold(blurBilateral,85,200,c.THRESH_TRUNC)

#wont use tozero and all the inverse thresholds because its unnecessary for this problem

#_____________________________________________________________________________

#mean adaptive binary thresholds
MeanBinaryAdaptGaus = c.adaptiveThreshold(blurGaus,255, c.ADAPTIVE_THRESH_MEAN_C, c.THRESH_BINARY, 333, 2)
MeanBinaryAdaptMean = c.adaptiveThreshold(blurMean,255, c.ADAPTIVE_THRESH_MEAN_C, c.THRESH_BINARY, 333, 2)
MeanBinaryAdaptMedian = c.adaptiveThreshold(blurMedian,255, c.ADAPTIVE_THRESH_MEAN_C, c.THRESH_BINARY, 333, 2)
MeanBinaryAdaptBilateral = c.adaptiveThreshold(blurBilateral,255, c.ADAPTIVE_THRESH_MEAN_C, c.THRESH_BINARY, 333, 2)

#_____________________________________________________________________________

#gaussian adaptive binary thresholds
GausBinaryAdaptGaus = c.adaptiveThreshold(blurGaus,255, c.ADAPTIVE_THRESH_GAUSSIAN_C, c.THRESH_BINARY, 333, 2)
GausBinaryAdaptMean = c.adaptiveThreshold(blurMean,255, c.ADAPTIVE_THRESH_GAUSSIAN_C, c.THRESH_BINARY, 333, 2)
GausBinaryAdaptMedian = c.adaptiveThreshold(blurMedian,255, c.ADAPTIVE_THRESH_GAUSSIAN_C, c.THRESH_BINARY, 333, 2)
GausBinaryAdaptBilateral = c.adaptiveThreshold(blurBilateral,255, c.ADAPTIVE_THRESH_GAUSSIAN_C, c.THRESH_BINARY, 333, 2)

c.waitKey(0)
c.destroyAllWindows()
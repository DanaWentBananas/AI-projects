import cv2 as c
#identify the perimeter and area of suquares
#print on blank page
#print perimeter and area
#each should have its own color

orgimg = c.imread("fuzzy.png",1)
grayimg = c.imread("fuzzy.png",0)

#blurs
blurGaus = c.GaussianBlur(grayimg,(7,7),20)
blurMean = c.blur(grayimg, (3,3))
blurMedian = c.medianBlur(grayimg, 5)
blurBilateral = c.bilateralFilter(grayimg, 5, 75, 75)
c.imshow("d",blurBilateral)

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

#_____________________________________________________________________________

#contours variables
img1 = orgimg.copy()
img2 = orgimg.copy()
img3 = orgimg.copy()
img4 = orgimg.copy()
img5 = orgimg.copy()
img6 = orgimg.copy()
img7 = orgimg.copy()
img8 = orgimg.copy()
img9 = orgimg.copy()
img10 = orgimg.copy()
img11 = orgimg.copy()
img12 = orgimg.copy()
img13 = orgimg.copy()
img14 = orgimg.copy()
img15 = orgimg.copy()
img16 = orgimg.copy()
index=-1
thickness = 4
color = (0,0,0)

#_____________________________________________________________________________

#contours on binary thresholds
binaryGausContours, heirarchy = c.findContours(binaryGaus, c.RETR_TREE, c.CHAIN_APPROX_SIMPLE)
c.drawContours(img1,binaryGausContours,index,color,thickness)

binaryMeanContours, heirarchy = c.findContours(binaryMean, c.RETR_TREE, c.CHAIN_APPROX_SIMPLE)
c.drawContours(img2,binaryMeanContours,index,color,thickness)

binaryMedianContours, heirarchy = c.findContours(binaryMedian, c.RETR_TREE, c.CHAIN_APPROX_SIMPLE)
c.drawContours(img3,binaryMedianContours,index,color,thickness)

binaryBilateralContours, heirarchy = c.findContours(binaryBilateral, c.RETR_TREE, c.CHAIN_APPROX_SIMPLE)
c.drawContours(img4,binaryBilateralContours,index,color,thickness)

#_____________________________________________________________________________

#contours on trunc thresholds
truncGausContours, heirarchy = c.findContours(truncGaus, c.RETR_TREE, c.CHAIN_APPROX_SIMPLE)
c.drawContours(img5,truncGausContours,index,color,thickness)

truncMeanContours, heirarchy = c.findContours(truncMean, c.RETR_TREE, c.CHAIN_APPROX_SIMPLE)
c.drawContours(img6,truncMeanContours,index,color,thickness)

truncMedianContours, heirarchy = c.findContours(truncMedian, c.RETR_TREE, c.CHAIN_APPROX_SIMPLE)
c.drawContours(img7,truncMedianContours,index,color,thickness)

truncBilateralContours, heirarchy = c.findContours(truncBilateral, c.RETR_TREE, c.CHAIN_APPROX_SIMPLE)
c.drawContours(img8,truncBilateralContours,index,color,thickness)

#_____________________________________________________________________________

#contours on mean adaptive binary thresholds
MeanBinaryAdaptGausContours, heirarchy = c.findContours(MeanBinaryAdaptGaus, c.RETR_TREE, c.CHAIN_APPROX_SIMPLE)
c.drawContours(img9,MeanBinaryAdaptGausContours,index,color,thickness)

MeanBinaryAdaptMeanContours, heirarchy = c.findContours(MeanBinaryAdaptMean, c.RETR_TREE, c.CHAIN_APPROX_SIMPLE)
c.drawContours(img10,MeanBinaryAdaptMeanContours,index,color,thickness)

MeanBinaryAdaptMedianContours, heirarchy = c.findContours(MeanBinaryAdaptMedian, c.RETR_TREE, c.CHAIN_APPROX_SIMPLE)
c.drawContours(img11,MeanBinaryAdaptMedianContours,index,color,thickness)

MeanBinaryAdaptBilateralContours, heirarchy = c.findContours(MeanBinaryAdaptBilateral, c.RETR_TREE, c.CHAIN_APPROX_SIMPLE)
c.drawContours(img12,MeanBinaryAdaptBilateralContours,index,color,thickness)

#_____________________________________________________________________________

#contours on gaussian adaptive binary thresholds
GausBinaryAdaptGausContours, heirarchy = c.findContours(GausBinaryAdaptGaus, c.RETR_TREE, c.CHAIN_APPROX_SIMPLE)
c.drawContours(img13,GausBinaryAdaptGausContours,index,color,thickness)

GausBinaryAdaptMeanContours, heirarchy = c.findContours(GausBinaryAdaptMean, c.RETR_TREE, c.CHAIN_APPROX_SIMPLE)
c.drawContours(img14,GausBinaryAdaptMeanContours,index,color,thickness)

GausBinaryAdaptMedianContours, heirarchy = c.findContours(GausBinaryAdaptMean, c.RETR_TREE, c.CHAIN_APPROX_SIMPLE)
c.drawContours(img15,GausBinaryAdaptMedianContours,index,color,thickness)

GausBinaryAdaptBilateralContours, heirarchy = c.findContours(GausBinaryAdaptBilateral, c.RETR_TREE, c.CHAIN_APPROX_SIMPLE)
c.drawContours(img16,GausBinaryAdaptBilateralContours,index,color,thickness)

c.waitKey(0)
c.destroyAllWindows()
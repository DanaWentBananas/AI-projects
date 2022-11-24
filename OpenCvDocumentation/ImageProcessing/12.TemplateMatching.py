import cv2 as c
import numpy as n

img= c.imread("findBallPic.jpg")
gray = c.cvtColor(img,c.COLOR_BGR2GRAY)
template = c.imread("findBall.png",0)
w, h = template.shape[::-1]

#match template methods

#TM_SQDIFF
#res = c.matchTemplate(gray, template, c.TM_SQDIFF)

#TM_SQDIFF_NORMED
#res = c.matchTemplate(gray, template, c.TM_SQDIFF_NORMED)

#TM_CCORR
#res = c.matchTemplate(gray, template, c.TM_CCORR)

#TM_CCORR_NORMED
#res = c.matchTemplate(gray, template, c.TM_CCORR_NORMED)

#TM_CCOEFF
#res = c.matchTemplate(gray, template, c.TM_CCOEFF)

#CCOEFF_NORMED METHOD
#res = c.matchTemplate(gray, template, c.TM_CCOEFF_NORMED)

c.imshow("d",res)

#find location
threshold = 0.73
loc = n.where(res >= threshold)


for i in zip(*loc[::-1]):
    c.rectangle(img, i, (i[0]+ w,i[1] +h), (0,0,255), 2)

c.imshow("x",img)

c.waitKey(0)
c.destroyAllWindows()
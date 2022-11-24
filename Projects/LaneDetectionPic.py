import cv2 as c
import numpy as n
import matplotlib.pyplot as plt

img = c.imread('stuff/lane1.png')

#matplotlib uses RGB
img = c.cvtColor(img, c.COLOR_BGR2RGB)

height = img.shape[0]
width = img.shape[1]

#Find ROI

#1 > 0,height
#2 > 400,10
#3 > width,300
#4 > 1120,height

ROIvertices = [
    (0,height),
    (390,5),
    (width,300),
    (width,height)
    ]

def ROI(img, vertices):
    #blank matrix with the shape of image
    mask = n.zeros_like(img)
    
    #match color
    #channels = img.shape[2]
    #maskColor = (255,)* channels
    
    c.fillPoly(mask,vertices, 255)
    
    maskedImg = c.bitwise_and(img,mask)
    
    return maskedImg

gray = c.cvtColor(img,c.COLOR_RGB2GRAY)
canny = c.Canny(gray, 100, 200)
ROIimg = ROI(canny, n.array([ROIvertices],n.int32))

def drawLines(img,lines):
    img2 = img.copy()
    lineImage = n.zeros((img2.shape[0],img.shape[1],3), dtype=n.uint8)
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            c.line(img, (x1,y1), (x2,y2), (255,0,0), 2)

    img = c.addWeighted(img, 0.8, lineImage, 1, 0.0)
    
    return img

lines = c.HoughLinesP(ROIimg, rho=6, theta=n.pi/60,threshold=160,minLineLength=40,maxLineGap=25)

pic = drawLines(img, lines)

plt.imshow(pic)
plt.show()


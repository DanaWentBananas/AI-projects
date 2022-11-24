import cv2 as c
import numpy as n

bodyClassifier = c.CascadeClassifier('stuff\haarcascade_fullbody.xml')

cap = c.VideoCapture('stuff\walking.mp4')

while True:
    
    ret, frame = cap.read()
    
    gray = c.cvtColor(frame,c.COLOR_BGR2GRAY)
    
    bodies = bodyClassifier.detectMultiScale(gray,1.2,3)
    
    for (x,y,w,h) in bodies:
        c.rectangle(frame,(x,y), (x+w, y+h), (0,255,0),2)
    
    c.imshow("walking",frame)
    
    if c.waitKey(3) & 0xFF == ord('q'):
        break
    
cap.release()
c.destroyAllWindows()
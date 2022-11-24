import cv2 as c
import numpy as n

#draw rectangle
rectangle = n.zeros((300,300), dtype="uint8")
c.rectangle(rectangle, (25,25), (275,275),255,-1)
c.imshow("rectangle",rectangle)

#draw circle
circle = n.zeros((300, 300), dtype = "uint8")
c.circle(circle, (150, 150), 150, 255, -1)
c.imshow("Circle", circle)

#AND
bitAnd = c.bitwise_and(rectangle,circle)
c.imshow("AND",bitAnd)

#OR
bitOr = c.bitwise_or(rectangle,circle)
c.imshow("OR",bitOr)

#XOR
bitXor = c.bitwise_xor(rectangle,circle)
c.imshow("XOR",bitXor)

#NOT
bitNot = c.bitwise_not(circle)
c.imshow("NOT",bitNot)

c.waitKey(0)
c.destroyAllWindows()
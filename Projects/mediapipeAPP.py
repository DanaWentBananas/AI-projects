import cv2
import mediapipe as mp
from tkinter import *
from PIL import Image
from PIL import ImageTk

root = Tk()

cap = cv2.VideoCapture(0)

#media pipe
draw = mp.solutions.drawing_utils
hands = mp.solutions.hands.Hands()
pose = mp.solutions.pose.Pose()
face = mp.solutions.face_detection.FaceDetection()
mesh = mp.solutions.face_mesh.FaceMesh()

x=''

#commands
def detectHands():
    global x
    x = 'HANDS'

def detectPose():
    global x
    x = 'POSE'

def detectFace():
    global x
    x = 'FACE'

def detectFaceMesh():
    global x
    x = 'FACEMESH'

#widgets

l = Label(root)
l.pack()

handBtn = Button(root,text='detect hands',command=detectHands)
handBtn.pack(side=LEFT)

poseBtn = Button(root, text = 'detect pose',command=detectPose)
poseBtn.pack(side=LEFT)

faceBtn = Button(root, text = 'detect face',command=detectFace)
faceBtn.pack(side=LEFT)

meshBtn = Button(root, text = 'detect faceMesh',command=detectFaceMesh)
meshBtn.pack(side=LEFT)



while True:
        
        ret,frame = cap.read()
        
        #flip
        frame = cv2.flip(frame,1)
        #turn to RGB
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        if x=='HANDS':
            #get result
            results = hands.process(frame).multi_hand_landmarks
            #draw
            if results:
                for landmark in results:
                    draw.draw_landmarks(frame,landmark,mp.solutions.hands.HAND_CONNECTIONS)
            frame = ImageTk.PhotoImage(Image.fromarray(frame))
            l.config(image = frame)
        elif x=='POSE':
            #get result
            results = pose.process(frame).pose_landmarks
            #draw
            if results:
                 draw.draw_landmarks(frame,results,mp.solutions.pose.POSE_CONNECTIONS)
            frame = ImageTk.PhotoImage(Image.fromarray(frame))
            l.config(image = frame)
        elif x=='FACE':
            #get result
            results = face.process(frame).detections
            #draw
            if results:
                for detection in results:
                    draw.draw_detection(frame,detection)
            frame = ImageTk.PhotoImage(Image.fromarray(frame))
            l.config(image = frame)
        elif x=='FACEMESH':
            #get result
            results = mesh.process(frame).multi_face_landmarks
            #draw
            if results:
                for faceMesh in results:
                    draw.draw_landmarks(frame,faceMesh)
            frame = ImageTk.PhotoImage(Image.fromarray(frame))
            l.config(image = frame)
        else:  
            #show in label
            frame = ImageTk.PhotoImage(Image.fromarray(frame))
            l.config(image = frame)
        
        root.update()
        
cap.release()
cv2.destroyAllWindows()



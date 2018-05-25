import cv2
import numpy as np
detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)
while(1):
    ret,frame=cap.read()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        #mask = np.zeros_like(img[x:x+w, y:y+h])
        frame[y:y+h, x:x+w] =0
    lower_blue=np.array([0,20,70])
    upper_blue=np.array([20,255,255])
    mask=cv2.inRange(hsv,lower_blue,upper_blue)
    res=cv2.bitwise_and(frame,frame,mask=mask)
    
    for (x,y,w,h) in faces:
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        #mask = np.zeros_like(img[x:x+w, y:y+h])
        mask[y:y+h, x:x+w] =0
    
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k=cv2.waitKey(5) &0xFF
    if k==27:
        break
cv2.destroyAllWindows()

import cv2
import numpy as np
cap=cv2.VideoCapture(0)
bs=cv2.createBackgroundSubtractorMOG2()
while True:
    ret,frame=cap.read()
    fmask=bs.apply(frame)
    kernel = np.ones((3,3),np.uint8)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0,20,70], dtype=np.uint8)
    upper_skin = np.array([20,255,255], dtype=np.uint8)
    

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.dilate(fmask,kernel,iterations =10)
    mask = cv2.GaussianBlur(fmask,(5,5),100)
    
     

    
    cv2.imshow('original',frame)
    cv2.imshow('bs',mask)

    k=cv2.waitKey(30) &0xff
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()
 

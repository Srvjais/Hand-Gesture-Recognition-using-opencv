import cv2
import numpy as np
facedetector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
id=input('enter user  id')
sample=0
while(True):
    
    ret,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=facedetector.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        sample=sample+1
        cv2.imwrite("dataset/User."+str(id)+"."+str(sample)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('frame',img)
    cv2.waitKey(1)
    if(sample>20):
        break
                    


cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()

import numpy as np
import cv2

detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create();
rec.read("recognizer\\trainingData.yml")
id=0;
font=cv2.FONT_HERSHEY_SIMPLEX


while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if id==1:
            id="sourav"
        if id==2:
            id="shubham"
        if id==3:
            id="tushar"
        cv2.putText(img,str(id),(x,y+h),font,1,255)
        #mask = np.zeros_like(img[x:x+w, y:y+h])
        #img[y:y+h, x:x+w] =0
        

        cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

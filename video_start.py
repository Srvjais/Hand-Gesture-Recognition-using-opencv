import cv2
import numpy as np
cap=cv2.VideoCapture(0)
font=cv2.FONT_HERSHEY_SIMPLEX


while(True):
    ret,frame=cap.read()
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.putText(frame,'100rav',(250,100),font,2,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

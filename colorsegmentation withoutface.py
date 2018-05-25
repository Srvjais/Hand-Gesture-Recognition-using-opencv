import cv2
import numpy as np
detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def nothing(x):
    pass



cap=cv2.VideoCapture(0)

cv2.namedWindow("hand")
cv2.createTrackbar("hue_lower","hand",0,255,nothing)  # 1) Creating trackbar for lower hue value so as to find the desired colored object in frame. 
cv2.createTrackbar("hue_upper","hand",30,255,nothing) # Creating trackbar for upper hue value for same reason as above.
cv2.createTrackbar("saturation_lower","hand",41,255,nothing)  # Creating trackbar for lower saturation value for same reason as above.
cv2.createTrackbar("saturation_upper","hand",152,255,nothing)  # Creating trackbar for upper saturation value for same reason as above.
cv2.createTrackbar("value_lower","hand",217,255,nothing)    # Creating trackbar for lower value for same reason as above.
cv2.createTrackbar("value_upper","hand",255,255,nothing)
cv2.createTrackbar("kernel1","hand",0,100,nothing)
cv2.createTrackbar("kernel2","hand",0,100,nothing)
while(1):
    ret,frame=cap.read()
    k1=cv2.getTrackbarPos("kernel1","hand")
    k2=cv2.getTrackbarPos("kernel2","hand")
    
    kernel = np.ones((5,5),np.uint8)
    kernel1 = np.ones((k1,k1),np.uint8)
    kernel2=np.ones((k2,k2),np.uint8)
    
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        #mask = np.zeros_like(img[x:x+w, y:y+h])
        frame[y:y+h, x:x+w] =0
    
    hl = cv2.getTrackbarPos("hue_lower","hand")  
    hu = cv2.getTrackbarPos("hue_upper","hand")           
    sl = cv2.getTrackbarPos("saturation_lower","hand")    
    su = cv2.getTrackbarPos("saturation_upper","hand")    
    vl = cv2.getTrackbarPos("value_lower","hand")         
    vu = cv2.getTrackbarPos("value_upper","hand")
    
        
    lower_blue=np.array([hl,sl,vl],dtype=np.uint8)
    upper_blue=np.array([hu,su,vu],dtype=np.uint8)
    
    mask=cv2.inRange(hsv,lower_blue,upper_blue)
    mask = cv2.dilate(mask,kernel1,iterations = 2)
    #mask = cv2.erode(mask,kernel,iterations = 2)
    mask = cv2.GaussianBlur(mask,(5,5),30)
    
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel1)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel2)
    #find contours
    _,contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
   #find contour of max area(hand)
    cnt = max(contours, key = lambda x: cv2.contourArea(x))
        
    #approx the contour a little
    epsilon = 0.0005*cv2.arcLength(cnt,True)
    approx= cv2.approxPolyDP(cnt,epsilon,True)

    hull = cv2.convexHull(cnt)
       
    
    
    res=cv2.bitwise_and(frame,frame,mask=mask)
    
    for (x,y,w,h) in faces:
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        #mask = np.zeros_like(img[x:x+w, y:y+h])
        mask[y:y+h, x:x+w] =0
    
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    #cv2.imshow('canny',canny)
    k=cv2.waitKey(5) &0xFF
    if k==27:
        break
cv2.destroyAllWindows()

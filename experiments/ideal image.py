import cv2
import numpy as np
img=cv2.imread('hand.jpg')
img1=cv2.imread('haath.jpg')
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
hsv1=cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)

lower_skin = np.array([0,20,70], dtype=np.uint8)
upper_skin = np.array([20,255,255], dtype=np.uint8)

mask=cv2.inRange(hsv,lower_skin,upper_skin)
mask1=cv2.inRange(hsv1,lower_skin,upper_skin)

cv2.imshow('first',img)
cv2.imshow('second',mask)
cv2.imshow('fourth',img1)
cv2.imshow('third',mask1)
k=cv2.waitKey(5) &0xFF
if k==ord('q'):
    cv2.destroyAllWindows()
    

        

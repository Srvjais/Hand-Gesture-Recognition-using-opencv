import cv2
import numpy as np
img=cv2.imread('gal.jpg',0)
font=cv2.FONT_HERSHEY_SIMPLEX

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.putText(img,'Gal Gadot',(1000,1000),font,4,(255,255,255),2,cv2.LINE_AA)
cv2.imshow('image',img)
k=cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows()
elif k==ord('s'):
    cv2.imwrite('galblack.png',img)
    cv2.destroyAllWindows()


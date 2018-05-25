import cv2
import numpy as np
img=cv2.imread('hand.jpg')
ret,threshold=cv2.threshold(img,12,255,cv2.THRESH_BINARY)
cv2.namedWindow('orig',cv2.WINDOW_NORMAL)
grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow('orig',img)
cv2.imshow('thres',grey)
k=cv2.waitKey(0)

if k==27:
    cv2.destroyAllWindows()
elif k==ord('s'):
    cv2.destroyAllWindows()


import cv2
import numpy as np
import matplotlib.pyplot as pyt
img=cv2.imread('hand.jpg')
hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
h=hsv[:,:,0]
s=hsv[:,:,1]
v=hsv[:,:,2]
f,(ax1,ax2,ax3)=pyt.subplots(1,3,figsize=(20,10))
ax1.imshow(h,cmap='gray')
ax2.imshow(s,cmap='gray')
ax3.imshow(v,cmap='gray')

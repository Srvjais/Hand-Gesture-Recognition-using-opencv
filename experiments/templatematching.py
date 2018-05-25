import cv2
import numpy as np
img=cv2.imread('mario.png')
grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

template=cv2.imread('coin.jpg')
w,h=template.shape[::-1]
res=cv2.matchTemplate(grey,template,cv2.TM_CCOEFF_NORMED)
threshold=0.90
loc=nc.where(res>=threshold)


for pt in zip(*loc[::-1]):
              
              cv2.rectangle(img,pt,(pt[0]+w,pt[1]+h),(0,0,255),2)

cv2.imshow('detected',img)
cv2.waitKey(0)
cv2.DestroyAllWindows()
              

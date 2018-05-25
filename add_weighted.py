import cv2
import numpy as np
s=0.3
img1=cv2.imread('gal.jpg')
img2=cv2.imread('passport.jpg')
dst=cv2.addWeighted(img1,0.3,img2,7,0)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()##two images should have same dimensions for image addition


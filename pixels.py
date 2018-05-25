import cv2
import numpy as np
img=cv2.imread('gal.jpg')
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.namedWindow('image1',cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
img2=img
img2[:,:,1]=0
cv2.imshow('image1',img2)





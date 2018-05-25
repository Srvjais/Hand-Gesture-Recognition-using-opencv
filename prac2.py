import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread('gal.jpg')
plt.imshow(img,cmap='gray',interpolation='bicubic')
plt.xticks([]),plt.yticks([])
plt.show()

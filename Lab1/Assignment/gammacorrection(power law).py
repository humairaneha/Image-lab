# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 15:35:03 2023

@author: ASUS
"""

import cv2
import numpy as np 
import math


#power law or gamma correction

img = cv2.imread('logimg.jpg',cv2.IMREAD_GRAYSCALE)


out=np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)


Xmax= img.max()
Xmin = img.min()

c = 255

gamma = float(input('Enter value of gamma'))

#gamma correction
for i in range (img.shape[0]):
    for j in range (img.shape[1]):
        out[i,j]=c*(img[i,j]/255)**gamma
 
      
cv2.normalize(out,out,0,255,cv2.NORM_MINMAX)
out = np.round(out).astype(np.uint8)
        
cv2.imshow('input image for gamma',img)
cv2.imshow('output image for inverse gamma',out)

# =============================================================================
cv2.waitKey(0) 
cv2.destroyAllWindows()
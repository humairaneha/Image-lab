# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 14:45:02 2023

@author: ASUS
"""


import cv2
import numpy as np 
import math


#log and inverse log 

img1 = cv2.imread('logimg2.jpg',cv2.IMREAD_GRAYSCALE)
img2= cv2.imread('logtransform.jpg',cv2.IMREAD_GRAYSCALE)

out1=np.zeros((img1.shape[0],img1.shape[1]),dtype=np.uint8)
out2=np.zeros((img2.shape[0],img2.shape[1]),dtype=np.uint8)

Xmax= img1.max()
Xmin = img1.min()

c1 = 255/np.log(1+Xmax)
c2 = 255/np.log(1+img2.max())
#log transformation
for i in range (img1.shape[0]):
    for j in range (img1.shape[1]):
        out1[i,j]=np.log(1+img1[i,j])*c1
 
      
 #inverse log

for i in range (img2.shape[0]):
     for j in range (img2.shape[1]):
         out2[i,j]=(math.exp(img2[i,j])**(1/c2))-1

        
cv2.imshow('input image for log transformation',img1)
cv2.imshow('output image for log transformation ',out1)
cv2.imshow('input image for inverse log transformation',img2)
cv2.imshow('output image for inverse log transformation ',out2)

# =============================================================================
cv2.waitKey(0) 
cv2.destroyAllWindows()
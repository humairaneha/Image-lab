# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 14:11:57 2023

@author: ASUS
"""


import cv2
import numpy as np 


#contrast stretching 

img = cv2.imread('LenaContrast.jpg',cv2.IMREAD_GRAYSCALE)
out=np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)

Xmax= img.max()
Xmin = img.min()
dif=Xmax-Xmin

for i in range (img.shape[0]):
    for j in range (img.shape[1]):
        out[i,j]=((img[i,j]-Xmin)/dif)*255
        
cv2.imshow('input image',img)
cv2.imshow('output image',out)
cv2.waitKey(0) 
cv2.destroyAllWindows()
        
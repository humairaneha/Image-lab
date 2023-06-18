# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 23:19:36 2023

@author: ASUS
"""

import numpy as np
import cv2



img = cv2.imread('lena.jpg',cv2.IMREAD_GRAYSCALE)

#out=img.copy()
out = np.zeros((512,512), dtype=np.float32)
print(img.max())
print(img.min())
out

kernel2 = np.array([[1, 2, 3,4,5],
                    [6, 7, 8,9,10],
                    [11, 12, 1,-12,-11],
                    [-10, -9, -8,-7,-6],
                    [-5, -4, -3,-2,-1]],dtype=np.float32)


kernel2
#bordered_img = cv2.copyMakeBorder(img,1,3,1,3,cv2.BORDER_CONSTANT,0)
bordered_img = cv2.copyMakeBorder(img,3,1,3,1,cv2.BORDER_CONSTANT,0)
cv2.imshow('bordered image',bordered_img)
bordered_img.shape
bordered_img=bordered_img.astype(np.float32)
bordered_img
out
m=1
n=1

flp_a = (5-m-1)
flp_b = (5-n-1)
flp_b
pad_top = flp_a
pad_left = flp_b
pad_bottom=(5-1-flp_a)
pad_right = (5-1-flp_b)

pad_top 
pad_bottom
pad_left
pad_right



for x in range(3,bordered_img.shape[0]-1):
    for y in range(3,bordered_img.shape[1]-1):
        sum=0.0
        for s in range (kernel2.shape[0]):
            a=s-m
            for t in range(kernel2.shape[1]):
                b=t-m
                sum+= bordered_img.item(x-a,y-a)*kernel2.item(s,t)
        out[x-3,y-3]=sum
        

print(out)
cv2.normalize(out,out, 0, 255, cv2.NORM_MINMAX)
print(out)
out= np.round(out).astype(np.uint8)
print(out)

cv2.imshow("classwork input image",img)
cv2.imshow("classwork output image",out)
cv2.waitKey(0) 
cv2.destroyAllWindows()



#cv2.normalize(src,des, 0, 255, cv2.NORM_MINMAX)
#s = np.round(s).astype(np.uint8)
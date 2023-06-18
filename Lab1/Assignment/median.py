# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 02:06:54 2023

@author: ASUS
"""

import numpy as np 
import cv2 
import math

#reading image as gray scale

img=cv2.imread('noise.png',cv2.IMREAD_GRAYSCALE)
out = np.zeros((512,512), dtype=np.float32)
#displaying image
img
cv2.imshow("input image",img)

mask_size = int(input("Enter size of the mask"))
median_index = (mask_size*mask_size)//2 

arr = np.zeros(mask_size*mask_size,dtype=np.uint8)

#assuming centre symmetric
a=mask_size//2
arr
bordered_img =  bordered_img = cv2.copyMakeBorder(img,a,a,a,a,cv2.BORDER_CONSTANT,0)
bordered_img.min()
for x in range (a,bordered_img.shape[0]-a): 
    for y in range(a,bordered_img.shape[1]-a):
        i=0
        for s in range (mask_size):
            m=s-a
            for t in range(mask_size):
                n=t-a
                arr[i]=bordered_img.item(x-m,y-n)
                i+=1
        temp=np.sort(arr)
        
        out[x-a,y-a]=temp[median_index]
        

      

out    


out = np.round(out).astype(np.uint8)
cv2.imshow("output image",out)

cv2.waitKey(0) 
cv2.destroyAllWindows()
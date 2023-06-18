# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 16:10:30 2023

@author: ASUS
"""

import numpy as np 
import cv2 
import math

#reading image as gray scale

img=cv2.imread('building.png',cv2.IMREAD_GRAYSCALE)
outx = np.zeros((img.shape[0],img.shape[1]), dtype=np.float32)
outy = np.zeros((img.shape[0],img.shape[1]), dtype=np.float32)
out = np.zeros((img.shape[0],img.shape[1]), dtype=np.float32)

#displaying image

img
cv2.imshow("input image",img)

def convolution(kernel1,kernel2):
   
    #kernel
   
    kernel_Ht =kernel1.shape[0]
    kernel_Wt=kernel1.shape[1]

    img_Ht = img.shape[0]
    img_Wt = img.shape[1]

    #kernel centre

    a = kernel_Ht//2
    b = kernel_Wt//2

    #centre after flipping 180

    flp_a = (kernel_Ht-a-1)
    flp_b = (kernel_Wt-b-1)
    print(flp_a)


    #padding calculation

    pad_top = flp_a
    pad_left = flp_b
    pad_bottom=(kernel_Ht-1-flp_a)
    pad_right = (kernel_Wt-1-flp_b)

    pad_top 
    pad_bottom
    pad_left
    pad_right

    #bordering image

    bordered_image = cv2.copyMakeBorder(img,pad_top,pad_bottom,pad_left,pad_right,cv2.BORDER_CONSTANT,0)

    for x in range (pad_top,bordered_image.shape[0]-pad_bottom):
        for y in range (pad_left,bordered_image.shape[1]-pad_right):
            sumx=0.0
            sumy=0.0
            for s in range (kernel_Ht):
                m=s-a
                for t in range (kernel_Wt):
                    n=t-b
                    sumx+=bordered_image.item(x-m,y-n)*kernel1.item(s,t)
                    sumy+=bordered_image.item(x-m,y-n)*kernel2.item(s,t)
                    
                    
            outx[x-pad_top,y-pad_left]=sumx
            outy[x-pad_top,y-pad_left]=sumy
            out[x-pad_top,y-pad_left]=np.sqrt(sumx**2+sumy**2)
            



hx = np.array(([1,0,-1],
              [2,0,-2],
              [1,0,-1]),np.float32)

hy = np.array(([-1,-2,-1],
              [0,0,0],
              [1,2,1]),np.float32)

convolution(hx, hy)
cv2.normalize(outx,outx, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(outy,outy, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(out,out, 0, 255, cv2.NORM_MINMAX)
outx = np.round(outx).astype(np.uint8)
outy = np.round(outy).astype(np.uint8)
out = np.round(out).astype(np.uint8)
out.max()
print(out)
out.shape[1]
cv2.imshow("output image x deriv",outx)
cv2.imshow("output image y deriv",outy)
cv2.imshow("output image gradient",out)


cv2.waitKey(0) 
cv2.destroyAllWindows()

    
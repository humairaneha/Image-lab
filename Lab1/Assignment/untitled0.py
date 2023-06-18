# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 23:23:30 2023

@author: ASUS
"""

import cv2
import numpy as np
import math

sigma = float(input("Enter value of sigma"))
img = cv2.imread('rubiks_cube.png',cv2.IMREAD_GRAYSCALE)
out=np.zeros((img.shape[0],img.shape[1]),dtype=np.float32)
k_size =5

if(k_size%2==0):
    k_size+=1

const=2*sigma*sigma
pi=3.14159
gaussfilter=np.zeros((k_size,k_size),dtype=np.float32)

#centre

a=k_size//2
b=k_size//2

#centre after flipping

flp_a=k_size-a-1
flp_b = k_size-b-1

#gaussian filter calculation
norm=1/(pi*const)
for i in range (k_size):
    x=i-a
    for j in range (k_size):
        y=j-b
        term=(x*x)+(y*y)
        term=math.exp(-(term)/const)*norm
        gaussfilter[i,j]=term


k_sum = gaussfilter.sum()

gaussfilter/=k_sum
   


pad_top=flp_a
pad_left=flp_b
pad_bottom=k_size-flp_a-1
pad_right=k_size-flp_b-1

bordered_img = cv2.copyMakeBorder(img,pad_top,pad_bottom,pad_left,pad_right,cv2.BORDER_CONSTANT,0)
bordered_img.shape[1]
for x in range (pad_top,bordered_img.shape[0]-pad_bottom):
    for y in range (pad_left,bordered_img.shape[1]-pad_right):
        cur_px=bordered_img.item(x,y)
        normalize=0
        summ=0.0
        for s in range (k_size):
            m=s-a
            for t in range (k_size):
                n=t-b
                nbr_px=bordered_img.item(x-m,y-m)
                diff=cur_px-nbr_px
                term=gaussfilter[s,t]*(math.exp(-(diff**2)/const))
                #print(term)
                normalize+=term
                summ+=term*nbr_px
        out[x-pad_top,y-pad_left]=summ/normalize
        
        



out/=255
cv2.normalize(out,out, 0, 255, cv2.NORM_MINMAX)
out = np.round(out).astype(np.uint8)
cv2.imshow("input",img)
cv2.imshow("output image x deriv",out)
cv2.waitKey(0) 
cv2.destroyAllWindows()

        
                
                
# =============================================================================
# [224, 229, 246, ..., 234, 237, 250],
#        [251, 239, 230, ..., 228, 239, 224],
#        [237, 242, 251, ..., 217, 244, 250],
#        ...,
#        [247, 251, 234, ..., 235, 239, 200],
#        [240, 238, 227, ..., 230, 215, 237],
#        [221, 240, 206, ..., 241, 214, 235]],
# =============================================================================















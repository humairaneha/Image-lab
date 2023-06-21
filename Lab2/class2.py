# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:06:37 2023

@author: User
"""

import numpy as np
import cv2
import math


img =cv2.imread("cube.png",cv2.IMREAD_GRAYSCALE)
ksize=7
sigma=80
height=img.shape[0]
width=img.shape[1]
output=np.zeros((height,width),dtype='float32')
k=ksize//2
bordered=cv2.copyMakeBorder(img, k,k,k,k,cv2.BORDER_CONSTANT)

def spatial_epanchnikov(ksize, sigma):
    kernel=np.zeros((ksize,ksize),dtype='float32')
    k=ksize//2
    d=ksize//2
    for i in range(-k,k+1):
        for j in range(-k,k+1):
            r=np.sqrt((i)**2+(j)**2)
            kernel[i+k][j+k]=(1-(r/d)**2)
            if(kernel[i+k][j+k]<0):
                kernel[i+k][j+k]=0
    return kernel


def range_gaussian(img, x, y, ksize, sigma):
    kernel=np.zeros((ksize,ksize),dtype='uint8')
    k=ksize//2
    Ip=img[x][y]
    for i in range(-k,k+1):
        for j in range(-k,k+1):
            Iq=img[x+i][y+j]
            kernel[i+k][j+k]=math.exp(-((int(Ip)-int(Iq))**2)/(2*sigma*sigma))
    return kernel

def multiply_kernels(sp_kernel, rng_kernel, ksize):
    final_kernel=np.zeros_like(img)
    for i in range(ksize):
        for j in range(ksize):
            final_kernel[i][j]=sp_kernel[i][j]*rng_kernel[i][j]
    
    sumfull=final_kernel.sum()
    final_kernel=final_kernel/sumfull
    return final_kernel


sp_kernel=spatial_epanchnikov(ksize, sigma)
sp_kernel1= cv2.resize(sp_kernel, None, fx = 50, fy = 50, interpolation = cv2.INTER_NEAREST)

for x in range(height):
    for y in range(width):
        rng_kernel=range_gaussian(bordered, x+k, y+k, ksize, sigma)
        kernel=multiply_kernels(sp_kernel, rng_kernel, ksize)
        for i in range(ksize):
            for j in range(ksize):
                output[x][y]+= bordered[x+i][y+j]*kernel[ksize-1-i][ksize-1-j]

#output=output*255
cv2.normalize(output,output,0,255,cv2.NORM_MINMAX)
output=np.round(output).astype(np.uint8)  

cv2.imshow("Outpout",output)
cv2.imshow("Input", img)
cv2.imshow("Spatial kernel",sp_kernel1)
cv2.waitKey(0)
cv2.destroyAllWindows()


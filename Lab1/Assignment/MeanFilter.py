# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 20:51:31 2023

@author: ASUS
"""



import numpy as np 
import cv2 
import math

#reading image as gray scale

img1=cv2.imread('Lena.jpg',cv2.IMREAD_GRAYSCALE)

#displaying image


cv2.imshow("input image",img1)

def convolution(kernel,img):
    out = np.zeros((img.shape[0],img.shape[1]), dtype=np.float32)
    #kernel
   
    kernel_Ht =kernel.shape[0]
    kernel_Wt=kernel.shape[1]

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
            sum=0.0
            for s in range (kernel_Ht):
                m=s-a
                for t in range (kernel_Wt):
                    n=t-b
                    sum+=bordered_image.item(x-m,y-n)*kernel.item(s,t)
                    
            out[x-pad_top,y-pad_left]=sum
    return out
    


#Mean or Averaging filter

meanKernel =(1/9)* np.array([[1,1,1],
              [1,1,1],
              [1,1,1]],dtype=np.float32)



meanout=convolution(meanKernel,img1)


#gaussian filter

#choosing sigma
sigma = float(input("Enter value of sigma:"))

#kernel size

k_size = int(5*sigma)

if(k_size%2==0):
    k_size+=1
#normalization constant
norm_const = 1/(2*3.141592*sigma*sigma)
norm_const

gaussianKernel = np.zeros((k_size,k_size),dtype=np.float32)
gaussianKernel
a=k_size//2
b=k_size//2
b

#eqn: (1/2pi(sigma^2))*exp^-(x^2+y^2)/2*sigma^2

for x in range (k_size):
    for y in range (k_size):
        term = (x-a)*(x-a) + (y-b)*(y-b)
        term/=(2*sigma*sigma)
        term = math.exp(-(term))
        gaussianKernel[x,y]= term*norm_const
        
        
ker_sum = gaussianKernel.sum()

#normalizing the filter so that the sum of the elements are equal to 1. 
#if the sum >1 then image will get brighter after blurring,if sum<1 image will get darker. 
# Normalization ensures that the average greylevel of the image remains the same after blurring

gaussianKernel/=ker_sum
img2=cv2.imread('noise.png',cv2.IMREAD_GRAYSCALE)
gaussout=convolution(gaussianKernel,img1)



laplacian = np.array(([0,-1,0],[-1,4,-1],[0,-1,0]),np.float32)

img3=cv2.imread('moon.png',cv2.IMREAD_GRAYSCALE)

laplacianout=convolution(laplacian,img3)
laplacianout+=img3


cv2.normalize(laplacianout,laplacianout, 0, 255, cv2.NORM_MINMAX)
laplacianout = np.round(laplacianout).astype(np.uint8)
 
cv2.imshow("inout image for laplacian",img3)
cv2.imshow("output image for laplacian",laplacianout)        


#cv2.normalize(gaussout,gaussout, 0, 255, cv2.NORM_MINMAX)
gaussout = np.round(gaussout).astype(np.uint8)

cv2.imshow("inout image for gaussian",img1)
cv2.imshow("output image for gaussian",gaussout)

meanout = np.round(meanout).astype(np.uint8)

cv2.imshow("input image for mean",img1)
cv2.imshow("output image for mean",meanout)


cv2.waitKey(0) 
cv2.destroyAllWindows()


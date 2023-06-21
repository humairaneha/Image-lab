# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 16:35:22 2023

@author: User
"""

import cv2 as cv
import numpy as np

def twirl(img,xc,yc,alpha,r_max):
    height=img.shape[0]
    width=img.shape[1]
    transformed=np.copy(img)
    
    for x_prime in range(img.shape[0]):
        for y_prime in range(img.shape[1]):
            dx=x_prime-xc
            dy=y_prime-yc
            r=np.sqrt(dx**2+dy**2)
            theta=np.arctan2(dx,dy) #dy/dx
            
            # print('hello')
            
            if r<r_max:
                newrad=(r_max-r)*alpha/r_max
                beta=theta+newrad
                x=int(xc+r*np.cos(beta))
                y=int(yc+r*np.sin(beta))
            
                if 0 <= x < height - 1 and 0 <= y < width - 1:
                    j=int(x)
                    k=int(y)
                    # j1=j+1
                    # k1=k+1
                
                    a=abs(x-j)
                    b=abs(y-k)
                        
                    transformed[x_prime,y_prime]=(1-b)*(1-a)*img[j,k]+(1-b)*a*img[j+1,k]+\
                    b*(1-a)*img[j,k+1]+b*a*img[j+1,k+1]
    return transformed
    
def angular(img,xc,yc,amplitude,frequency):
    height=img.shape[0]
    width=img.shape[1]
    transformed=np.copy(img)
    for x_prime in range(img.shape[0]):
        for y_prime in range(img.shape[1]):
            dx=x_prime-xc
            dy=y_prime-yc
            r=np.sqrt(dx**2+dy**2)
            theta=np.arctan2(dx,dy) #dy/dx
            
            if True:
                displacement=amplitude*np.sin((2*np.pi*r)/frequency)
                beta=theta+displacement
                x=int(xc+r*np.cos(beta))
                y=int(yc+r*np.sin(beta))
            
                if 0 <= x < height - 1 and 0 <= y < width - 1:
                    j=int(x)
                    k=int(y)
                    # j1=j+1
                    # k1=k+1
                
                    a=abs(x-j)
                    b=abs(y-k)
                        
                    transformed[x_prime,y_prime]=(1-b)*(1-a)*img[j,k]+(1-b)*a*img[j+1,k]+\
                    b*(1-a)*img[j,k+1]+b*a*img[j+1,k+1]
    return transformed


def ripple(img_rp,amplitude_x,amplitude_y,frequency_x,frequency_y):
    height=img_rp.shape[0]
    width=img_rp.shape[1]
    transformed=np.zeros_like(img_rp)
    for x_prime in range(img_rp.shape[0]):
        for y_prime in range(img_rp.shape[1]):
            x=x_prime+amplitude_x*np.sin((2*np.pi*y_prime)/frequency_x)
            y=y_prime+amplitude_y*np.sin((2*np.pi*x_prime)/frequency_y)
            
            if 0 <= x < height - 1 and 0 <= y < width - 1:
                j=int(x)
                k=int(y)
                # j1=j+1
                # k1=k+1
            
                a=abs(x-j)
                b=abs(y-k)
                    
                transformed[x_prime,y_prime]=(1-b)*(1-a)*img_rp[j,k]+(1-b)*a*img_rp[j+1,k]+\
                b*(1-a)*img_rp[j,k+1]+b*a*img_rp[j+1,k+1]
    return transformed


def tapestry(img,x_c,y_c,amplitude_x,amplitude_y,frequency_x,frequency_y):
    height=img.shape[0]
    width=img.shape[1]
    transformed=np.zeros_like(img)
    for x_prime in range(img.shape[0]):
        for y_prime in range(img.shape[1]):
            x=x_prime+amplitude_x*np.sin((2*np.pi*(x_prime-x_c))/frequency_x)
            y=y_prime+amplitude_y*np.sin((2*np.pi*(y_prime-y_c))/frequency_y)
            
            if 0 <= x < height - 1 and 0 <= y < width - 1:
                j=int(x)
                k=int(y)
                # j1=j+1
                # k1=k+1
            
                a=abs(x-j)
                b=abs(y-k)
                    
                transformed[x_prime,y_prime]=(1-b)*(1-a)*img[j,k]+(1-b)*a*img[j+1,k]+\
                b*(1-a)*img[j,k+1]+b*a*img[j+1,k+1]
    return transformed


img=cv.imread('twirl.jpg')
img_rp=cv.imread('tap.png')
img_tp=cv.imread('tap.png')

alpha=int(input('Enter alpha: '))
alpha=alpha
alpha=np.deg2rad(alpha)
xc = img_tp.shape[0] // 2
yc= img_tp.shape[1] // 2
r_max = min(xc,yc)- 10
alpha = np.deg2rad(90)
amplitude=0.1
frequency=50
out1=twirl(img,xc,yc,alpha,r_max)
out2=angular(img,xc,yc,amplitude,frequency)
out3=angular(img,xc,yc,0.08,50)
out4=ripple(img_rp,10,15,50,70) #ripple(img,amplitude_x=10,amplitude_y=15,frequency_x=50,frequency_y=70)
out5=tapestry(img_tp,xc,yc,5,5,30,30) #tapestry(img,x_c,y_c,amplitude_x,amplitude_y,frequency_x,frequency_y)
cv.imshow("Twirl Output",out1)
cv.imshow("Angular outut", out2)
cv.imshow("Angular output lower amplitude", out3)
cv.imshow("Ripple",out4)
cv.imshow("Tapestry",out5)
cv.waitKey(0)
cv.destroyAllWindows()
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

img=cv2.imread("twirl.jpg",cv2.IMREAD_GRAYSCALE)
result=np.zeros((img.shape),dtype=np.float32)
xc=img.shape[0]//2
yc=img.shape[1]//2
alp=int(input("Enter the angle\n"))
alp=np.radians(alp)
r_max=min(xc,yc)

for i in range(result.shape[0]):
  for j in range(result.shape[1]):
    dx=i-xc
    dy=j-yc
    r=np.sqrt((dx*dx)+(dy*dy))
    beta=np.arctan2(dy,dx)+(alp*((r_max-r)/r_max))
    if(r<=r_max):
        x=xc+r*np.cos(beta)
        y=yc+r*np.sin(beta)
    else: 
        x=i
        y=j
    
    jj=int(np.floor(x))
    kk=int(np.floor(y))
    a=x-jj
    b=y-kk
    if(jj<img.shape[0]-1 and kk<img.shape[1]-1):
        arr=[[img[jj][kk],img[jj+1][kk]],
             [img[jj][kk+1],img[jj+1][kk+1]]]
        temp2=[[1-b,b]]
        temp1=[[1-a],
               [a]]
        ans=np.matmul(arr,temp1)
        ans=np.matmul(temp2,ans)
        ans=ans[0][0]
    else:
        ans=0
        
    result[i][j]=ans


cv2.imshow("out",result)
cv2.waitKey(0)
    
    

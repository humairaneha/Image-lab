import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("cube.png",0)

sigma = 100

k_size = 7

centre = k_size//2

padding = k_size//2

bordered_img = cv2.copyMakeBorder(img,padding,padding,padding,padding,cv2.BORDER_CONSTANT,0)

kernel = np.zeros((k_size,k_size),np.float32)

def epanech_filter(k_size,centre):
    
 
    d=k_size/2
    xc=0
    yc=0
    for i in range (k_size):
        x=i-centre
        for j in range (k_size):
            y=j-centre
            
            r=np.sqrt((x-xc)**2 + (y-yc)**2)
            
            kernel[i][j] = 1 - (r/d)**2
            if(kernel[i][j]<0):
                kernel[i][j]=0
            
            
    


epanech_filter(k_size, centre)


def range_filter(sigma,diff):
    
    return np.exp(-(((diff)**2)/(2*sigma*sigma)))



out = np.zeros((img.shape[0],img.shape[1]),np.float32)
for i in range (padding,bordered_img.shape[0]-padding):
    
    for j in range(padding,bordered_img.shape[1]-padding):
        sum_bilat=0.0
        for x in range (k_size):
            s=x-centre
            for y in range(k_size):
                t=y-centre
                ip = int(bordered_img[i,j])
                iq = int(bordered_img[i-s,j-t])
                diff = ip-iq
                bilateral = kernel[x,y]*range_filter(sigma, diff)
                sum_bilat+=bilateral
                out[i-padding,j-padding]+= bordered_img[i,j]*bilateral
        
        out[i-padding,j-padding]/=sum_bilat
        out[i-padding,j-padding]/=255


#cv2.normalize(out,out,0,1,cv2.NORM_MINMAX) 

out1 = np.zeros((img.shape[0],img.shape[1]),np.float32)
#cv2.bilateralFiter(out1,out1,20,80,BORDER_DEFAULT)
 
cv2.imshow("input",img)
cv2.imshow("output",out)  
cv2.waitKey(0)
cv2.destroyAllWindows()    
      

        
        
         





































import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

img=cv2.imread("tapestry.jpg")



def tapestry(out,ax,ay,Tx,Ty):
 output = np.zeros((out.shape[0],out.shape[1]),np.uint8)
 xc=out.shape[0]//2
 yc=out.shape[1]//2
 for i in range (out.shape[1]): # column or width or x axis  wise 
    for j in range(out.shape[0]): #row or height or y axis wise 
        
        x = i + ax* np.sin((2*np.pi*(i-xc))/Tx) 
        y = j + ay* np.sin((2*np.pi*(j-yc))/Ty) 
        
        # four surrounding pixels
        #top left boundary
        x1 = int(x)
        y1 = int(y)
       
        #bottom right boundary
        
        x2 = x1 + 1
        y2 = y1 + 1
        
        if x1 in range (0,out.shape[1]) and y1 in range (0,out.shape[0]) and x2 in range (0,out.shape[1]) and y2 in range (0,out.shape[0]) :
            
            a =  x - x1
            b= y-y1
            
            top_left = out[y1,x1]
            top_right = out[y1,x2]
            bottom_left = out[y2,x1]
            bottom_right = out[y2,x2]
            
            mat_a = np.array([[1-a],[a]])
            mat_b = np.array([1-b,
                     b])
            mat_pixels = np.array([[top_left,top_right],
                          [bottom_left,bottom_right]])
            
            
            new = np.dot(mat_pixels,mat_a)
            new_intensity = np.dot(mat_b,new)
            
            output[j][i] = new_intensity.astype(np.uint8)
        
 return output



b,g,r = cv2.split(img)
out_b = tapestry(b,5,5,30,30)
out_g = tapestry(g,5,5,30,30)
out_r = tapestry(r,5,5,30,30)

out1=cv2.merge([out_b,out_g,out_r])
cv2.imshow('tapestry Effect 1', out1  )


# # b,g,r = cv2.split(img)
# # out_b = ripple(b,10,10,20,20)
# # out_g = ripple(g,10,10,20,20)
# # out_r = ripple(r,10,10,20,20)

# # out2=cv2.merge([out_b,out_g,out_r])
        
# cv2.imshow('Original Image', img)
# cv2.imshow('Ripple Effect 2', out2  )

cv2.waitKey(0)
cv2.destroyAllWindows()       
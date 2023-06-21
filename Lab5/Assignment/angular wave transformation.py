import cv2
import matplotlib.pyplot as plt
import math
import numpy as np



image=cv2.imread('twirl.jpg',cv2.IMREAD_GRAYSCALE)
# plt.imshow(image,'gray')
# plt.show()

result=np.zeros((image.shape),np.float32)
center_x=image.shape[0]//2
center_y=image.shape[1]//2
# a=float(input("input a:"))
# tau=float(input("input tau:"))


a=0.05
tau=50

for i in range (0,result.shape[0]):
    for j in range(0,result.shape[1]):
        dx=i-center_x
        dy=j-center_y
        r=np.sqrt(pow(dx,2)+pow(dy,2))
        r_max=min(center_x,center_y)
        b = math.atan2(dy, dx) + a * np.sin((2 * np.pi * r) / tau)
        xn=center_x+r*np.cos(b)
        yn=center_y+r*np.sin(b)
        y=int(np.floor(yn))
        x=int(np.floor(xn))
        aa=xn-x
        bb=yn-y
        if(x<image.shape[0]-1 and y<image.shape[1]-1):
            image_array=[[image[x][y],image[x+1][y]]
                        ,[image[x][y+1],image[x+1][y+1]]]
            temp1=[[1-aa],
                    [aa]]
            temp2=[[1-bb,bb]]
            ans=np.matmul(image_array,temp1)
            ans=np.matmul(temp2,ans)
            ans=ans[0][0]
        else:
            ans=0
        result[i][j]=ans

cv2.imshow("output",result)
cv2.waitKey(0)

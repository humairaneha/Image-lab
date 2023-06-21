import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

img = cv2.imread("input.jpg",0)

X=np.array([[0,0,0],[1,1,0],[1,0,0]],np.uint8)
r,img= cv2.threshold(img,130,255,cv2.THRESH_BINARY)
cv2.imshow("input",img)

def hitormiss(A,Ac,W,X):
    erode1=cv2.erode(A,X,iterations=1)
    W=cv2.bitwise_not(X)
    erode2=cv2.erode(Ac,W,iterations=1)
    out = cv2.bitwise_and(erode1,erode2)
    return out

rate =50

X=cv2.resize(X,None,fx=rate,fy=rate,interpolation=cv2.INTER_NEAREST)
X*=255
kernel = np.ones((3,3),np.uint8)
W = cv2.dilate(X,kernel,iterations=1)
A=img
Ac=255-img
out=hitormiss(A, Ac, W, X)


cv2.imshow("Se",X)

cv2.imshow("out",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
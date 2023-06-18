import cv2
import numpy as np

def hitormiss(A,W,X,Ac):
    out1= np.zeros((400,400),np.uint8)
    erode1 = cv2.erode(A,X,iterations = 1)
    
    
    W=cv2.bitwise_not(X)
    cv2.imshow("window",W)
    erode2= cv2.erode(Ac,W,iterations = 1)
    
    
    out1 = cv2.bitwise_and(erode1,erode2)
    return out1
    

    
img = cv2.imread("input.jpg", 0)

r, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)#_INV)

out = np.zeros((img.shape[0],img.shape[1]),np.uint8)
cv2.imshow("Original", img)


X1 = np.array(([0,0,0],
               [1,1,0],
               [1,0,0]),np.uint8)
X2 = np.array(([0,1,1],
               [0,0,1],
               [0,0,1]),np.uint8)
X3 = np.array(([1,1,1],
               [0,1,0],
               [0,1,0]),np.uint8)

W =np.array(([1,1,1],
               [1,1,1],
               [1,1,1]),np.uint8)


rate = 50
X1 = cv2.resize(X1, None, fx = rate, fy = rate, interpolation = cv2.INTER_NEAREST)

X2 = cv2.resize(X2, None, fx = rate, fy = rate, interpolation = cv2.INTER_NEAREST)

X3 = cv2.resize(X3, None, fx = rate, fy = rate, interpolation = cv2.INTER_NEAREST)

W = cv2.resize(W, None, fx = rate, fy = rate, interpolation = cv2.INTER_NEAREST)



# X1 hit or miss 

A=img
Ac = 255-A

#out1= hitormiss(A, 255*W, 255*X1, Ac)
#out2=hitormiss(A, 255*W, 255*X2, Ac)
out3=hitormiss(A, 255*W, 255*X3, Ac)



cv2.imshow("SE X1",255*X1)
#cv2.imshow("output for X1",out1)

cv2.imshow("SE X2",255*X2)
#cv2.imshow("output for X2",out2)

cv2.imshow("SE X3",255*X3)
cv2.imshow("output for X3",out3)

cv2.imshow(" W",255*W)


cv2.waitKey(0)
cv2.destroyAllWindows()
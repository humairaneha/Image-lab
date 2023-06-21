import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


img = cv2.imread("twirl.jpg")

xc = img.shape[1]//2
yc = img.shape[0]//2
#alpha= int(input("ENTER ANGLE"))
#alpha = np.deg2rad(alpha)
rmax = min(xc,yc)
def twirl(inp,xc,yc,alpha,rmax):
    out = np.zeros_like(inp) 
    for i in range(inp.shape[1]):
        for j in range(inp.shape[0]):
            
            dx= i - xc
            dy=j-yc
            r=np.sqrt(dx**2+dy**2)
            beta = np.arctan2(dy,dx) + alpha*((rmax-r)/rmax)
            
            if r<=rmax:
                x=xc+r*np.cos(beta)
                y=yc+r*np.sin(beta)
            else:
                x=i
                y=j
            
            #bilinear interop
            
            x1=int(x)
            y1=int(y)
            
            x2 = x1+1
            y2= y1+1
            
            if x1 in range(0,inp.shape[1]) and x2 in range(0,inp.shape[1]) and y1 in range(0,inp.shape[0]) and y2 in range(0,inp.shape[0]):
                
                tl = inp[y1,x1]
                tr = inp[y1,x2]
                bl= inp[y2,x1]
                br=inp[y2,x2]
                
                a=x-x1
                b=y-y1
                
                mat_a = np.array([[1-a],[a]])
                mat_b = np.array([[1-b,b]])
                mat_px = np.array([[tl,tr],[bl,br]])
                
                new = np.dot(mat_px,mat_a)
                
                new_intensity= np.dot(mat_b,new)
                out[j,i] = new_intensity.astype(np.uint8)
                
    return out      
def angular(inp,xc,yc,a,T):
    out = np.zeros_like(inp) 
    for i in range(inp.shape[1]):
        for j in range(inp.shape[0]):
            a=0.3
            T=50
            dx= i - xc
            dy=j-yc
            r=np.sqrt(dx**2+dy**2)
            term=np.sin((2*np.pi*r)/T)
            dp= a*term
            beta = (math.atan2(dy,dx))+dp 
            
            
            x=xc+(r*np.cos(beta))
            y=yc+(r*np.sin(beta))
            
            
            #bilinear interop
            
            x1=int(x)
            y1=int(y)
            
            x2 = x1+1
            y2= y1+1
            
            if x1 in range(0,inp.shape[1]) and x2 in range(0,inp.shape[1]) and y1 in range(0,inp.shape[0]) and y2 in range(0,inp.shape[0]):
                
                tl = inp[y1,x1]
                tr = inp[y1,x2]
                bl= inp[y2,x1]
                br=inp[y2,x2]
                
                a=x-x1
                b=y-y1
                
                mat_a = np.array([[1-a],[a]])
                mat_b = np.array([[1-b,b]])
                mat_px = np.array([[tl,tr],[bl,br]])
                
                new = np.dot(mat_px,mat_a)
                
                new_intensity= np.dot(mat_b,new)
                out[j,i] = new_intensity.astype(np.uint8)
                
    return out              

b,g,r = cv2.split(img)
#out_b = twirl(b,xc,yc,alpha,rmax)
#out_g = twirl(g,xc,yc,alpha,rmax)
#out_r = twirl(r,xc,yc,alpha,rmax)

a=0.1
T=50
out_b1 =angular(b,xc,yc,a,T)
out_g1 =angular(g,xc,yc,a,T)
out_r1 =angular(r,xc,yc,a,T)
output1 = cv2.merge([out_b1,out_g1,out_r1])
#output = cv2.merge([out_b,out_g,out_r])
cv2.imshow("input",img)
#cv2.imshow("output",output)
cv2.imshow("output angular",output1)
cv2.waitKey(0)
cv2.destroyAllWindows()















            
            
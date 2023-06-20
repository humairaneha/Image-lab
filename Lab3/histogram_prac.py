import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("bd.jpeg")

b,g,r = cv2.split(img)
freq = np.zeros((256),np.int32)
pdf = np.zeros((256),np.float32)
cdf = np.zeros((256),np.float32)
def equalize(inp):
    out = np.zeros((inp.shape[0],inp.shape[1]),np.float32)
    for i in range (inp.shape[0]):
        for j in range(inp.shape[1]):
            intensity = inp[i,j]
            freq[intensity]= freq[intensity]+1
    
    
    
    size=float(inp.shape[0]*inp.shape[1])
    #print(size)
    pdf = freq/size
    #print(pdf)
   
    cdf[0]=pdf[0]
    
    for i in range(1,256):
        cdf[i]=cdf[i-1]+pdf[i]
    
    print(cdf.sum())
    
    for i in range (inp.shape[0]):
        for j in range(inp.shape[1]):
            
            intensity = inp[i,j]
            
            newIn = 255*cdf[intensity]
            out[i,j]=np.round(newIn)
    
    return out


out_b = equalize(b)
out_g = equalize(g)
out_r = equalize(r)

out = cv2.merge((out_b,out_g,out_r))

plt.subplot(3,2,1)
plt.title("histogram of individual channel")
hist_b,_ = np.histogram(img[:,:,0],256,[0,256])
plt.plot(hist_b,color='b')
hist_g,_ = np.histogram(img[:,:,1],256,[0,256])
plt.plot(hist_g,color='g')
hist_r,_ = np.histogram(img[:,:,2],256,[0,256])
plt.plot(hist_r,color='r')

plt.subplot(3,2,2)
plt.title("equalized histogram of individual channel")
ohist_b,_ = np.histogram(out[:,:,0],256,[0,256])
plt.plot(ohist_b,color='b')
ohist_g,_ = np.histogram(out[:,:,1],256,[0,256])
plt.plot(ohist_g,color='g')
ohist_r,_ = np.histogram(out[:,:,2],256,[0,256])
plt.plot(ohist_r,color='r')

plt.subplot(3,2,3)
plt.title("histogram of img")
hist = cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(hist,color='b')    

plt.subplot(3,2,4)
plt.title("equalized histogram of img")
hisb = cv2.calcHist([out],[0],None,[256],[0,256])
plt.plot(hisb,color='b')          
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

equalize(b)
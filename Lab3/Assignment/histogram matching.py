import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("histogram.jpg",0)
cv2.imshow("input image",img)
cdf = np.zeros(256,dtype=np.float32) 

def make_pdf(freq,size):
    pdf = np.zeros(256,dtype=np.float32) 
    pdf=freq/size
    return pdf

def make_cdf(pdf):
    cdf[0]=pdf[0]
    for i in range (1,256):
        cdf[i]=cdf[i-1]+pdf[i]
    return cdf
outimg = np.zeros((img.shape[0],img.shape[1]),np.uint8)
matchemg=np.zeros((img.shape[0],img.shape[1]),np.uint8)
def equalize (img):
    
    freq = np.zeros(256,dtype=np.int32)
    
    for i in range(img.shape[0]):
        for j in range (img.shape[1]):
            freq[img[i,j]] +=1
    print(freq)
    
    size = img.shape[0]*img.shape[1]
    
    pdf = make_pdf(freq, size)
    
    cdf=make_cdf(pdf)
    
    S=np.round(255*cdf)
  
     
    return S
    

def gaussian(miu,sigma):
    
    g = np.zeros(256,dtype=np.float32)
    variance = sigma*sigma
    constant = 1/(np.sqrt(2*3.1416)*sigma)
    
    for i in range (256):
        g[i]=np.exp(-((i-miu)**2)/(2*variance))*constant
    
    return g

out = equalize(img) 

for i in range(img.shape[0]):
    for j in range (img.shape[1]):
         k=img[i,j]
         outimg[i,j]=np.round(out[k])

cv2.imshow("output image after histogram equalization",outimg)

plt.subplot(3,2,1) 
plt.title("input image histogram")
hist = cv2.calcHist([img],[0], None, [256],[0,255])
histout = cv2.calcHist([outimg],[0], None, [256],[0,255])
plt.plot(hist,c="b")
plt.plot(histout,c="r")

plt.subplot(3,2,2) 
plt.title("cdf of equalized histogram")
plt.plot(cdf)

gauss1=gaussian(30, 10)
gauss2= gaussian(60, 20)

plt.subplot(3,2,3) 
plt.title("gaussian function")
plt.plot(gauss1,c="b")

plt.plot(gauss2,c="r")

doublegauss = gauss1+gauss2


size=sum(doublegauss)
pdf_matching = make_pdf(doublegauss,size)
cdf_matching = make_cdf(pdf_matching)

eq_matching = np.zeros(256,np.uint8)

for i in range (256):
    eq_matching[i]=np.round(255*cdf_matching[i])


def search(a,arr):
    for i in range(len(arr)):
        if(a==arr[i]):
            return i
        elif (a<arr[i]):
            prev=arr[i-1]
            cur=arr[i]
            if((a-prev)<(cur-a)):
                return i-1
            else:
                return i
    return 255
          
def map (out,eq_matching):
    map_out = np.zeros((256),np.uint8)
    for i in range (256):
        
            intensity = out[i]
            index = search(intensity,eq_matching)
            map_out[i]=index
    return map_out      

map_out = map(out,eq_matching)
output = np.zeros((img.shape[0],img.shape[1]),np.uint8)
for i in range(img.shape[0]):
     for j in range (img.shape[1]):
         k=img[i,j]
         output[i,j]=map_out[k]
maphistout = cv2.calcHist([output],[0], None, [256],[0,255])
cv2.imshow("histogram matching output image",output)
plt.subplot(3,2,4) 
plt.title("double gaussian")
plt.plot(doublegauss)

plt.subplot(3,2,5) 
plt.title("cdf of double gaussian ")
plt.plot(cdf_matching)
plt.subplot(3,2,6) 
plt.title("histogram matching output ")
plt.plot(maphistout)

cv2.waitforkey(0)
cv2.destroyAllWindows()

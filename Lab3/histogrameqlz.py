import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('bd.jpeg')

b,g,r=cv2.split(img)

img_ht=img.shape[0]
img_wt= img.shape[1]
size= img_ht*img_wt

b.shape[0]
b.shape[1]
#histb = cv2.calcHist([img],[1], None, [256],[0,255])
#histg = cv2.calcHist([img],[2], None, [256],[0,255])
#histr = cv2.calcHist([img],[3], None, [256],[0,255])

plt.subplot(3, 2, 1)
plt.title("input channel histogram")
histr, _ = np.histogram(img[:,:,2],256,[0,256])
plt.plot(histr,color = 'r')
histg, _ = np.histogram(img[:,:,1],256,[0,256])
plt.plot(histg,color = 'g')
histb, _ = np.histogram(img[:,:,0],256,[0,256])
plt.plot(histb,color = 'b')



pmf_b = np.zeros((256),dtype=np.float32)
pmf_g = np.zeros((256),dtype=np.float32)
pmf_r = np.zeros((256),dtype=np.float32)

cmf_b = np.zeros((256),dtype=np.float32)
cmf_g = np.zeros((256),dtype=np.float32)
cmf_r = np.zeros((256),dtype=np.float32)

b_freq = np.zeros((256),dtype=np.int32)
g_freq = np.zeros((256),dtype=np.int32)
r_freq = np.zeros((256),dtype=np.int32)

out_b = np.zeros((img_ht,img_wt),np.uint8)
out_g= np.zeros((img_ht,img_wt),np.uint8)
out_r= np.zeros((img_ht,img_wt),np.uint8)
out_v= np.zeros((img_ht,img_wt),np.uint8)

for i in range (img_ht):
    for j in range (img_wt):
        intensity_b=b[i,j]
        b_freq[intensity_b]+=1
        intensity_g=g[i,j]
        g_freq[intensity_g]+=1 
        intensity_r=r[i,j]
        r_freq[intensity_r]+=1  
            

pmf_b = b_freq/size
pmf_g=g_freq/size
pmf_r = r_freq/size

cmf_b[0]=pmf_b[0]
cmf_g[0]=pmf_g[0]
cmf_r[0]=pmf_r[0]

for i in range (1,256):
        cmf_b[i]=cmf_b[i-1]+pmf_b[i]
        cmf_g[i]=cmf_g[i-1]+pmf_g[i]
        cmf_r[i]=cmf_r[i-1]+pmf_r[i]
        

for i in range (img_ht):
    for j in range (img_wt):
        intensity_b=b[i,j]
        out_b[i,j]=np.round(255*cmf_b[intensity_b])
        intensity_g=g[i,j]
        out_g[i,j]=np.round(255*cmf_g[intensity_g])
        intensity_r=r[i,j]
        out_r[i,j]=np.round(255*cmf_r[intensity_r])
        

output = cv2.merge((out_b,out_g,out_r))
plt.subplot(3, 2, 2)
plt.title("equalized channel histogram")
histr, _ = np.histogram(output[:,:,2],256,[0,256])
plt.plot(histr,color = 'r')
histg, _ = np.histogram(output[:,:,1],256,[0,256])
plt.plot(histg,color = 'g')
histb, _ = np.histogram(output[:,:,0],256,[0,256])
plt.plot(histb,color = 'b')
cv2.imshow("input image",img)
cv2.imshow("equalized  image",output)

img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

plt.subplot(3, 2, 3)
histr, _ = np.histogram(img_hsv[:,:,2],256,[0,256])
plt.title("Hsv model v channel")
plt.plot(histr,color = 'b')

h,s,v = cv2.split(img_hsv)

pmf_v = np.zeros((256),dtype=np.float32)
cmf_v = np.zeros((256),dtype=np.float32)
v_freq = np.zeros((256),dtype=np.int32)
for i in range (img_ht):
    for j in range (img_wt):
        intensity_v=v[i,j]
        v_freq[intensity_v]+=1


pmf_v = v_freq/size

cmf_v[0]=pmf_v[0]

for i in range (1,256):
        cmf_v[i]=cmf_v[i-1]+pmf_v[i]
        
plt.subplot(3, 2, 4)
plt.title("cdf of v channel")
plt.plot(cmf_v,color = 'b') 
         
for i in range (img_ht):
    for j in range (img_wt):
        intensity_v=v[i,j]
        out_v[i,j]=np.round(255*cmf_v[intensity_v])

plt.subplot(3, 2, 5)
histr, _ = np.histogram([out_v],256,[0,256])
plt.title("Hsv model equalized v channel")
plt.plot(histr,color = 'b')

img_hsv = cv2.merge((h,s,out_v))
image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)


cv2.imshow(" hsv equalized  image",image)

cv2.imshow("i/p blue channel",b)
cv2.imshow("i/p green channel",g)
cv2.imshow("i/p red channel",r)

cv2.imshow("eq blue channel",out_b)
cv2.imshow("eq green channel",out_g)
cv2.imshow("eq red channel",out_r)


cv2.waitKey(0)
cv2.destroyAllWindows()




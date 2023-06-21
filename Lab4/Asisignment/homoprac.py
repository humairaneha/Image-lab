import numpy as np
import matplotlib
import cv2
import math



img=cv2.imread('unnamed.png',0)


def make_pattern(center1,center2,radius1,radius2):
    pattern = np.zeros_like(img)
    for i in range (0,img.shape[0]):
        for j in range (0,img.shape[1]):
            
            dis = np.sqrt((i-center1[0])**2+(j-center1[1])**2)
            pattern[i,j] = 0 + 255* (np.exp(-(dis)/radius1)**2)
            
     
    for i in range (img.shape[0]-1,radius2+1,-1):
        for j in range (img.shape[1]-1,radius2+1,-1):
                
                dis = np.sqrt((i-center2[0])**2+(j-center2[1])**2)
                pattern[i,j] = 0+ 255* (np.exp(-(dis)/radius2)**2)
    return pattern



center1 = (0,0)
center2 =(img.shape[0]-1,img.shape[1]-1)
radius1 = img.shape[0]//4
radius2 = img.shape[1]//3


pattern = make_pattern(center1,center2,radius1,radius2)

pattern = np.clip(pattern,0,255)
angle=90     
angle = np.deg2rad(angle)

x = np.linspace(1,0,img.shape[1])
y = np.linspace(1,0,img.shape[0])
xx, yy = np.meshgrid(x, y)


grad_dir = np.array([np.cos(angle), np.sin(angle)])


illum_pattern = grad_dir[0] * xx + grad_dir[1] * yy

illum_pattern/= illum_pattern.max()
#corrupt_img = cv2.add(pattern,img)
corrupt_img = np.multiply(illum_pattern,img)
corrupt_img/=255
cv2.imshow("pattern",pattern)

cv2.imshow("corrupted img",corrupt_img)

#logtransform

corrupt_log = np.log1p(corrupt_img)

ft = np.fft.fft2(corrupt_log)

ft_shift = np.fft.fftshift(ft)

mag = np.abs(ft_shift)

ang = np.angle(ft_shift)


def HomoFilter(mag,gh,gl,cutoff,d0):
      cx=mag.shape[0]//2
      cy=mag.shape[1]//2
      kernel =  np.ones_like(mag)
      for u in range(kernel.shape[0]):
          for v in range(kernel.shape[1]):
              
              dis = np.sqrt((u-cx)**2+(v-cy)**2)
              term =(1 - np.exp(-(dis/d0)**2))*cutoff
              term = (gh-gl)*term +gl
              kernel[u,v]=term
      return kernel
  
gh=1.3
gl=0.5
cutoff=0.1
d0=50


#illum_pattern -= illum_pattern.min()
#illum_pattern /= illum_pattern.max()

cv2.imshow("restored ",illum_pattern)    

H = HomoFilter(mag, gh, gl, cutoff, d0)

output = mag*H

result = np.multiply(output,np.exp(1j*ang))

result = np.real(np.fft.ifft2(np.fft.ifftshift(result)))

final_out = np.expm1(result)

final_out = (final_out - np.min(final_out))*(255/(np.max(final_out))-np.min(final_out))

final_out= final_out.astype(np.uint8)

              
cv2.imshow("restored img",final_out)             
          













cv2.waitKey(0)
cv2.destroyAllWindows()
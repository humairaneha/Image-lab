import cv2
import numpy as np

def generate_illuminative_circular_gradient(image_size, center, radius, mean):

    pattern = np.zeros(image_size, dtype=np.uint8)

    #gradient pattern with Gaussian distribution
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            #distance from the current pixel to the center
            distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            ##print(np.exp(-((distance / radius) ** 2)))
            #intensity 
            intensity = mean +  255* np.exp(-((distance/radius) ** 2))
            pattern[i, j] = np.clip(intensity, 0, 255)
    for i in range(image_size[0]-1,radius2-1,-1):
        for j in range(image_size[1]-1,radius2-1,-1):
            
            distance = np.sqrt((i - center2[0]) ** 2 + (j - center2[1]) ** 2)
            # print(distance)
            # print(np.exp(-((distance / radius) ** 2))
            intensity = mean + 255 * np.exp(-((distance/radius2) ** 2))
            pattern[i, j] = np.clip(intensity, 0, 255)
    return pattern

def HomomorphicFilter(gL,gH,c,d0,shape):
    
    ht=shape[0]
    wt=shape[1]
    centerx = ht//2
    centery=wt//2 
    H= np.zeros(shape,np.float32)
    for i in range(ht):
        for j in range(wt):
            
            u=i-centerx
            v=j-centery
            term = c*((u**2)+(v**2))/(d0**2)
            term = 1- np.exp(-term)  
            H[i,j]=(gH-gL)*term + gL
    return H
            
            
img =cv2.imread("unnamed.png",0)
cv2.imshow("input",img)

# image size, center, radius, mean of the Gaussian distribution

image_size = (img.shape[0],img.shape[1])  
center = (0,0) 
center2=(image_size[0]-1,image_size[1]-1) # Center coordinates
radius = min(image_size) // 4  # Radius of the circular gradient
radius2 = min(image_size) // 3
mean = 0


# Generate the illuminative circular gradient pattern
pattern = generate_illuminative_circular_gradient(image_size, center, radius, mean)
#cv2.normalize(pattern,pattern,0,1,cv2.NORM_MINMAX)
corrupt = cv2.add(img,pattern)
cv2.normalize(corrupt,corrupt,0,255,cv2.NORM_MINMAX)
cv2.imshow("Illuminative Circular Gradient Pattern",pattern)
cv2.imshow("corrupted image", corrupt)
#log transform 

log_corrupt = np.log1p(corrupt)

spectrum = np.fft.fft2(log_corrupt)


# Shift the zero frequency component to the center
shifted_spectrum = np.fft.fftshift(spectrum)

magnitude_spectrum = np.abs(shifted_spectrum)
phase = np.angle(shifted_spectrum)

#cutoff frequency and gamma values
cutoff_freq = 0.1
gamma_l = 0.5
gamma_h = 1.3
d0 = 50

H=HomomorphicFilter(gamma_l, gamma_h, cutoff_freq, d0, corrupt.shape)

magnitude_spectrum = magnitude_spectrum*H

filtered_img = np.multiply(magnitude_spectrum,np.exp(1j*phase))
                           
out = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_img)))                          

out=np.expm1(out)
out = (out - np.min(out)) * (255 / (np.max(out) - np.min(out)))
out =out.astype(np.uint8)

cv2.imshow("restored",out)

cv2.waitKey(0)
cv2.destroyAllWindows()

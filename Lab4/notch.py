import matplotlib



import matplotlib.pyplot as plt

import numpy as np



import cv2


def min_max_normalize(img_inp):
    inp_min = np.min(img_inp)
    inp_max = np.max(img_inp)

    for i in range (img_inp.shape[0]):
        for j in range(img_inp.shape[1]):
            img_inp[i][j] = (((img_inp[i][j]-inp_min)/(inp_max-inp_min))*255)
    return np.array(img_inp, dtype='uint8')

def onclick(event):
    global x, y
    ax = event.inaxes

    if ax is not None:
        x, y = ax.transData.inverted().transform([event.x, event.y])
        x = int(round(x))
        y = int(round(y))
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, x, y))
        point_list.append((x,y))
   
    


img = cv2.imread("bird.jpg",0)
size = img.shape[0]*img.shape[1]

#fourier transform

ft_img = np.fft.fft2(img)

ft_shift = np.fft.fftshift(ft_img)

phase = np.angle(ft_shift)

magnitude = np.abs(ft_shift)

magnitude_spectrum = np.log(magnitude+1)
  


magnitude_spectrum_scaled = min_max_normalize(magnitude_spectrum)

point_list=[]

# click and seed point set up
x = None
y = None

# The mouse coordinate system and the Matplotlib coordinate system are different, handle that


X = np.zeros_like(img)
plt.title("Please select seed pixel from the input")
im = plt.imshow(magnitude_spectrum_scaled, cmap='gray')
im.figure.canvas.mpl_connect('button_press_event', onclick)


plt.show(block=True)

d0=50
order=2
print(point_list)
notch = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
m=img.shape[0]//2
n=img.shape[1]//2


for u in range(img.shape[0]):
    for v in range(img.shape[1]):
        
        for k in range (len(point_list)):
            uk = point_list[k][0]
            vk = point_list[k][1]
            uk, vk = vk, uk
            d1 = np.sqrt((u - uk) ** 2 + (v - vk) ** 2)
            d2 = np.sqrt((u + uk) ** 2 + (v + vk) ** 2)
            if d1!=0 or d2!=0:
               notch[u][v]*=(1.0 / (1.0 + pow((d0 * d0) / (d1 * d2), order))) 



# Inverse Fourier transform

#filtered_img = np.multiply(magnitude_spectrum,np.exp(1j*phase))
filtered_shift = ft_shift * notch

# Inverse Fourier transform
filtered_img = np.fft.ifftshift(filtered_shift)
filtered_img = np.fft.ifft2(filtered_img)

# Calculate magnitude spectrum of filtered image
magnitude_filtered = np.abs(filtered_img)

# Normalize and display the output
out = min_max_normalize(magnitude_filtered)

cv2.imshow("Filtered Image", out)

cv2.imshow("input",img)

cv2.imshow("filter",notch)
         
        

cv2.imshow("spectrum",magnitude_spectrum_scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()




import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from copy import deepcopy as dpc
matplotlib.use('TkAgg')


img=cv2.imread("period_input.jpg",cv2.IMREAD_GRAYSCALE)

img = dpc(img)

x=None 
y=None 
point_list=[]
def onclick(event):
    global x,y
    ax=event.inaxes
    if ax is not None:
        x,y=ax.transData.inverted().transform([event.x,event.y])
        x=int(round(x))
        y=int(round(y))
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
       (event.button, event.x, event.y, x, y))
        point_list.append((x,y))
        
def min_max_normalize(img_inp):
    inp_min = np.min(img_inp)
    inp_max = np.max(img_inp)

    for i in range (img_inp.shape[0]):
        for j in range(img_inp.shape[1]):
            img_inp[i][j] = (((img_inp[i][j]-inp_min)/(inp_max-inp_min))*255)
    return np.array(img_inp, dtype='uint8')

ft=np.fft.fft2(img)

ft_shift=np.fft.fftshift(ft)

mag=np.abs(ft_shift)

mag_show=np.log1p(np.abs(ft_shift)+1)

mag_scaled = min_max_normalize(mag_show)

plt.title("Please select seed pixel from the input")
im = plt.imshow(mag_scaled, cmap='gray')
im.figure.canvas.mpl_connect('button_press_event', onclick)
plt.show(block=True)

notch_filter=np.ones((img.shape),dtype=np.float32)

m=notch_filter.shape[0]//2

n=notch_filter.shape[1]//2

print(point_list)
d0=15
for i in range(notch_filter.shape[0]):
    for j in range(notch_filter.shape[1]):
      for k in point_list:
            y=k[0]
            x=k[1]
            
            if(x<=m):
                x2=m+abs(m-x)
            else:
                x2=m-abs(m-x)
            if(y<=n):
                y2=n+abs(n-y)
            else:
                y2=n-abs(n-y)
                
            d1=np.sqrt((i-x)*(i-x)+(j-y)*(j-y))
            d2=np.sqrt((i-x2)*(i-x2)+(j-y2)*(j-y2))
      
            if(d1>d0):
                notch_filter[i,j]*=1
            else:
                notch_filter[i,j]*=0
            if(d2>d0):
                notch_filter[i,j]*=1
            else:
                notch_filter[i,j]*=0
                
cv2.imshow("notch",notch_filter)

cv2.waitKey(0)
                
ang=np.angle(ft_shift)
output=mag*notch_filter
## phase add
final_result = np.multiply(output, np.exp(1j*ang))

# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
img_back_scaled = min_max_normalize(img_back)

## plot
cv2.imshow("input", img)


cv2.imshow("Inverse transform",img_back_scaled)





cv2.waitKey(0)
cv2.destroyAllWindows()
            





        
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy as dpc
matplotlib.use('TkAgg')
img = cv2.imread("period_input.jpg",0)
img = dpc(img)

def min_max(inp):
    i_min = np.min(inp)
    i_max= np.max(inp)
    
    for i in range(inp.shape[0]):
        for j in range(inp.shape[1]):
            inp[i,j]=((inp[i,j]-inp.min())/(i_max-i_min))*255
    return np.round(inp).astype(np.uint8)


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
        print(x,y)
        point_list.append((x,y))


ft = np.fft.fft2(img)
ft_shift = np.fft.fftshift(ft)

mag = np.abs(ft_shift)

mag_show =np.log1p(np.abs(ft_shift)+1)
mag_scaled = min_max(mag_show)

plt.title("please select seed input")

im = plt.imshow(mag_scaled,cmap='gray')
im.figure.canvas.mpl_connect('button_press_event',onclick)
plt.show(block=True)

#notch
m=img.shape[0]
n=img.shape[1]
notch = np.ones((m,n),np.float32)
cx=m//2
cy=n//2
d0=15
for u in range (m):
    for v in range (n):
        for k in (point_list):
            y=k[0]
            x=k[1]
            dx = abs(cx-x)
            dy = abs(cy-y)
            if(x<=cx):
                x1=cx+dx
            else:
                x1=cx-dx
            if y<=cy:
                y1=cy+dy
            else:
                y1=cy-dy
            
            dk1 = np.sqrt((u-x)**2+(v-y)**2)
            dk2= np.sqrt((u-x1)**2+(v-y1)**2)
            
            #bandpass
            
            
            if(dk1>d0):
                notch[u,v]*=1
            else:
                notch[u,v]*=0
            if(dk2>d0):
                notch[u,v]*=1
            else:
                notch[u,v]*=0

cv2.imshow("filter",notch)
cv2.waitKey(0)

out = mag*notch

ang = np.angle(ft_shift)

result = np.multiply(out,np.exp(1j*ang ))

img_back = np.real(np.fft.ifft2(np.fft.ifftshift(result)))
img_back_scaled = min_max(img_back)

cv2.imshow("input",img)
cv2.imshow("output",img_back_scaled)
            
cv2.waitKey(0)
cv2.destroyAllWindows()              
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
            
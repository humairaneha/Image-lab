import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import cv2


img = cv2.imread("bird.jpg",0)

def min_max_norm(inp):
    i_max= inp.max()
    i_min = inp.min()
    dif= i_max-i_min
    out=np.zeros((inp.shape),np.uint8)
    for i in range(inp.shape[0]):
        for j in range(inp.shape[1]):
            
            out[i,j]=((inp[i,j]-i_min)/dif)*255
    return np.round(out).astype(np.uint8)

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


ft =np.fft.fft2(img)

ft_shift = np.fft.fftshift(ft)

mag = np.abs(ft_shift)
ang = np.angle(ft_shift)

mag_show = np.log1p(np.abs(ft_shift)+1)
mag_scaled = min_max_norm(mag_show)

plt.title("seed input")
im = plt.imshow(mag_scaled,cmap='gray')
im.figure.canvas.mpl_connect('button_press_event',onclick)
plt.show(block=True)

Btr_notch = np.ones(mag.shape,np.float32)

m=img.shape[0]
n=img.shape[1]

cx=m//2
cy=n//2
d0=10
conj =[]
for k in (point_list):
   
    y1=k[0]
    x1=k[1]
    dx=abs(x1-cx)
    dy=abs(y1-cy)
    if(x1<=cx):
       x2=cx+dx
    else:
        x2=cx-dx
    if(y1<=cy):
        y2=cy+dy
    else:
         y2=cy-dy
    conj.append((y2,x2))
point_list+=conj
for u in range (m):
    for v in range (n):
        t=1 
        for k in (point_list):
           
            y1=k[0]
            x1=k[1]
            dk1 = np.sqrt((u-x1)**2+(v-y1)**2)
            dk1p =np.sqrt((u+x1)**2+(v+y1)**2)
            
            t1 = (1.0/(1.0+(d0/dk1)**2))*( 1.0/(1.0+(d0/dk1p)**2))
            Btr_notch[u,v]*=t1
    
    
#Btr_notch = min_max_norm(Btr_notch)

cv2.imshow("filter",Btr_notch)
ang=np.angle(ft_shift)
output=mag*Btr_notch
## phase add
final_result = np.multiply(output, np.exp(1j*ang))

# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
img_back_scaled = min_max_norm(img_back)

## plot
cv2.imshow("input", img)


cv2.imshow("Inverse transform",img_back_scaled)       
cv2.waitKey(0)
cv2.destroyAllWindows()    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
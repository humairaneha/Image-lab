import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2


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
        point_list.append((y,x))






img = cv2.imread("hole.jpg",0)

r,img = cv2.threshold(img,130,255,cv2.THRESH_BINARY)

SE = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
SE*=255

plt.title("seed input")
im=plt.imshow(img,cmap='gray')
im.figure.canvas.mpl_connect('button_press_event',onclick)
plt.show(block = True)


xx = point_list[0][0]
yy= point_list[0][1]

comp = cv2.bitwise_not(img)

X = np.zeros((img.shape,np.uint8))
X[xx,yy] = 255
tmp=X
while True:
    dilated = cv2.dilate(tmp,SE,iterations=1)
    dilated = cv2.bitwise_and(dilated,comp)
    if(tmp==dilated).all())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
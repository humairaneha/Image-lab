import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt



point_list=[]

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

img = cv2.imread('hole.jpg', 0)
r, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)#_INV)

X = np.zeros_like(img)
plt.title("Please select seed pixel from the input")
im = plt.imshow(img, cmap='gray')
im.figure.canvas.mpl_connect('button_press_event', onclick)
plt.show(block=True)

kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)) #cv2.MORPH_RECT for all 1s

kernel1 = (kernel1) *255
kernel = np.uint8(kernel1)


comp=cv2.bitwise_not(img)




img2=np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)


xx=point_list[0][0]
yy=point_list[0][1]
img2[xx,yy]=255

tmp=img2
while(1):
    
    dilated = cv2.dilate(tmp,kernel,iterations = 1)
    dilated=cv2.bitwise_and(comp,dilated)
    x=(dilated == tmp).all()
    if(x==True):            
        break
    tmp=dilated
    
out=cv2.bitwise_or(dilated,img)


cv2.imshow("click",img2)
cv2.imshow("outtt",out)

cv2.waitKey(0)



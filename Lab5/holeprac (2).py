import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
x=None
y=None
point_list=[]

def onclick(event):
    global x,y
    ax=event.inaxes
    if ax is not None:
        x,y = ax.transData.inverted().transform([event.x,event.y])
        x=int(x)
        y=int(y)
        print(x,y)
        point_list.append((x,y))


SE = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
SE*=255
img = cv2.imread("hole.jpg")

r,img = cv2.threshold(img,130,255,cv2.THRESH_BINARY)


plt.title("seed input")
im = plt.imshow(img,cmap='gray')
im.figure.canvas.mpl_connect('button_press_event',onclick)
plt.show(block=True)
comp = 255-img

xx=point_list[0][0]
yy=point_list[0][1]
new = np.zeros_like(img)
new[x,y]=255


while True:
    
    dilated = cv2.dilate(new,SE,iterations=1)
    dilated = cv2.bitwise_and(dilated,comp)
    if((new==dilated).all()):
        break
    new = dilated
out=cv2.bitwise_or(new,img)
cv2.imshow("new",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
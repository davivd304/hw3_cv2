import numpy as np
import cv2
import matplotlib.pyplot as plt
o=cv2.imread('original.png',cv2.IMREAD_GRAYSCALE)
sobelx=cv2.Sobel(o,cv2.CV_64F,1,0)
sobelx=cv2.convertScaleAbs(sobelx)#转回nuit8
sobely=cv2.Sobel(o,cv2.CV_64F,0,1)
sobely=cv2.convertScaleAbs(sobely)


sobelxy=cv2.addWeighted(sobelx,0.5,sobely,0.5,0)

#开图
fig,axes=plt.subplots(1,2,figsize=(10,4))
axes[0].imshow(o,cmap='gray')
axes[0].set_title('orginal')
axes[0].axis('off')

axes[1].imshow(sobelxy,cmap='gray')
axes[1].set_title('sobelxy')
axes[1].axis('off')

plt.tight_layout
plt.savefig('sobel_compare.png',dpi=150)
plt.show()


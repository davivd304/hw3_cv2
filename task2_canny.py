import numpy as np
import cv2
import matplotlib.pyplot as plt
o=cv2.imread('original.png',cv2.IMREAD_GRAYSCALE)
r=cv2.Canny(o,100,200)

#开图
fig,axes=plt.subplots(1,2,figsize=(10,4))
axes[0].imshow(o,cmap='gray')
axes[0].set_title('orginal')
axes[0].axis('off')

axes[1].imshow(r,cmap='gray')
axes[1].set_title('r')
axes[1].axis('off')

plt.tight_layout
plt.savefig('canny_compare.png',dpi=150)
plt.show()


#高斯滤波
img = cv2.imread('original.png', cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(img, (5, 5), sigmaX=1.4)

#梯度幅值与方向
gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1)
magnitude = np.hypot(gx, gy)
angle = np.degrees(np.arctan2(gy, gx)) % 180

#开梯度图
plt.figure(figsize=(6, 6))
plt.imshow(magnitude, cmap='gray')
plt.title('gradient magnitude')
plt.axis('off')
plt.show()

#NMS
H, W = magnitude.shape
nms = np.zeros_like(magnitude)  
for y in range(1, H - 1):      
    for x in range(1, W - 1):
        a = angle[y, x]
        m = magnitude[y, x]

        # 根据梯度方向，选出这条方向上的前后两个邻居
        if a < 22.5 or a >= 157.5:       # 接近 0°：左、右
            n1 = magnitude[y, x - 1]
            n2 = magnitude[y, x + 1]
        elif a < 67.5:                   # 接近 45°：右上、左下
            n1 = magnitude[y - 1, x + 1]
            n2 = magnitude[y + 1, x - 1]
        elif a < 112.5:                  # 接近 90°：上、下
            n1 = magnitude[y - 1, x]
            n2 = magnitude[y + 1, x]
        else:                            # 接近 135°：左上、右下
            n1 = magnitude[y - 1, x - 1]
            n2 = magnitude[y + 1, x + 1]

       
        if m >= n1 and m >= n2:
            nms[y, x] = m


#开nms后的图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(magnitude, cmap='gray')
axes[0].set_title('gradient magnitude')
axes[0].axis('off')

axes[1].imshow(nms, cmap='gray')
axes[1].set_title('after NMS')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('canny_nms_compare.png', dpi=150)
plt.show()



#阈值筛选与弱边缘连接

max_value = nms.max()
high_threshold = 0.15 * max_value
low_threshold = 0.05 * max_value

# 强边缘：大于等于高阈值
strong = nms >= high_threshold

# 弱边缘：大于等于低阈值，同时不是强边缘
weak = (nms >= low_threshold) & (~strong)

H, W = nms.shape

edges = np.zeros((H, W), dtype=np.uint8)

# 强边缘直接保留为白色
edges[strong] = 255

# 把所有强边缘像素的坐标放进栈，作为连接的起点
stack = list(map(tuple, np.argwhere(strong)))

while stack:
    y, x = stack.pop()

    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:

            if dy == 0 and dx == 0:
                continue

            ny = y + dy
            nx = x + dx

            # 防止越过图像边界
            if 0 <= ny < H and 0 <= nx < W:

                if weak[ny, nx]:
                    edges[ny, nx] = 255

                    # 标记为已处理，避免重复入栈
                    weak[ny, nx] = False

                    # 新的弱边缘也变成起点，继续向外连接
                    stack.append((ny, nx))


#最后单独显示效果图
plt.figure(figsize=(6, 6))
plt.imshow(edges, cmap='gray')
plt.title('final edges')
plt.axis('off')

plt.tight_layout()
plt.savefig('canny_edges.png', dpi=150)
plt.show()
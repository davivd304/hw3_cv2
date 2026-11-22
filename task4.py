import numpy as np
import cv2
import matplotlib.pyplot as plt


def manual_histogram_equalization(img):


    # hist[k] 表示灰度值 k 在图像中出现了多少次
    hist = np.zeros(256, dtype=np.int64)

    for value in img.ravel():
        hist[value] += 1

    # 计算累积分布函数 CDF

    cdf = np.zeros(256, dtype=np.int64)
    cdf[0] = hist[0]

    for k in range(1, 256):
        cdf[k] = cdf[k - 1] + hist[k]

    # 找到第一个非零灰度级对应的 CDF
    # 用于把映射后的最小灰度调整到 0
    nonzero = np.flatnonzero(hist)
    cdf_min = cdf[nonzero[0]]

    total_pixels = img.size
    denominator = total_pixels - cdf_min

    # 特殊情况
    if denominator == 0:
        return img.copy(), hist

    #生成灰度映射表
    lut = np.zeros(256, dtype=np.uint8)

    for k in range(256):
        mapped = (cdf[k] - cdf_min) * 255 / denominator

        # 裁剪到合法范围，防止低灰度位置出现负数
        mapped = np.clip(mapped, 0, 255)
        lut[k] = round(mapped)

    # 第4步：逐像素查表
    # 原图中灰度为 k 的像素，全部替换成 lut[k]
    equalized = lut[img]

    return equalized, hist



img = cv2.imread('original.png', cv2.IMREAD_GRAYSCALE)
assert img is not None, '没读到 original.png'

# 调用手动实现的直方图均衡化
equalized, hist = manual_histogram_equalization(img)

# 统计均衡化后的直方图，用于对比
hist_equalized = np.zeros(256, dtype=np.int64)
for value in equalized.ravel():
    hist_equalized[value] += 1

# 使用 2×2 画布展示结果
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 原图
axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('original')
axes[0, 0].axis('off')

# 均衡化后的图像
axes[0, 1].imshow(equalized, cmap='gray', vmin=0, vmax=255)
axes[0, 1].set_title('histogram equalization')
axes[0, 1].axis('off')

# 原图直方图
axes[1, 0].bar(np.arange(256), hist, width=1.0, color='gray')
axes[1, 0].set_title('original histogram')
axes[1, 0].set_xlim(0, 255)
axes[1, 0].set_xlabel('gray level')
axes[1, 0].set_ylabel('pixel count')

# 均衡化后的直方图
axes[1, 1].bar(np.arange(256), hist_equalized, width=1.0, color='gray')
axes[1, 1].set_title('equalized histogram')
axes[1, 1].set_xlim(0, 255)
axes[1, 1].set_xlabel('gray level')
axes[1, 1].set_ylabel('pixel count')

plt.tight_layout()
plt.savefig('equalization_result.png', dpi=150)
plt.show()
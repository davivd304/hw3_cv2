import numpy as np
import cv2
import matplotlib.pyplot as plt


def harris_detect(gray, window_size=3, alpha=0.04, threshold_ratio=0.01):
    
    # 第1步：用小方差高斯滤波器对图像滤波，抑制噪声
    blur = cv2.GaussianBlur(gray, (3, 3), sigmaX=1.0)

    # 第2步：用 Sobel 一阶差分算子计算 Ix、Iy
    Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

    # 第3步：计算  三幅图像
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    IxIy = Ix * Iy

    # 第4步：分别对三幅图进行高斯滤波
    A = cv2.GaussianBlur(Ix2, (window_size, window_size), sigmaX=1.0)
    B = cv2.GaussianBlur(IxIy, (window_size, window_size), sigmaX=1.0)
    C = cv2.GaussianBlur(Iy2, (window_size, window_size), sigmaX=1.0)


    # 第5,6步：计算 Harris 响应 R 
    det_M = A * C - B * B
    trace_M = A + C
    R = det_M - alpha * trace_M * trace_M

    # 第7步：用正阈值 T 筛选候选角点
    R_max = R.max()
    if R_max <= 0:
        # 图中没有正响应，直接返回空结果
        return R, np.zeros(gray.shape, dtype=bool)

    T = threshold_ratio * R_max
    candidates = R > T

    # 第8步：在 5×5 邻域内进行非极大值抑制
    corners = np.zeros(gray.shape, dtype=bool)
    H, W = gray.shape
    nms_radius = 2

    for y, x in np.argwhere(candidates):
        # 确定局部邻域，同时避免越过图像边界
        y1 = max(0, y - nms_radius)
        y2 = min(H, y + nms_radius + 1)
        x1 = max(0, x - nms_radius)
        x2 = min(W, x + nms_radius + 1)

        local_max = R[y1:y2, x1:x2].max()

        if R[y, x] >= local_max:
            corners[y, x] = True

    return R, corners


def draw_corners(gray, corners):
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for y, x in np.argwhere(corners):
        # OpenCV 的坐标参数为 (x, y)
        cv2.circle(result, (x, y), 4, (0, 0, 255), 1)

    # 转成 RGB，供 Matplotlib 正确显示
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


#主程序
gray = cv2.imread('harris.jpg', cv2.IMREAD_GRAYSCALE)

#不同窗口对应的最终结果
R3, corners3 = harris_detect(gray, window_size=3)
R5, corners5 = harris_detect(gray, window_size=5)
R7, corners7 = harris_detect(gray, window_size=7)

result3 = draw_corners(gray, corners3)
result5 = draw_corners(gray, corners5)
result7 = draw_corners(gray, corners7)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(result3)
axes[0].set_title('window size = 3by3')
axes[0].axis('off')

axes[1].imshow(result5)
axes[1].set_title('window size = 5by5')
axes[1].axis('off')

axes[2].imshow(result7)
axes[2].set_title('window size = 7by7')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('harris_window_comparison.png', dpi=150)
plt.show()



# ==================== OpenCV Harris 对照 ====================

gray_float = np.float32(gray)


R_cv = cv2.cornerHarris( gray_float,3,  3, 0.04 )

# 使用与手写版相同比例的阈值筛选候选角点
cv_corners = R_cv > 0.01 * R_cv.max()

# 把检测到的角点画成红圈
cv_result = draw_corners(gray, cv_corners)


# 单独显示 OpenCV 的检测结果
plt.figure(figsize=(8, 8))
plt.imshow(cv_result)
plt.title('Harris corners - OpenCV')
plt.axis('off')
plt.tight_layout()
plt.savefig('harris_opencv.png', dpi=150)
plt.show()
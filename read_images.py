import numpy as np
import cv2
from matplotlib import pyplot as plt
# img:原始图像
# gray:灰度图像
# blurred：高斯模糊之后
# gradient:梯度运算
# edge:canny边缘检测
# draw_img:轮廓图
# crop_img:切割之后的图像

# 读入图片，灰度化
img = cv2.imread('C:\\Users\\Y\\Desktop\\szq0.png')
# 预处理函数


def preprocess(img):
    # filter 2D
    # kernel = np.ones((5, 5), np.float32) / 25
    # dst = cv2.filter2D(img, -1, kernel)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Harris角点检测

    # 中值滤波
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # 梯度
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1)

    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    blurred = cv2.GaussianBlur(gradient, (9, 9), 0)

    # 阈值分割
    # 背景为黑色，物体为白色，这样才能使用findContours
    thresh1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
    thresh1 = cv2.bitwise_not(thresh1)
    # 形态学
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
    # 膨胀一次，让轮廓突出
    dilation = cv2.dilate(thresh1, kernel2, iterations=1)
    # 腐蚀一次，去掉细节
    erosion = cv2.erode(dilation, kernel1, iterations=3)
    # 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, kernel2, iterations=3)
    # kernel_e = np.ones((3, 3), np.uint8)
    dilation2 = cv2.bitwise_not(dilation2)
    return dilation2


dilation2 = preprocess(img)
# blurred = cv2.GaussianBlur(dilation2, (9, 9),0)
# 中值滤波
# img_median = cv2.medianBlur(blurred, 7)
# canny边缘检测
# edges = cv2.Canny(dilation2, 100, 200)
# findContours
# 函数只接受二值图像，cv2.RETR_TREE表示建立一个等级树结构的轮廓
# cv2.CHAIN_APPROX_SIMPLE为轮廓的近似方法,一个矩形轮廓只需4个点来保存轮廓信息
# cv2.CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1
#
# 查找轮廓
image, contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("contours size: ", len(contours))
# -1表示绘制所以轮廓，(0, 0, 255)表示红色，3表示线宽
#cont = cv2.drawContours(img, contours, -1, (0,255,0), 3)
# drawContour
# 将所有轮廓按照面积大小排序
c = sorted(contours, key=cv2.contourArea, reverse=True)[0] #数字代表第几个区域
# 计算指定点集的最小区域的边界矩形
rect = cv2.minAreaRect(c)
# cv2.boxPoints查找旋转矩形的 4 个顶点（用于绘制旋转矩形的辅助函数）
box = np.int0(cv2.boxPoints(rect))

draw_img = cv2.drawContours(img.copy(), [box], -1, (0, 0, 255), 3)
#cv2.imshow("draw_img", draw_img)

# for c in contours:
#     rect = cv2.minAreaRect(c)
#     box = cv2.boxPoints(rect)
#     box = np.int0(box)
#     cv2.drawContours(img, [box], -1, (0, 0, 255), 5)

# BGR->HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

c = sorted(contours, key=cv2.contourArea, reverse=True)[0]


rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))
# 切割
Xs = [i[0] for i in box]
Ys = [i[1] for i in box]
x1 = min(Xs)
x2 = max(Xs)
y1 = min(Ys)
y2 = max(Ys)
height = y2 - y1
width = x2 - x1
crop_img= img[y1:y1 + height, x1:x1 + width]
#cv2.imshow('crop_img', crop_img)
# draw a bounding box arounded the detected barcode and display the image
#draw_img = cv2.drawContours(img.copy(), [box], -1, (0, 0, 255), 3)
#cv2.imshow("draw_img", draw_img)
# 分水岭分割

# plt.subplot(211), plt.imshow(img)
# plt.subplot(212), plt.imshow(dst)
# plt.subplot(223), plt.imshow(unknown)
# plt.subplot(224), plt.imshow(markers1)
# #plt.xlim([0,256])
#plt.show()

# 显示
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', crop_img)
cv2.waitKey(delay=0)
cv2.destroyAllWindows()


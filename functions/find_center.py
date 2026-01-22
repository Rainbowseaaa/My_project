import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import os
import matplotlib
import torch.nn.functional as F
from PIL import Image
import cv2
from my_dataset.my_dataset import MyDataset
import glob
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm
def find_center_method1(fp):
    # 提取图像轮廓---判断是否近似圆形---提取圆形轮廓中心点
    # 可以调整二值化阈值调整效果

    # 读取图像
    image = cv2.imread(fp)
    if image is None:
        print(f"Error: Unable to load image at {fp}")
        return
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化处理
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 边缘检测
    edges = cv2.Canny(binary, 50, 150)

    # 轮廓检测
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    # 筛选圆形轮廓
    for contour in contours:
        # 计算轮廓的面积和周长
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # 计算轮廓的圆度
        if perimeter > 0:
            circularity = 4 * np.pi * (area / (perimeter * perimeter))

            # 筛选出近似圆形的轮廓
            if 0.5 < circularity < 1.5:
                # 计算最小外接圆
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                centers.append(center)
                # 在图像上绘制圆心和圆
                cv2.circle(image, center, radius, (0, 255, 0), 2)
                cv2.circle(image, center, 2, (0, 0, 255), -1)

    # 缩小图像
    scale_percent = 50  # 缩小比例，例如50%表示缩小一半
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # 显示结果
    cv2.imshow('Detected Circle', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return centers


def non_max_suppression(energy_map, r, n):
    """
    非极大值抑制，确保每个局部区域内只有一个最强的圆心
    :param energy_map: 能量图
    :param r: 半径
    :param n: 需要找的圆心数量
    :return: 最强的 n 个圆心
    """
    local_maxima = np.zeros_like(energy_map, dtype=bool)
    h, w = energy_map.shape

    # 遍历每个像素，找到局部最大值
    for i in range(r, h - r):
        for j in range(r, w - r):
            window = energy_map[i - r:i + r + 1, j - r:j + r + 1]
            if energy_map[i, j] == np.max(window):
                local_maxima[i, j] = True

    # 找到局部最大值中能量最大的 n 个圆心
    flat_indices = np.argpartition(energy_map[local_maxima].flatten(), -n)[-n:]
    indices = np.unravel_index(flat_indices, energy_map[local_maxima].shape)

    # 提取圆心位置
    centers = []
    for idx in zip(*indices):
        i, j = np.where(local_maxima)[0][idx[0]], np.where(local_maxima)[1][idx[0]]
        centers.append((j, i))  # (x,y)

    return centers


def find_center_method2(fp, r=10, n=10, strike=1):
    """
    遍历所有点，计算半径r内的强度，取强度最大的点为中心。
    :param r: 计算强度的而半径
    :param n: 要找的点的数目
    :param strike: 计算滑动窗口的大小
    """
    image = cv2.imread(fp)
    # 检查图像是否成功读取
    if image is None:
        print(f"Error: Unable to load image at {fp}")
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        window_size = 2 * r + 1

        # 初始化能量数组
        energy_map = np.zeros_like(gray, dtype=np.float32)

        # 计算每个窗口的能量
        for i in range(r, gray.shape[0] - r, strike):
            for j in range(r, gray.shape[1] - r, strike):
                window = gray[i - r:i + r + 1, j - r:j + r + 1]
                energy = np.sum(window)
                energy_map[i, j] = energy

        # 找到能量最大的 n 个圆心
        centers = non_max_suppression(energy_map, r, n)  # (x, y),x正方向向右，y正方向向下

        # 在图像上绘制圆心和圆
        for i, center in enumerate(centers):
            x, y = center
            cv2.circle(image, (x, y), r, (255, 0, 0), 2)
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
            # 添加文本标签
            text = f"{i}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_color = (255, 255, 255)  # 白色
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_x = x - text_size[0] // 2  # 文本的 x 坐标
            text_y = y - r - 10  # 文本的 y 坐标，距离圆心上方一定的距离

            cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # 缩小图像以便显示
        scale_percent = 100  # 缩小比例，例如50%表示缩小一半
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        # 显示结果
        cv2.imshow('Detected Circle', resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return centers

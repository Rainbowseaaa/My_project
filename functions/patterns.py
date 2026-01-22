import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
import torchvision
import numpy as np
import os
import matplotlib
from model import optical_unit
from model import optical_unit
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from utils import gen_SLM_mask
from PIL import Image


def circle_pattern(r, size=(1080, 1920)):
    p = torch.zeros(size)
    x = torch.arange(size[1]).to(torch.float)
    x = x - torch.mean(x)
    y = torch.arange(size[0]).to(torch.float)
    y = y - torch.mean(y)
    x, y = torch.meshgrid(x, y, indexing="xy")
    r_ = torch.sqrt(x ** 2 + y ** 2)
    p[r_ <= r] = 1
    return p
def dual_circle_pattern(r, xshift1,xshift2, size=(1080, 1920)):
    p1 = torch.zeros(size)
    p2 = torch.zeros(size)
    x = torch.arange(size[1]).to(torch.float)
    x = x - torch.mean(x)
    y = torch.arange(size[0]).to(torch.float)
    y = y - torch.mean(y)
    x, y = torch.meshgrid(x, y, indexing="xy")
    r_ = torch.sqrt((x-xshift1) ** 2 + y ** 2)
    p1[r_ <= r] = 1
    r_ = torch.sqrt((x-xshift2) ** 2 + y ** 2)
    p2[r_ <= r] = 1
    return p1+p2

def focal_pattern(f, coor, focus_index, lamb, size):
    '''
    :param f:焦距
    :param coor:坐标
    :param focus_index:焦斑的横向位置，(x,y)
    :param size:pattern占的区域大小
    :return:
    '''
    pattern = torch.zeros_like(coor.x) + 0j
    # phase = exp(-jk/(2f)*(x^2+y^2))
    k = 2 * torch.pi / lamb
    x = coor.x[0, focus_index[0]]
    y = coor.y[focus_index[1], 0]
    temp = torch.exp(-1j * k / (2 * f) * ((coor.x - x) ** 2 + (coor.y - y) ** 2))
    top = max(focus_index[1] - size // 2, 0)
    bottom = min(focus_index[1] + size // 2, coor.y.shape[0])
    left = max(focus_index[0] - size // 2, 0)
    right = min(focus_index[0] + size // 2, coor.x.shape[1])

    pattern[top:bottom, left:right] = temp[top:bottom, left:right]
    return pattern


def tilt_pattern(angle, coor, lamb):
    angle = torch.tensor(angle)
    max_angle = torch.atan(torch.tensor(lamb / 24e-6)) * 180 / torch.pi
    if angle > max_angle:
        angle = max_angle
        raise ValueError("max_angle is 1.2 degree!")
    angle = angle * torch.pi / 180
    phase = torch.exp(-1j * coor.x * torch.sin(angle) / lamb * 2 * torch.pi)
    return phase


def pad(x, t):
    if t == 'slm':
        size = (1200, 1920)
    elif t == 'dmd':
        size = (1080, 1920)
    else:
        raise ValueError("t should be slm or dmd!")
    row, col = x.shape
    left = (size[1] - col) // 2
    top = (size[0] - row) // 2
    return torch.nn.functional.pad(x, (left, left, top, top))


def depad(x, size):
    row, col = x.shape
    left = (col - size[1]) // 2
    top = (row - size[0]) // 2
    return x[top:-top, left:-left]


def focal_patterns_on_target(f, coor, focal_ratios, lamb, region_size):
    '''
    :param f: 焦距
    :param coor: 坐标
    :param focal_ratios: 焦点横向位置，(x,y)
    :param lamb: 波长
    :param size: 靶面大小，(H,W)
    :region_size: 每个焦点pattern的大小
    :return:
    '''
    row, col = coor[1].x.shape
    left = int(abs((coor[1].xstart-coor[-1].xstart) / coor[1].px))
    bottom = int(abs((coor[1].ystart-coor[-1].ystart) / coor[1].py))
    pattern = torch.zeros_like(coor[1].x) + 0j
    for center in focal_ratios:
        x_index = int(center[0] * coor[-1].Nx * coor[-1].px/coor[1].px + left)
        y_index = int(center[1] * coor[-1].Ny * coor[-1].py/coor[1].py + bottom)
        pattern += focal_pattern(f=f, coor=coor[1], focus_index=[x_index, y_index], lamb=lamb, size=region_size)
    return pattern


def multiple_patterns(*args, mode='multiply', n_bits=8):
    # mode:multiply 表示exp(j*phi)项相乘，相当于phi相加。add表示exp(j*phi)相加。
    # n_bits表示读入的图像的bit数
    shapes = []
    patterns = []
    for arg in args:
        img = Image.open(arg)
        img = torch.from_numpy(np.array(img))
        shapes.append(img.shape)
        if shapes[0][0] != shapes[-1][0] or shapes[0][1] != shapes[-1][1]:
            raise ValueError("all images should have the same height and width!")
        if n_bits == 8:
            img = torch.exp(1j * img / (2 ** 8 - 1) * 2 * torch.pi)
        if n_bits == 10:
            img = torch.exp(1j * img / (2 ** 10 - 1) * 2 * torch.pi)
        patterns.append(img)
    if mode == 'add':
        pattern = torch.sum(torch.stack(patterns, dim=-1), dim=-1)
    if mode == 'multiply':
        pattern = torch.prod(torch.stack(patterns, dim=-1), dim=-1)
    return pattern


def DMD_process(image, z_angle=45):
    theta = torch.tensor(24 * torch.pi / 180)
    stretch_factor = 1 / torch.cos(theta)
    width, height = image.size
    # 计算拉伸后的尺寸
    new_width = int(width * stretch_factor)
    new_height = height  # 如果只在水平方向拉伸，高度不变
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    image = image.rotate(z_angle, expand=True)
    return image
def gen_coor(sps):
    c = []
    l = sps['layer_params']
    for i in range(len(l)):
        c.append(optical_unit.FreDomainCoordinate(l[i]['Ny_s'], l[i]['Ny_e'], l[i]['Nx_s'], l[i]['Nx_e'], l[i]['py'],
                                                  l[i]['px'], device))
    return c

if __name__ == "__main__":
    # 产生DMD显示的图片时，要从0-1归一化到0-255
    type = 'SLM'  # SLM or DMD
    device = "cpu"
    mps = { # model parameters
        "layers_num": 1,  # hidden layer数
        "residual_num": [],  # residual layer 编号列表
        'residual_ratio': 1,  # 残差比例
        "model_path": "test.pth",  # 模型存储路径
        "train_valid_split_ratio": 0.98,  # 训练集和验证集比例
        "dropout": 0,  # dropout比例
        "batch_size": 32,  # minibatch大小，建议2的n次方
        "epochs": 1,  # 训练轮次
        "lr": 0.05,  # 学习率
        "early_stop": 1000,  # 早停数
        "num_workers": 16,  # 推荐CPU核心数-1或-2
        "seed": 318,  # 随机数种子
        "class_region_centers": [[0.2, 0.2], [0.5, 0.2], [0.8, 0.2]],
        # "class_region_centers": [[0.3, 0.25], [0.5, 0.25], [0.7, 0.25],
        #                          [0.2, 0.5], [0.4, 0.5], [0.6, 0.5], [0.8, 0.5],
        #                          [0.3, 0.75], [0.5, 0.75], [0.7, 0.75]
        #                          ],         # 不同类区域的中心坐标(百分比,(x,y))
        "class_region_pixels": 3  # 分类区域单边大小
    }
    scale_factor = 1  # 整体放缩因子，一般为1
    inp_params = {
            'Ny_s': int(-500*scale_factor),
            'Ny_e': int(499*scale_factor),
            'Nx_s': int(-500*scale_factor),
            'Nx_e': int(499*scale_factor),
            'px': 7.56e-6/scale_factor,
            'py': 7.56e-6/scale_factor
        }  # 输入面大小, [y,x]
    slm_params = {
            'Ny_s': int(-500*scale_factor),
            'Ny_e': int(499*scale_factor),
            'Nx_s': int(-500*scale_factor),
            'Nx_e': int(499*scale_factor),
            'px': 8e-6/scale_factor,
            'py': 8e-6/scale_factor
        }
    cam_params = {
            'Ny_s': int(-500*scale_factor),
            'Ny_e': int(499*scale_factor),
            'Nx_s': int(-500*scale_factor),
            'Nx_e': int(499*scale_factor),
            'px': 5.86e-6/scale_factor,
            'py': 5.86e-6/scale_factor
        }
    sps = {  # system parameters
        "lambda": 532e-9,  # 波长，单位都是m
        "distance": [200e-3, 200e-3],  # 层间距
        "method": ['Bluestein', ],  # 传播方法 AFM, SAFM, Blustein
        "image_pixels": 100,
        "layer_params": [slm_params],
        "shiftx": [0],
        "shifty": [0],
        "discretion": False,  # 是否对slm相位离散化
        "discretion_bits": 10,  # 离散化位深
    }
    coor = gen_coor(sps)
    if type == 'SLM':
        # 产生多、焦点pattern
        p = focal_patterns_on_target(f=200e-3, coor=coor, focal_ratios=mps['class_region_centers'],
                                     lamb=sps["lambda"], region_size=2000)
        # p = focal_patterns_on_target(f=100e-3, coor=coor, focal_ratios=[[0.5, 0.5],],
        #                              lamb=config["lambda"], size=(1000, 1000), region_size=2000)
        # p = multiple_patterns('../SLM_pattern_8bits/CR-EL.bmp',
        #                       '../SLM_pattern_8bits/tilt_01degree.bmp')
        slm_pattern = gen_SLM_mask.gen_single_SLM_mask(Mask=p.angle())
        slm_pattern = slm_pattern.cpu().numpy()
        image = Image.fromarray(slm_pattern)
        image.save('../SLM_pattern_8bits/test.bmp')
    if type == 'DMD':
        p = circle_pattern(300)
        p = (p>0.5).numpy() # 二值化，保证输出1位的图像。
        image = Image.fromarray(p)
        angle_list = torch.linspace(30, 35, 51)
        for angle in angle_list:
            image_ = DMD_process(image, z_angle=angle)
            # 获取图像的宽度和高度
            width, height = image_.size
            # 计算裁剪的左上角和右下角坐标
            left = (width - 1920) // 2
            top = (height - 1080) // 2
            right = left + 1920
            bottom = top + 1080
            # 裁剪图像
            cropped_image = image_.crop((left, top, right, bottom))
            # 保存裁剪后的图像
            cropped_image.save(f'../DMD_imgs/circle_300_{int(angle * 10)}.bmp')

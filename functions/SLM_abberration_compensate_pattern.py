# Editor: WangNing
# Date:2024/7/9
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
from model import optical_unit


from utils import gen_SLM_mask

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from PIL import Image
import numpy
import PIL.Image

from UPO_SLM_80Rplus.SLM_UPOLabs import *


def lens_phase(config, coor, f):
    '''
    :param coor:
    :param f: 焦距
    :return:
    '''
    k = 2 * torch.pi / config["lambda"]
    pattern = torch.exp(-1j * k / (2 * f) * (coor.r ** 2))
    return pattern


def crop(pattern, size):
    h, w = pattern.shape
    left = (w - size[1]) // 2
    top = (h - size[0]) // 2
    return pattern[top:top + size[0], left:left + size[1]]


def const_phase(phase=0., slm_size=(1200, 1920)):
    pattern = torch.ones(slm_size) + 0j
    pattern = torch.exp(1j * pattern * phase)
    return pattern


if __name__ == "__main__":
    device = "cpu"
    config = {
        "lambda": 532e-9,  # 波长，单位都是m
        "neural_pixels": 2000,  # layer面大小
        "n_pad": 0,  # 单边填充像素
        "pixel_pitch": 8e-6,  # 像素大小
    }
    config["total_cal_pixels"] = int(config["neural_pixels"] + 2 * config["n_pad"])
    coor = optical_unit.Coordinate(config, device)

    slm_size = (1200, 1920)  # H*W
    # f = torch.linspace(-100000e-3, -30000e-3, 100)
    f = [-15000e-3]
    # for i, fi in enumerate(f):
    #     len = lens_phase(config, coor, fi)
    #     plt.figure()
    #     plt.imshow(len.angle())
    #     plt.show()
    #     len_slm = crop(len, slm_size)
    #     shift1 = const_phase(torch.pi*i/2, slm_size)
    #
    #     mask = shift1
    #
    #     slm_mask = gen_SLM_mask.gen_single_SLM_mask(SLM_size=(1200, 1920), bits=8, Mask=mask.angle())
    #     slm_mask_np = slm_mask.cpu().numpy()
    #     image = Image.fromarray(slm_mask_np)
    #
    #     bmp = f'D:/ONN/13-single layer for exp/slm_bg/slm_bg{i}.bmp'
    #     image.save(bmp)
    # image.show()
    #
    # slm = SLM_UP()
    # slm.Open_window(screenNum=1)
    # slm.Disp_ReadImage(path=bmp, screenNum=1, bits=8)

    # slm.Close_window(1)
    lens = lens_phase(config, coor, f=15000e-3)
    I = (lens + 1).abs()**2
    I = crop(I, slm_size)
    plt.figure()
    plt.imshow(I)
    plt.show()
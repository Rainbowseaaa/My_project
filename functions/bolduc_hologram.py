import numpy as np
from functions import OAM_gen as LG
import torch
import scipy.interpolate as interp
from functions.utils import *
from functions import utils
def target_field(l,p,Ny,Nx,w0,dx):
    wavelength = 532e-9  # 波长 (m)
    k = 2 * np.pi / wavelength
    grid_size_x = dx*Nx/2  # 空间范围 (m)
    grid_size_y = dx*Ny/2
    zR = np.pi * w0 ** 2 / wavelength
    # 坐标网格
    x = np.linspace(-grid_size_x, grid_size_x, Nx)
    y = np.linspace(-grid_size_y, grid_size_y, Ny)
    X, Y = np.meshgrid(x, y, indexing='xy')
    r = np.sqrt(X ** 2 + Y ** 2)
    phi = np.arctan2(Y, X)
    # LG
    L = LG.LG_beam(r, phi, wavelength, w0, zR, 0, l, p, True)
    return L
from scipy.interpolate import interp1d
def build_inverse_sinc_lut(m_min=-np.pi, m_max=-0.00001, N=10000):
    m_vals = np.linspace(m_min, m_max, N)
    a_vals = np.sin(m_vals)/m_vals  # sinc = sin(pi m) / pi m
    # 降序排列（保证 a_vals 是单调的）
    sort_idx = np.argsort(-a_vals)
    a_vals_sorted = a_vals[sort_idx]
    m_vals_sorted = m_vals[sort_idx]
    # 构建 a -> m 的插值函数
    return interp1d(a_vals_sorted, m_vals_sorted, kind='linear', bounds_error=False, fill_value=1.0)

# 构建查找表（一次性）
inverse_sinc_lut = build_inverse_sinc_lut()

# 应用于二维矩阵 A（必须在范围内）
def inverse_sinc_fast(A):
    A_clipped = np.clip(A, 0.00001, 0.9999)  # 避免极限区域不稳定
    return inverse_sinc_lut(A_clipped)

def bolduc_phase_encoding(amplitude, phase, period):
    amplitude = amplitude/np.max(amplitude*0.9) # 如果最大值是1，图案有问题。0.9保证图案质量。
    M = 1 + inverse_sinc_fast(amplitude) / np.pi
    F = phase - np.pi * M
    c, r = np.shape(amplitude)
    X, Y = np.meshgrid(np.arange(r), np.arange(c))
    X = X-np.mean(X)
    phi = F + 2 * np.pi * X / period
    phi = np.mod(phi, 2*np.pi)*M
    return phi


def bolduc_phase_encoding(amplitude, phase, period):
    # Override the earlier definition to keep amplitude normalization external.
    M = 1 + inverse_sinc_fast(amplitude) / np.pi
    F = phase - np.pi * M
    c, r = np.shape(amplitude)
    X, Y = np.meshgrid(np.arange(r), np.arange(c))
    X = X-np.mean(X)
    phi = F + 2 * np.pi * X / period
    phi = np.mod(phi, 2*np.pi)*M
    return phi


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    tar = target_field(l=[1], p=[0], Nx=1920, Ny=1200, w0=12e-4, dx=8e-6)
    tar_amp = np.abs(tar)
    tar_pha = np.angle(tar)
    # tar_amp = np.zeros([1200,1920])+1e-9
    # tar_amp[400:500,800:1200] = 1
    # tar_amp[500:700,900:1000] = 1
    # tar_pha = np.zeros([1200,1920])+1e-9
    period = 8
    CHG = bolduc_phase_encoding(tar_amp, tar_pha, period)
    utils.to_slm_8bit_png(CHG, "OAM_CGH.png", src_type="phase_rad",
                          correct_phase=r'D:\实验\20251219测试\UPO\phase_correct_8bit.bmp')
    fig,ax = plt.subplots(1,1)
    ax.imshow(CHG,'gray')
    plt.show()
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(tar_amp)
    ax[1].imshow(tar_pha)
    plt.show()
    deltaf = 1/period
    CHG_cmp = np.exp(1j*CHG)
    CHG_pad = np.pad(CHG_cmp, 2000)
    r,c = np.shape(CHG_pad)
    rr,cc = np.meshgrid(np.arange(r), np.arange(c),indexing='ij')
    mid = c//2
    filter_r = 300
    first_order_pixel = deltaf/(1/c)
    fre = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(CHG_pad)))
    plt.figure()
    plt.imshow(np.log(np.abs(fre)))
    plt.show()
    # fre[((rr-r//2)**2 + (cc-c//2-first_order_pixel)**2 > filter_r**2)] = 1e-9
    fre[np.abs(cc - (mid + first_order_pixel)) > filter_r] = 1e-9
    plt.figure()
    plt.imshow(np.log(np.abs(fre)))
    plt.show()
    recover_filed = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(fre)))
    recover_filed = recover_filed[2000:-2000,2000:-2000]

    r,c = np.shape(recover_filed)
    rr,cc = np.meshgrid(np.arange(c), np.arange(r))
    tilt_phase = np.exp(1j*2*np.pi*rr/period)
    recover_filed = recover_filed*tilt_phase

    fig,ax = plt.subplots(1,2)
    ax[0].imshow(np.angle(recover_filed))
    ax[1].imshow(np.abs(recover_filed))
    plt.show()

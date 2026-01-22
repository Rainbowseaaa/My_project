import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt

def generate_vonkarman_phase_screen(N, delta, r0, L0, l0, seed=None):
    """
    生成二维湍流相位屏（考虑外尺度 L0 和内尺度 l0）。

    参数
    ----
    N     : int
        相位屏的像素尺寸，生成 N x N 的数组。
    delta : float
        空间采样间隔（单位：米/像素）。
    r0    : float
        Fried 参数，相干长度（单位：米）。
    L0    : float
        湍流外尺度（energy injection scale，单位：米）。
    l0    : float
        湍流内尺度（dissipation scale，单位：米）。
    seed  : int or None
        随机数种子（可选），指定后可复现随机相位屏。

    返回
    ----
    phi   : ndarray, shape (N, N)
        在实空间中的湍流相位屏（单位：弧度）。
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. 构造中心化后的二维频率网格
    #    np.fft.fftfreq 返回的顺序是 [0, +..., -..., ...]，需要 fftshift 把零频移到中心
    fx = np.fft.fftfreq(N, d=delta)  # 单位：1/m
    fy = np.fft.fftfreq(N, d=delta)
    fx = np.fft.fftshift(fx)
    fy = np.fft.fftshift(fy)
    FX, FY = np.meshgrid(fx, fy)     # 形状：(N, N)
    rho = np.sqrt(FX**2 + FY**2)     # 频率幅值 |f|，单位：1/m

    # 2. 计算 Von Kármán 功率谱（含外尺度和内尺度修正）
    #    Phi(f) = C * (rho^2 + L0^{-2})^{-11/6} * exp[- (rho*l0)^2 ]
    C = 0.023 * (r0**(-5.0/3.0))
    # (rho^2 + L0^{-2})^(-11/6)
    PSD = C * (rho**2 + (1.0 / L0)**2)**(-11.0/6.0)
    # 乘以内尺度指数衰减 exp[-(rho*l0)^2]
    PSD *= np.exp(- (rho * l0)**2)

    # 注意处理 rho=0 的位置，使其有限：
    # 当 rho=0 时，(rho^2 + L0^{-2})^(-11/6) = (L0^{-2})^(-11/6) = L0^(11/3)
    PSD[rho == 0] = C * (L0**(11.0/3.0)) * np.exp(-0.0)

    # 3. 离散化：乘以频域采样间隔 Δf = 1/(N*Δ)
    delta_f = 1.0 / (N * delta)
    PSD *= delta_f

    # 4. 生成复高斯随机谱：实部和虚部独立 N(0,1)，乘以 sqrt(PSD/2)
    #    使得 E[|Cn(f)|^2] = PSD(f)
    cn_real = np.random.randn(N, N)
    cn_imag = np.random.randn(N, N)
    cn = (cn_real + 1j * cn_imag) * np.sqrt(PSD / 2.0)

    # 5. 频域→时域：做二维 IFFT，得到实部相位屏，并乘以 (N*Δ)^2 进行归一化补偿
    phi = np.fft.ifft2(np.fft.ifftshift(cn))  # 先 ifftshift 再 ifft2
    phi = np.real(phi) * (N * delta)**2

    return phi


if __name__ == "__main__":
    # ================================
    # 示例：生成并可视化带外/内尺度湍流相位屏
    # ================================
    # 参数设置
    N = 5120            # 屏幕像素：512×512
    delta = 0.001      # 空间采样间隔：5 mm/像素
    r0 = 0.1           # Fried 参数：0.1 m
    L0 = 1         # 湍流外尺度：50 m
    l0 = 0.001         # 湍流内尺度：2 mm
    seed = 2025        # 随机数种子（可选）

    # 生成相位屏
    phase_screen = generate_vonkarman_phase_screen(N=N, delta=delta,
                                                  r0=r0, L0=L0, l0=l0,
                                                  seed=seed)

    # 可视化：伪彩色图
    plt.figure(figsize=(6, 5))
    im = plt.imshow(phase_screen,
                    cmap='jet',
                    extent=[0, N*delta, 0, N*delta])
    plt.colorbar(im, label="Phase (radians)")
    plt.title(f"Von Kármán Phase Screen\n"
              f"N={N}, Δ={delta*1e3:.1f} mm, r₀={r0*100:.1f} cm, L₀={L0:.1f} m, l₀={l0*1e3:.1f} mm")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.tight_layout()
    plt.show()

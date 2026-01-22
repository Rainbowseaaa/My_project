import matplotlib

matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt


def generate_fractal_phase_screen(size, H=5 / 6, seed=None):
    """
    使用中点位移（Diamond-Square）算法生成二维分形湍流相位屏。

    参数
    ----
    size : int
        相位屏的边长，必须为 2^n + 1 的形式，例如 257, 513, 1025 等。
    H    : float
        Hurst 指数，对应湍流的能谱斜率。Kolmogorov 湍流的 H ≈ 5/6。
    seed : int or None
        随机数种子，可选。指定后可保证结果可复现。

    返回
    ----
    phi : ndarray, shape (size, size)
        生成的二维分形相位屏（单位：任意，相对相位）。通常零均值，方差可自行缩放。
    """
    if seed is not None:
        np.random.seed(seed)

    # 检查 size 是否为 2^n + 1
    def is_valid_size(s):
        return ((s - 1) & (s - 2)) == 0  # (s-1) must be power of two

    if not is_valid_size(size):
        raise ValueError("size 必须为 2^n + 1，例如 257, 513, 1025 等。")

    # 初始化二维网格，全部填零
    phi = np.zeros((size, size), dtype=float)

    # 初始四个角设为 0（也可随机，但这里取 0）
    # 如果想要更随机，可以把下面四行替换为随机值
    # phi[0, 0] = np.random.randn()
    # phi[0, size-1] = np.random.randn()
    # phi[size-1, 0] = np.random.randn()
    # phi[size-1, size-1] = np.random.randn()

    # 初始位移标准差 sigma_0：可取 1.0，后续按尺度递减
    sigma = 1.0

    # 中点位移算法：step 是当前子格边长
    step = size - 1
    while step > 1:
        half = step // 2

        # —— Diamond 步骤 ——
        # 对于每个以 (i, j) 为左上角的子格：
        #   center = 平均(四个角) + N(0, sigma)
        for i in range(0, size - 1, step):
            for j in range(0, size - 1, step):
                x0, y0 = i, j
                x1, y1 = i + step, j + step
                xc, yc = i + half, j + half
                avg_corners = (phi[x0, y0] + phi[x0, y1] +
                               phi[x1, y0] + phi[x1, y1]) / 4.0
                phi[xc, yc] = avg_corners + np.random.randn() * sigma

        # —— Square 步骤 ——
        # 对于每个 diamond：以中心点 (i, j) 为中心，取上下左右四个点平均 + N(0, sigma)
        for i in range(0, size, half):
            for j in range((i + half) % step, size, step):
                # 计算对角相邻四个点的坐标（如果在边界之外则忽略）
                s = []
                if i - half >= 0:
                    s.append(phi[i - half, j])
                if i + half < size:
                    s.append(phi[i + half, j])
                if j - half >= 0:
                    s.append(phi[i, j - half])
                if j + half < size:
                    s.append(phi[i, j + half])
                avg_neighbors = np.mean(s)
                phi[i, j] = avg_neighbors + np.random.randn() * sigma

        # 每次迭代后，尺度减半，sigma 也按 2^{-H} 缩减
        step = half
        sigma *= 2 ** (-H)

    # 最后：将 phi 均值置零，便于后续根据实际 r0 缩放
    phi -= np.mean(phi)
    return phi


if __name__ == "__main__":
    # ================================
    # 示例：生成并可视化分形湍流相位屏
    # ================================
    size = 513  # 必须为 2^n + 1，例如 513 = 2^9 + 1
    H = 5 / 6  # Kolmogorov 湍流的 Hurst 指数
    seed = 2025  # 随机数种子，结果可复现

    # 生成相位屏
    phase_screen = generate_fractal_phase_screen(size, H=H, seed=seed)

    # 可视化：伪彩色图
    extent = [0, size, 0, size]  # 单位可按像素索引
    plt.figure(figsize=(6, 5))
    im = plt.imshow(phase_screen, cmap='jet', extent=extent)
    plt.colorbar(im, label="Relative Phase (arb. units)")
    plt.title(f"Fractal Turbulence Phase Screen\n(size={size}, H={H:.3f})")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.tight_layout()
    plt.show()

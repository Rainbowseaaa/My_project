import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.special import genlaguerre
import scipy.signal


# -------- Laguerre-Gaussian 基础函数 --------
def w_z(z, w0, zR):
    return w0 * np.sqrt(1 + (z / zR) ** 2)


def R_z(z, zR):
    return np.inf if z == 0 else np.abs(z) * (1 + (zR / z) ** 2)


def gouy_z(z, zR):
    return np.arctan(z / zR)


def LG_beam(r, phi, wavelength, w0, zR, z, ls, ps, norm=True):
    """多阶 LG 叠加"""
    wz, Rz, gouy = w_z(z, w0, zR), R_z(z, zR), gouy_z(z, zR)
    LGs = np.zeros_like(r, dtype=complex)

    for l, p in zip(ls, ps):
        lag = genlaguerre(p, abs(l))(2 * r ** 2 / wz ** 2)
        amp = (np.sqrt(2) * r / wz) ** abs(l) * lag * np.exp(-r ** 2 / wz ** 2)
        phase = (np.exp(1j * l * phi) *
                 np.exp(-1j * 2 * np.pi / wavelength * r ** 2 / (2 * Rz)) *
                 np.exp(1j * (2 * p + abs(l) + 1) * gouy))
        LG = (amp * phase) / wz
        if norm:
            LG /= np.max(np.abs(LG))
        LGs += LG
    return LGs


# -------- 主环能量占比 --------
def analyze_main_ring_energy(intensity, r):
    """
    统计主环（主峰两侧最近极小值之间）的能量占比
    """
    r_max = r.max()
    r_bins = np.linspace(0, r_max, 1200)
    radial = np.zeros_like(r_bins)

    r_flat, I_flat = r.ravel(), intensity.ravel()
    for i in range(len(r_bins) - 1):
        mask = (r_flat >= r_bins[i]) & (r_flat < r_bins[i + 1])
        if np.any(mask):
            radial[i] = I_flat[mask].mean()

    # 平滑抑制毛刺
    radial_smooth = np.convolve(radial, np.ones(5) / 5, mode='same')

    # 找主峰
    peaks, _ = scipy.signal.find_peaks(radial_smooth, prominence=0.05)
    if len(peaks) == 0:
        return 0.0
    main = peaks[0]

    # 找主峰两侧最近极小值
    minima, _ = scipy.signal.find_peaks(-radial_smooth)
    lmin = minima[minima < main][-1] if np.any(minima < main) else 0
    rmin = minima[minima > main][0] if np.any(minima > main) else len(r_bins) - 1

    mask_ring = (r >= r_bins[lmin]) & (r < r_bins[rmin])
    return intensity[mask_ring].sum() / intensity.sum()


# ========= 主程序 =========
if __name__ == "__main__":
    # ---------- 基础参数 ----------
    wavelength = 532e-9          # m
    w0 = 1e-3                    # m
    ls, ps = [22], [5]           # OAM 指数 / 径向数
    N = 800                      # 网格
    grid_size = 2e-2             # m, 坐标范围 ±grid_size
    z_list = np.linspace(0, 20, 5)  # 传播距离序列 (m)

    dx = 2 * grid_size / N
    zR = np.pi * w0 ** 2 / wavelength

    # ---------- 空间 / 频域坐标 ----------
    x = np.linspace(-grid_size, grid_size, N)
    X, Y = np.meshgrid(x, x)
    r = np.hypot(X, Y)
    phi = np.arctan2(Y, X)

    fx = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(np.fft.fftshift(fx), np.fft.fftshift(fx))
    FSQ = FX ** 2 + FY ** 2

    # ---------- 初始场 ----------
    LG0 = LG_beam(r, phi, wavelength, w0, zR, 0, ls, ps, norm=True)

    # ---------- Figure & Subplots ----------
    fig, (ax_int, ax_phase) = plt.subplots(1, 2, figsize=(10, 5),
                                           constrained_layout=True)

    extent = [-grid_size * 1e3, grid_size * 1e3,
              -grid_size * 1e3, grid_size * 1e3]      # mm

    im_int = ax_int.imshow(np.abs(LG0) ** 2, extent=extent,
                           cmap='inferno', origin='lower')
    ax_int.set_title('Intensity')
    ax_int.set_xlabel('x (mm)')
    ax_int.set_ylabel('y (mm)')
    fig.colorbar(im_int, ax=ax_int, fraction=0.046).set_label('|E|²')

    im_phase = ax_phase.imshow(np.angle(LG0), extent=extent,
                               cmap='twilight', vmin=-np.pi, vmax=np.pi,
                               origin='lower')
    ax_phase.set_title('Phase')
    ax_phase.set_xlabel('x (mm)')
    ax_phase.set_ylabel('y (mm)')
    cbar_ph = fig.colorbar(im_phase, ax=ax_phase, fraction=0.046)
    cbar_ph.set_ticks([-np.pi, 0, np.pi])
    cbar_ph.set_ticklabels(['-π', '0', 'π'])

    # ---------- 动画更新函数 ----------
    def update(idx):
        z = z_list[idx]

        # 频域传播因子 (Fresnel)
        H = np.exp(1j * 2 * np.pi * z *
                   np.sqrt(np.maximum(0, 1 / wavelength ** 2 - FSQ)))

        U_f = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(LG0)))
        Uz = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(U_f * H)))

        I = np.abs(Uz) ** 2
        I /= I.max()
        phase = np.angle(Uz)

        # 更新图像
        im_int.set_data(I)
        im_phase.set_data(phase)
        ax_int.set_title(f'Intensity  z={z:.2f} m')

        # 主环能量
        ratio = analyze_main_ring_energy(I, r)
        if hasattr(ax_int, 'text_handle'):
            ax_int.text_handle.remove()
        ax_int.text_handle = ax_int.text(
            0.03, 0.97, f'Main lobe: {ratio:.2%}',
            color='w', transform=ax_int.transAxes,
            ha='left', va='top',
            bbox=dict(fc='black', alpha=0.4, ec='none')
        )
        return im_int, im_phase, ax_int.text_handle

    # ---------- 生成动画 ----------
    ani = FuncAnimation(fig, update, frames=len(z_list),
                        interval=400, blit=False, repeat=False)

    plt.show()

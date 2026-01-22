import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre
import tkinter as tk
from matplotlib.animation import FuncAnimation

def w_z(z, w0, zR):
    return w0 * np.sqrt(1 + (z / zR) ** 2)

def R_z(z, zR):
    return np.inf if z == 0 else np.abs(z) * (1 + (zR / z) ** 2)

def gouy_z(z, zR):
    return np.arctan(z / zR)

def LG_beam(r, phi, wavelength, w0, zR, z, ls, ps, normalization21=True):
    wz = w_z(z, w0, zR)
    Rz = R_z(z, zR)
    gouy = gouy_z(z, zR)
    norm = 1 / wz
    LGs = np.zeros(r.shape, dtype=complex)
    for i, l in enumerate(ls):
        if len(ps) != len(ls):
            raise ValueError("p and l must have same length!")
        p = ps[i]
        laguerre_poly = genlaguerre(p, np.abs(l))(2 * r ** 2 / wz ** 2)
        amplitude = (np.sqrt(2) * r / wz) ** np.abs(l) * laguerre_poly * np.exp(-r ** 2 / wz ** 2)
        phase = np.exp(1j * l * phi) * np.exp(-1j * 2 * np.pi / wavelength * r ** 2 / (2 * Rz)) * np.exp(1j * (2 * p + abs(l) + 1) * gouy)
        LG = norm * amplitude * phase
        if normalization21:
            LG = LG / np.max(np.abs(LG))
        LGs += LG
    return LGs
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

def analyze_main_ring_energy(intensity, r, plot_result=True):
    """
    统计主环（主峰左右两个极小值之间）能量占比
    """
    # 1. 径向平均
    r_flat = r.flatten()
    intensity_flat = intensity.flatten()
    r_bins = np.linspace(0, np.max(r), 1200)
    radial_mean = np.zeros_like(r_bins)
    for i in range(len(r_bins)-1):
        idx = (r_flat >= r_bins[i]) & (r_flat < r_bins[i+1])
        if np.any(idx):
            radial_mean[i] = intensity_flat[idx].mean()

    # 2. 平滑处理，减少小波动的影响
    smoothed_radial_mean = np.convolve(radial_mean, np.ones(5)/5, mode='same')

    # 3. 找所有极大值（主峰）
    peaks, _ = scipy.signal.find_peaks(smoothed_radial_mean, height=0.1, prominence=0.05)
    if len(peaks) == 0:
        print("未找到主环主峰！")
        return None
    main_peak_idx = peaks[0]

    # 4. 找所有极小值
    minima, _ = scipy.signal.find_peaks(-smoothed_radial_mean)
    # 找主峰左侧最近的极小值
    left_minima = minima[minima < main_peak_idx]
    if len(left_minima) == 0:
        left_idx = 0
    else:
        left_idx = left_minima[-1]
    # 找主峰右侧最近的极小值
    right_minima = minima[minima > main_peak_idx]
    if len(right_minima) == 0:
        right_idx = len(r_bins) - 1
    else:
        right_idx = right_minima[0]

    r_left = r_bins[left_idx]
    r_right = r_bins[right_idx]

    # 5. 能量积分
    mask = (r >= r_left) & (r < r_right)
    main_ring_energy = np.sum(intensity[mask])
    total_energy = np.sum(intensity)
    ratio = main_ring_energy / total_energy

    print(f"主环（主峰两侧极小值区间）能量占比：{ratio:.4f}")

    if plot_result:
        plt.figure()
        plt.plot(r_bins, radial_mean, label='径向平均')
        plt.plot(r_bins, smoothed_radial_mean, label='平滑处理后径向平均', linestyle='--')
        plt.axvline(r_left, color='g', linestyle='--', label='主环左极小值')
        plt.axvline(r_bins[main_peak_idx], color='r', linestyle='--', label='主峰')
        plt.axvline(r_right, color='b', linestyle='--', label='主环右极小值')
        plt.axvspan(r_left, r_right, color='orange', alpha=0.3, label='主环区间')
        plt.xlabel('r (m)')
        plt.ylabel('Average Intensity')
        plt.title('Radial Intensity Profile & Main Ring Area')
        plt.legend()
        plt.show()

    return ratio

def calculate_rms_radius(intensity, r):
    r_flat = r.flatten()
    inten_flat = intensity.flatten()
    return np.sqrt(np.sum(r_flat ** 2 * inten_flat) / np.sum(inten_flat))
def cal_p(l, MIPV2):
    p = np.round((0.5 * np.abs(l) + 1) ** 2 / (2 * MIPV2) - 0.5 * (np.abs(l) + 1))
    return p

def cal_MIPV(l, p):
    MIPV2 = (np.abs(l) + 2) ** 2 / (4 * (np.abs(l) + 2 * p + 1))
    return MIPV2
def calculate_main_ring_radius(intensity, r):
    N = r.shape[0]
    x = r[N//2, :]
    radial_profile = intensity.mean(axis=0)
    center = N // 2
    prof = radial_profile[center:]
    xprof = x[center:]
    main_idx = np.argmax(prof)
    return xprof[main_idx]


def simulate_and_animate(l_values, p_values, wavelength, w0,
                         N, pixel_size, z_gen,
                         z_start, z_end, num_frames):

    # ---------- 网格与初场 ----------
    grid_size = N * pixel_size / 2
    x = np.linspace(-grid_size, grid_size, N)
    X, Y = np.meshgrid(x, x)
    r   = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)
    zR  = np.pi * w0**2 / wavelength

    LG  = LG_beam(r, phi, wavelength, w0, zR,
                  z_gen, l_values, p_values)

    # 频域坐标
    fx = np.linspace(-1/(2*pixel_size), 1/(2*pixel_size), N)
    FX, FY = np.meshgrid(fx, fx)
    FSQ = FX**2 + FY**2

    z_list, all_z, rms_radii, main_radii = (
        np.linspace(z_start, z_end, num_frames), [], [], []
    )

    # ---------- Figure: 2×2 子图 ----------
    fig, axs = plt.subplots(2, 2, figsize=(12, 10),
                            constrained_layout=True)
    ax_int, ax_ph = axs[0]
    ax_rms, ax_main = axs[1]

    # (0,0) 强度
    im_int = ax_int.imshow(np.abs(LG)**2,
                           extent=[-grid_size*1e3, grid_size*1e3,
                                   -grid_size*1e3, grid_size*1e3],
                           cmap='inferno', origin='lower')
    ax_int.set_xlabel('x (mm)')
    ax_int.set_ylabel('y (mm)')
    fig.colorbar(im_int, ax=ax_int, fraction=0.046)
    ax_int.set_title('LG Intensity')

    # (0,1) 相位
    phase0 = np.angle(LG)
    im_ph = ax_ph.imshow(phase0,
                         extent=[-grid_size*1e3, grid_size*1e3,
                                 -grid_size*1e3, grid_size*1e3],
                         cmap='twilight', vmin=-np.pi, vmax=np.pi,
                         origin='lower')
    ax_ph.set_xlabel('x (mm)')
    ax_ph.set_ylabel('y (mm)')
    cbar_ph = fig.colorbar(im_ph, ax=ax_ph, fraction=0.046)
    cbar_ph.set_ticks([-np.pi, 0, np.pi])
    cbar_ph.set_ticklabels(['-π', '0', 'π'])
    ax_ph.set_title('LG Phase')

    # (1,0) RMS
    line_rms, = ax_rms.plot([], [], 'b-')
    ax_rms.set_xlim(z_start, z_end)
    ax_rms.set_xlabel('z (m)')
    ax_rms.set_ylabel('RMS Radius (m)')
    ax_rms.set_title('RMS Radius vs z')

    # (1,1) 主环
    line_main, = ax_main.plot([], [], 'g-')
    ax_main.set_xlim(z_start, z_end)
    ax_main.set_xlabel('z (m)')
    ax_main.set_ylabel('Main Ring Radius (m)')
    ax_main.set_title('Main Ring Radius vs z')

    # ---------- 帧函数 ----------
    def update(frame):
        z = z_list[frame]

        H  = np.exp(1j * 2*np.pi * z *
                    np.sqrt(np.maximum(0, 1/wavelength**2 - FSQ)))
        Uz_f = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(LG)))
        Uz   = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(Uz_f * H)))

        intensity = np.abs(Uz)**2
        intensity /= intensity.max()
        phase = np.angle(Uz)

        # 更新图像
        im_int.set_data(intensity)
        im_ph.set_data(phase)
        im_int.set_clim(0, 1)

        ax_int.set_title(f'Intensity  z={z:.3f} m')
        ax_ph.set_title(  f'Phase      z={z:.3f} m')

        # 能量文字（主环能量占比）
        ratio = analyze_main_ring_energy(intensity, r, plot_result=False)
        if hasattr(ax_int, 'text_handle'):
            ax_int.text_handle.remove()
        ax_int.text_handle = ax_int.text(
            0.02, 0.98, f'Main lobe: {ratio:.2%}',
            color='w', fontsize=11, transform=ax_int.transAxes,
            ha='left', va='top',
            bbox=dict(fc='black', ec='none', alpha=0.4))

        # 曲线数据
        rms_radii.append(calculate_rms_radius(intensity, r))
        main_radii.append(calculate_main_ring_radius(intensity, r))
        all_z.append(z)

        line_rms.set_data(all_z, rms_radii)
        line_main.set_data(all_z, main_radii)
        ax_rms.set_ylim(0, max(rms_radii)*1.2)
        ax_main.set_ylim(0, max(main_radii)*1.2)

        return [im_int, im_ph, line_rms, line_main]

    # ---------- 动画 ----------
    global animation_reference
    animation_reference = FuncAnimation(
        fig, update, frames=num_frames,
        interval=120, blit=False, repeat=False)
    plt.show()


def run_ui():
    root = tk.Tk()
    root.title("LG Beam Simulation with RMS and Main Ring Tracking")

    # Entries
    entries = {}
    default_values = {
        "l values (comma)": "1",
        "p values (comma)": "0",
        "Wavelength (m)": "532e-9",
        "Beam waist w0 (m)": "1e-3",
        "Grid size N": "512",
        "Pixel size (m)": "1e-4",
        "Generation z (m)": "0",
        "z1 start (m)": "0",
        "z2 end (m)": "10",
        "Number of frames": "30"
    }

    def update_zR(*args):
        try:
            wavelength = float(eval(entries["Wavelength (m)"].get()))
            w0 = float(eval(entries["Beam waist w0 (m)"].get()))
            zR = np.pi * w0 ** 2 / wavelength
            zR_label.config(text=f"zR = {zR:.4f} m")
        except:
            zR_label.config(text="zR = Invalid input")

    # Create labels and entries
    for i, (label_text, default) in enumerate(default_values.items()):
        tk.Label(root, text=label_text).grid(row=i, column=0)
        entry = tk.Entry(root)
        entry.insert(0, default)
        entry.grid(row=i, column=1)
        entries[label_text] = entry

    # zR display
    zR_label = tk.Label(root, text="zR =")
    zR_label.grid(row=len(default_values), column=0, columnspan=2)

    # Trace wavelength and w0 entries
    entries["Wavelength (m)"].bind('<KeyRelease>', update_zR)
    entries["Beam waist w0 (m)"].bind('<KeyRelease>', update_zR)

    def on_run():
        wavelength = float(eval(entries["Wavelength (m)"].get()))
        w0 = float(eval(entries["Beam waist w0 (m)"].get()))
        zR = np.pi * w0 ** 2 / wavelength

        l_values = [float(val.strip()) for val in entries["l values (comma)"].get().split(',')]
        p_values = [int(val.strip()) for val in entries["p values (comma)"].get().split(',')]
        N = int(entries["Grid size N"].get())
        pixel_size = float(eval(entries["Pixel size (m)"].get()))
        z_gen = float(eval(entries["Generation z (m)"].get(), {"zR": zR}))
        z1 = float(eval(entries["z1 start (m)"].get(), {"zR": zR}))
        z2 = float(eval(entries["z2 end (m)"].get(), {"zR": zR}))
        frames = int(entries["Number of frames"].get())

        simulate_and_animate(
            l_values, p_values, wavelength, w0,
            N, pixel_size, z_gen, z1, z2, frames
        )

    run_button = tk.Button(root, text="Run Simulation", command=on_run)
    run_button.grid(row=len(default_values) + 1, column=0, columnspan=2)

    update_zR()
    root.mainloop()

def cal_p(l, MIPV2):
    p = np.round((0.5 * np.abs(l) + 1) ** 2 / (2 * MIPV2) - 0.5 * (np.abs(l) + 1))
    return p

def cal_MIPV(l, p):
    MIPV2 = (np.abs(l) + 2) ** 2 / (4 * (np.abs(l) + 2 * p + 1))
    return MIPV2
if __name__ == "__main__":
    run_ui()



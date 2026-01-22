import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure  # 提供 marching_cubes 实现
from matplotlib.animation import FuncAnimation
def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_valid_split(dataset, train_ratio=0.8, seed=5201314):
    # 数据集分割为训练集和验证集
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    valid_size = dataset_size - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size],
                                                                 generator=torch.Generator().manual_seed(seed))
    return train_dataset, valid_dataset

def binarize(image):
    return (image > 0.5).float()  # 将像素值大于0.5的设为1，其余设为0
def filter_dataset(dataset, classes):
    indices = [i for i, (x, y) in enumerate(dataset) if y in classes]
    return Subset(dataset, indices)
def center_crop(tensor, crop_size):
    # 计算中心位置
    height, width = tensor.shape
    if crop_size > min(height, width):
        Warning("Crop size cannot be larger than the input tensor size.")
        return tensor
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    # 进行切片操作
    cropped_tensor = tensor[top:top + crop_size, left:left + crop_size]
    return cropped_tensor
def convert_to_DMDsize(img, DMD_size=(1080, 1920)):    # 补零适应DMD显示区域
    img_size = img.shape    # (H, W)
    left = (DMD_size[1] - img_size[1]) // 2
    right = DMD_size[1] - img_size[1] - left
    top = (DMD_size[0] - img_size[0]) // 2
    bottom = DMD_size[0] - img_size[0] - top
    img = torch.nn.functional.pad(img, (left, right, top, bottom), mode="constant", value=0)
    return img
def center_zeros(tensor, crop_size, ):
    # 计算中心位置
    tensor_ = torch.zeros_like(tensor)
    height, width = tensor.shape
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    # 进行切片操作
    tensor_[top:top + crop_size, left:left + crop_size] = tensor[top:top + crop_size, left:left + crop_size]
    return tensor_
def FWHM(y):
    """
    计算一维信号的半高全宽（含亚像素插值）

    参数:
        y (array): 一维强度数组

    返回:
        float: 半高全宽对应的精确像素数（含小数）
    """
    if not isinstance(y, np.ndarray) or len(y.shape) != 1:
        raise ValueError("输入需为一维numpy数组")

    # 生成隐含的x轴坐标
    x = np.arange(len(y))

    # 寻找最大值位置
    max_idx = np.argmax(y)
    max_val = y[max_idx]
    half_max = max_val / 2

    # 查找左侧交界点
    left_idx = max_idx
    while left_idx > 0 and y[left_idx] > half_max:
        left_idx -= 1

    # 查找右侧交界点
    right_idx = max_idx
    while right_idx < len(y) - 1 and y[right_idx] > half_max:
        right_idx += 1

    # 亚像素插值
    def interpolate(i1, i2):
        """线性插值计算精确交点"""
        if y[i2] == y[i1]:  # 防止除以零
            return x[i1]
        w = (half_max - y[i1]) / (y[i2] - y[i1])
        return x[i1] + w * (x[i2] - x[i1])

    # 计算精确交点
    x_left = interpolate(left_idx, left_idx + 1)
    x_right = interpolate(right_idx - 1, right_idx)

    return abs(x_right - x_left)

def show_3D_isosurface(value):
    size = value.shape[0]
    x, y, z = np.mgrid[:size, :size, :size]
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=[8, 8])
    ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
    # 等值面绘制
    verts, faces, _, _ = measure.marching_cubes(value, level=value.max() * 0.5)
    mesh = Poly3DCollection(verts[faces], alpha=0.5)
    mesh.set_facecolor(plt.cm.viridis(0.5))
    ax.add_collection3d(mesh)
    # 设置坐标轴范围
    ax.set_xlim(0, size), ax.set_ylim(0, size), ax.set_zlim(0, size)
    plt.tight_layout()
    plt.show()
    return
def show_3D_target(value, threshold=0.1):
    size = value.shape[0]
    xs, ys, zs = np.meshgrid(
        np.arange(size),  # x轴坐标
        np.arange(size),  # y轴坐标
        np.arange(size),  # z轴坐标
        indexing='xy'  # 确保索引顺序一致
    )
    value = value/value.max()
    loc = value.flatten() > threshold
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=[7, 7])
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    # 强制设置坐标轴范围为 0 到 19
    ax.set_xlim([0, size - 1])
    ax.set_ylim([0, size - 1])
    ax.set_zlim([0, size - 1])
    # 或强制统一 z-order（替代方案）
    scatter = ax.scatter(
        xs.flatten()[loc],
        ys.flatten()[loc],
        zs.flatten()[loc],
        c=value.flatten()[loc],
        cmap='viridis',
        zorder=1,  # 统一渲染层级
        s=50
    )
    # 添加颜色条
    fig.colorbar(scatter, shrink=0.5, aspect=5)
    plt.show()
    return
def show_3D_projections(value):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # X-Y 平面最大值投影
    axs[0].imshow(np.max(value, axis=2), cmap='viridis')
    axs[0].set_title("X-Y Projection")

    # X-Z 平面最大值投影
    axs[1].imshow(np.max(value, axis=1), cmap='viridis')
    axs[1].set_title("X-Z Projection")

    # Y-Z 平面最大值投影
    axs[2].imshow(np.max(value, axis=0), cmap='viridis')
    axs[2].set_title("Y-Z Projection")

    plt.tight_layout()
    plt.show()


def animate_3D_slices(value):
    fig, ax = plt.subplots()
    def update(frame):
        ax.clear()
        ax.imshow(value[:, :, frame], cmap='viridis', vmax = value.max())
        ax.set_title(f"Slice {frame}")

    ani = FuncAnimation(
        fig, update,
        frames=value.shape[2],
        interval=500,
        blit=False  # 关键改动：禁用 blit
    )
    plt.show()
    return ani
def shift_tensor_fft(x, shiftx=0.0, shifty=0.0):
    """
    使用傅里叶变换在横向和纵向上进行亚像素偏移。

    参数:
    x (torch.Tensor): 输入 tensor，假设为 2D，float 类型。
    shiftx (float): 横向偏移（像素），正值右移，负值左移。
    shifty (float): 纵向偏移（像素），正值下移，负值上移。

    返回:
    torch.Tensor: 偏移后的 tensor。
    """
    if x.ndim == 2:
        raise ValueError("只支持 2D tensor。")

    H, W = x.shape
    # 生成频率坐标
    fy = torch.fft.fftfreq(H, d=1.0).to(x.device).unsqueeze(1)  # [H, 1]
    fx = torch.fft.fftfreq(W, d=1.0).to(x.device).unsqueeze(0)  # [1, W]

    # 计算相位因子
    phase_shift = torch.exp(-2j * torch.pi * (fx * shiftx + fy * shifty))  # [H, W]

    # FFT、乘相位、逆FFT
    X_fft = torch.fft.fft2(x)
    X_fft_shifted = X_fft * phase_shift
    x_shifted = torch.fft.ifft2(X_fft_shifted)  # 取实部

    return x_shifted
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# ===== 1) 复场 -> RGB（主图用） =====
def complex_to_rgb(Z, vmin=0.0, vmax=1, gamma=1.0, floor=0.0):
    phase = np.angle(Z)
    mag   = np.abs(Z)
    # if vmax is None:
    #     vmax = np.percentile(mag, 99)
    m = np.clip((mag - vmin) / (vmax - vmin + 1e-12), 0, 1)**gamma
    H = (phase + np.pi) / (2*np.pi)
    S = np.ones_like(m)
    V = floor + (1 - floor) * m
    rgb = hsv_to_rgb(np.stack([H, S, V], axis=-1))
    return rgb

# ===== 2) 相位-幅度圆盘（独立 Figure） =====
def fig_complex_legend(gamma=1.0, floor=0.0, N=200, figsize=(2.2,2.2), dpi=200):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x = np.linspace(-1, 1, N); y = np.linspace(-1, 1, N)
    xx, yy = np.meshgrid(x, y)
    rr = np.hypot(xx, yy)
    ang = np.arctan2(yy, xx)

    H = (ang + np.pi) / (2*np.pi)          # 相位→色相
    S = np.ones_like(rr)
    V = floor + (1 - floor) * np.clip(rr, 0, 1)**gamma  # 半径→亮度
    img = hsv_to_rgb(np.stack([H, S, V], axis=-1))
    img[rr > 1] = 1.0                       # 圆外留白

    ax.imshow(img, origin='lower', extent=[-1,1,-1,1])
    # 简洁十字与标签
    # ax.plot([-1,1],[0,0], color='k', lw=0.8, alpha=0.5)
    # ax.plot([0,0],[-1,1], color='k', lw=0.8, alpha=0.5)
    # ax.text(0, 1.08, "Imag", ha='center', va='bottom', fontsize=10)
    # ax.text(1.08, 0, "Real", ha='left', va='center', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-1.2, 1.5); ax.set_ylim(-1.2, 1.2)
    plt.axis('off')
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig, ax
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

def bar3d_interp(ax, Z, dx=0.8, dy=0.8, n_layers=24, cmap='viridis'):
    """
    Z: 2D 矩阵，行=真实标签，列=预测类别
    每根柱子切 n_layers 层，颜色随绝对高度渐变（拟合 MATLAB 的 interp 效果）
    """
    Z = np.asarray(Z)
    Hmax = float(Z.max()) if Z.size and Z.max() > 0 else 1.0
    norm = colors.Normalize(vmin=0.0, vmax=Hmax)
    cmap = cm.get_cmap(cmap)

    nx, ny = Z.shape
    for i in range(nx):
        for j in range(ny):
            h = float(Z[i, j])
            if h <= 0:
                continue
            edges = np.linspace(0.0, h, n_layers + 1)
            z0s   = edges[:-1]
            dzs   = np.diff(edges)
            zmid  = z0s + 0.5 * dzs
            cols  = cmap(norm(zmid))
            # 在同一 (x,y) 上叠加很多薄柱
            x = np.full_like(z0s, i, dtype=float)
            y = np.full_like(z0s, j, dtype=float)
            ax.bar3d(x, y, z0s, dx, dy, dzs, color=cols, shade=True, zsort='average')

# todo: 相位转化为SLM灰度图
'''
调用方式
------------
相位矩阵（弧度）→ 8-bit PNG（SLM）
to_slm_8bit_png(phase, "slm_phase.png", src_type="phase_rad")
------------
读入一张图（本来就是0~255编码）→ 强制保存成标准 8-bit PNG
to_slm_8bit_png("input_any.png", "slm_ready.png",
                src_type="image", image_meaning="encoded_0_255")
------------
如果图片其实表示“相位”（例如灰度 0..255 对应 0..2π）
to_slm_8bit_png("phase_image.png", "slm_phase.png",
                src_type="image", image_meaning="phase_rad_0_2pi")
'''
from pathlib import Path
from PIL import Image
_TWO_PI = 2.0 * np.pi
def phase_to_uint8(phase_rad: np.ndarray,
                   *,
                   wrap: bool = True,
                   quantize: str = "floor") -> np.ndarray:
    """
    将相位(弧度)映射到 uint8 灰度(0..255)，用于 SLM。

    phase_rad: 相位矩阵，单位弧度，支持任意范围（可负/可>2pi）
    wrap: 是否对 2π 取模到 [0, 2π)
    quantize:
      - "floor": level = floor(phase/(2π)*256)  -> 256级，推荐
      - "round": level = round(phase/(2π)*255)  -> 也常见，但步长略不同
    """
    p = np.asarray(phase_rad, dtype=np.float64)

    # NaN/Inf 处理：变成0相位
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

    if wrap:
        p = np.mod(p, _TWO_PI)  # [0, 2π)

    if quantize == "floor":
        # 256级：0..255。p∈[0,2π) -> scaled∈[0,256)
        scaled = p / _TWO_PI * 256.0
        u8 = np.floor(scaled).astype(np.uint8)
    elif quantize == "round":
        # 另一种常见做法：0..255 对应 0..2π（近似）
        scaled = p / _TWO_PI * 255.0
        u8 = np.rint(np.clip(scaled, 0.0, 255.0)).astype(np.uint8)
    else:
        raise ValueError("quantize must be 'floor' or 'round'")

    return u8


def load_image_as_gray_array(path: str | Path) -> np.ndarray:
    """
    读入任意图片，返回灰度数组（float64），范围保持原始数值（不自动归一化）。
    """
    with Image.open(path) as im:
        im.load()
        # SLM 通常用单通道；如果是RGB就转灰度
        g = im.convert("L")
        return np.asarray(g, dtype=np.float64)


def to_slm_8bit_png(src,
                    out_path: str | Path,
                    *,
                    src_type: str = "auto",
                    image_meaning: str = "auto",
                    wrap: bool = True,
                    quantize: str = "floor",
                    save_optimize: bool = True,
                    correct_phase: str = None,
                    correct_phase_meaning: str = 'auto',
                    ) -> Path:
    """
    将输入转换为 SLM 可用的 8-bit PNG（灰度 L，0..255）

    src:
      - np.ndarray：相位矩阵（弧度）或数值矩阵（取决于 src_type）
      - str/Path：图片文件路径

    src_type:
      - "auto":  路径 -> 按 image_meaning；数组 -> 认为是相位弧度
      - "phase_rad": 数组按弧度相位处理
      - "image":  src 是路径

    image_meaning（仅当 src 是图片路径时生效）:
      - "encoded_0_255": 图片已经是 SLM 编码（0..255），直接保存为8-bit PNG
      - "normalized_0_1": 图片灰度在 0..1，先乘 2π 当相位，再量化
      - "phase_rad_0_2pi": 图片灰度在 0..255 但表示 0..2π 的相位（常见：把相位存成8-bit图）
      - "auto": 自动猜：大多数情况下，你给的图就是要投到SLM的编码图
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 判定输入类型
    if src_type == "auto":
        if isinstance(src, (str, Path)):
            src_type = "image"
        else:
            src_type = "phase_rad"

    # 加载和处理输入源
    if src_type == "phase_rad":
        u8 = phase_to_uint8(src, wrap=wrap, quantize=quantize)

    elif src_type == "image":
        g = load_image_as_gray_array(src)
        gmin, gmax = float(np.min(g)), float(np.max(g))

        if image_meaning == "auto":
            # 默认策略：大多数情况下，你给的图就是要投到SLM的编码图
            image_meaning = "encoded_0_255"

        if image_meaning == "encoded_0_255":
            u8 = np.clip(g, 0.0, 255.0).astype(np.uint8)

        elif image_meaning == "normalized_0_1":
            # 0..1 -> 相位弧度 -> uint8
            p = np.clip(g, 0.0, 1.0) * _TWO_PI
            u8 = phase_to_uint8(p, wrap=wrap, quantize=quantize)

        elif image_meaning == "phase_rad_0_2pi":
            # 0..255 表示 0..2π 的相位（常见保存方式）
            p = (np.clip(g, 0.0, 255.0) / 256.0) * _TWO_PI
            u8 = phase_to_uint8(p, wrap=wrap, quantize=quantize)

        else:
            raise ValueError(f"Unknown image_meaning: {image_meaning}")

    else:
        raise ValueError("src_type must be 'auto', 'phase_rad', or 'image'")

    # 如果有矫正相位，需要进行叠加
    if correct_phase is not None:
        if isinstance(correct_phase, (str, Path)):
            correct_phase = load_image_as_gray_array(correct_phase)  # 假设矫正相位是图像路径
        if correct_phase_meaning == "auto":
            correct_phase_meaning = "encoded_0_255"  # 默认策略

        # 确保 correct_phase 和 u8 具有相同的意义
        if correct_phase_meaning == "encoded_0_255":
            correct_phase = np.mod(correct_phase, 256).astype(np.uint8)
        elif correct_phase_meaning == "normalized_0_1":
            correct_phase = np.mod(correct_phase, 1).astype(np.float32)
            correct_phase *= _TWO_PI
        elif correct_phase_meaning == "phase_rad_0_2pi":
            correct_phase = np.mod(correct_phase, 256).astype(np.uint8)
            correct_phase = (correct_phase / 256.0) * _TWO_PI
        else:
            raise ValueError(f"Unknown correct_phase_meaning: {correct_phase_meaning}")

        # 叠加相位
        if image_meaning == "encoded_0_255":
            u8 = np.mod(u8 + correct_phase, 256).astype(np.uint8)  # 包裹到 0-255 范围
        elif image_meaning == "normalized_0_1":
            u8 = np.mod(u8 + correct_phase, 1).astype(np.float32)  # 包裹到 0-1 范围
        elif image_meaning == "phase_rad_0_2pi":
            u8 = np.mod(u8 + correct_phase, _TWO_PI).astype(np.float32)  # 包裹到 0-2π 范围


    # 将结果保存为 PNG
    img = Image.fromarray(u8, mode="L")
    img.save(out_path, format="PNG", optimize=save_optimize)
    return out_path




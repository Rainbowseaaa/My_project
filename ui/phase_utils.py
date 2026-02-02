import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.io import loadmat

_TWO_PI = 2.0 * math.pi


class PhaseFileError(ValueError):
    pass


def _normalize_phase(phase: np.ndarray) -> np.ndarray:
    return np.mod(phase, _TWO_PI)


def load_phase_file(path: str, meaning: str = "auto", mat_key: str = "phase") -> np.ndarray:
    """
    读取相位文件，统一返回单位为弧度的相位矩阵。

    - .npy: float 相位 (rad) 或 0..255 编码
    - .png/.bmp: 默认灰度 0..255 -> [0, 2π)
    - .mat: 默认读取 mat_key
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".npy":
        data = np.load(file_path)
        if meaning == "auto":
            if data.dtype in (np.uint8, np.uint16):
                meaning = "encoded_0_255"
            else:
                meaning = "phase_rad"
        if meaning == "encoded_0_255":
            return (data.astype(np.float64) / 255.0) * _TWO_PI
        if meaning == "phase_rad":
            return data.astype(np.float64)
        raise PhaseFileError(f"不支持的 npy meaning: {meaning}")

    if suffix in {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}:
        img = Image.open(file_path).convert("L")
        data = np.asarray(img, dtype=np.float64)
        if meaning == "auto":
            meaning = "encoded_0_255"
        if meaning == "encoded_0_255":
            return (data / 255.0) * _TWO_PI
        if meaning == "phase_rad":
            return data
        raise PhaseFileError(f"不支持的图像 meaning: {meaning}")

    if suffix == ".mat":
        mat = loadmat(file_path)
        if mat_key not in mat:
            raise PhaseFileError(f"MAT 文件未找到 key: {mat_key}")
        data = mat[mat_key]
        if meaning == "auto":
            meaning = "phase_rad"
        if meaning == "encoded_0_255":
            return (data.astype(np.float64) / 255.0) * _TWO_PI
        if meaning == "phase_rad":
            return data.astype(np.float64)
        raise PhaseFileError(f"不支持的 mat meaning: {meaning}")

    raise PhaseFileError(f"不支持的相位文件格式: {file_path}")


def phase_to_uint8(phase: np.ndarray) -> np.ndarray:
    phase_wrapped = _normalize_phase(phase)
    return np.round(phase_wrapped / _TWO_PI * 255).astype(np.uint8)


def resize_phase(phase: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if phase.shape == shape:
        return phase
    img = Image.fromarray(phase.astype(np.float32))
    img = img.resize((shape[1], shape[0]), Image.BILINEAR)
    return np.array(img, dtype=np.float64)


def apply_compensation(phase: np.ndarray, comp: Optional[np.ndarray]) -> np.ndarray:
    if comp is None:
        return phase
    return _normalize_phase(phase + comp)


def load_compensation(path: str, shape: Tuple[int, int], meaning: str = "encoded_0_255") -> Optional[np.ndarray]:
    if not path:
        return None
    comp = load_phase_file(path, meaning=meaning)
    return resize_phase(comp, shape)


# [ui/phase_utils.py]
# 只需更新 compose_layers 函数，其他保持不变，但为了方便，这里给出 compose_layers 的完整代码

def compose_layers(
        slm_shape: Tuple[int, int],
        window_size: Tuple[int, int],
        layers: List[Dict],  # 字典现在包含 'flip_h', 'flip_v'
        outside_mode: str = "zero",
        block_mode: str = "checkerboard",
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    height, width = slm_shape
    win_h, win_w = window_size

    if outside_mode == "zero":
        base = np.zeros((height, width), dtype=np.float64)
    elif outside_mode == "block":
        from calibrate_insitu import blocking_phase
        base = blocking_phase(slm_shape, block_mode).astype(np.float64)
    else:
        raise ValueError(f"未知 outside_mode: {outside_mode}")

    rects: List[Tuple[int, int, int, int]] = []

    for layer in layers:
        if not layer.get("enabled", True):
            rects.append((0, 0, 0, 0))
            continue

        phase = layer["phase"]

        # 处理单独层的翻转
        if layer.get("flip_h", False):
            phase = np.fliplr(phase)
        if layer.get("flip_v", False):
            phase = np.flipud(phase)

        cx, cy = layer["center"]
        top = int(round(cy - win_h / 2))
        left = int(round(cx - win_w / 2))
        bottom = top + win_h
        right = left + win_w

        top_clip = max(top, 0)
        left_clip = max(left, 0)
        bottom_clip = min(bottom, height)
        right_clip = min(right, width)

        if top_clip >= bottom_clip or left_clip >= right_clip:
            rects.append((0, 0, 0, 0))
            continue

        phase_window = resize_phase(phase, (win_h, win_w))
        phase_crop = phase_window[(top_clip - top):(bottom_clip - top), (left_clip - left):(right_clip - left)]
        base[top_clip:bottom_clip, left_clip:right_clip] = _normalize_phase(phase_crop)
        rects.append((left_clip, top_clip, right_clip - left_clip, bottom_clip - top_clip))

    return _normalize_phase(base), rects

def make_preview_image(
    slm_shape: Tuple[int, int],
    rects: List[Tuple[int, int, int, int]],
    centers: List[Tuple[int, int]],
    enabled: List[bool],
    base_phase: Optional[np.ndarray] = None,
    scale: float = 0.2,
) -> Image.Image:
    height, width = slm_shape
    preview_w = max(1, int(width * scale))
    preview_h = max(1, int(height * scale))

    if base_phase is not None:
        phase = resize_phase(base_phase, slm_shape)
        phase_u8 = phase_to_uint8(phase)
        img = Image.fromarray(phase_u8, mode="L").resize((preview_w, preview_h), Image.BILINEAR).convert("RGB")
    else:
        img = Image.new("RGB", (preview_w, preview_h), color=(20, 20, 20))

    from PIL import ImageDraw

    painter = ImageDraw.Draw(img)
    colors = [(255, 80, 80), (80, 255, 80), (80, 120, 255), (255, 200, 80)]

    for idx, rect in enumerate(rects):
        if rect == (0, 0, 0, 0):
            continue
        left, top, w, h = rect
        left_s = int(left * scale)
        top_s = int(top * scale)
        right_s = int((left + w) * scale)
        bottom_s = int((top + h) * scale)
        color = colors[idx % len(colors)]
        if not enabled[idx]:
            color = (120, 120, 120)
        painter.rectangle([left_s, top_s, right_s, bottom_s], outline=color, width=2)

    for idx, center in enumerate(centers):
        cx, cy = center
        cx_s = int(cx * scale)
        cy_s = int(cy * scale)
        color = colors[idx % len(colors)]
        if not enabled[idx]:
            color = (120, 120, 120)
        painter.ellipse([cx_s - 3, cy_s - 3, cx_s + 3, cy_s + 3], outline=color, width=2)

    return img


def generate_window_grating_phase(
    slm_shape: Tuple[int, int],
    center: Tuple[float, float],
    window_size_px: float,
    grating_period_px: float,
    window_shape: str = "square",
) -> np.ndarray:
    height, width = slm_shape
    period = float(grating_period_px)
    if period == 0:
        raise ValueError("光栅周期不能为 0")
    period_abs = abs(period)
    yy, xx = np.mgrid[0:height, 0:width]
    grating = (2 * np.pi * (xx / period_abs)) % (2 * np.pi)
    if period < 0:
        grating = -grating

    cx, cy = center
    half = max(float(window_size_px) / 2.0, 0.0)
    if window_shape == "circle":
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= half ** 2
    else:
        mask = (abs(xx - cx) <= half) & (abs(yy - cy) <= half)
    phase = np.zeros_like(grating)
    phase[mask] = grating[mask]
    return phase


def generate_lens_phase(
    slm_shape: Tuple[int, int],
    center: Tuple[float, float],
    focus_mm: float,
    pixel_pitch_um: float = 8.0,
    wavelength_nm: float = 532.0,
) -> np.ndarray:
    height, width = slm_shape
    focus_mm = float(focus_mm)
    if focus_mm == 0:
        return np.zeros((height, width), dtype=np.float64)
    pitch = float(pixel_pitch_um) * 1e-6
    wavelength = float(wavelength_nm) * 1e-9
    f = focus_mm * 1e-3

    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = center
    x = (xx - cx) * pitch
    y = (yy - cy) * pitch
    r2 = x ** 2 + y ** 2
    phase = (-np.pi / (wavelength * f)) * r2
    return phase


def load_hologram_file(path: str, shape: Tuple[int, int]) -> np.ndarray:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        data = np.load(file_path)
        if np.iscomplexobj(data):
            data = np.abs(data)
        if data.dtype != np.uint8:
            data = np.clip(data, 0, 255)
            data = data.astype(np.uint8)
    else:
        img = Image.open(file_path).convert("L")
        data = np.asarray(img, dtype=np.uint8)
    img = Image.fromarray(data)
    img = img.resize((shape[1], shape[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def load_field_file(path: str, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        data = np.load(file_path)
        if np.iscomplexobj(data):
            amp = np.abs(data).astype(np.float64)
            phase = np.angle(data).astype(np.float64)
        else:
            amp = data.astype(np.float64)
            phase = np.zeros_like(amp)
    else:
        img = Image.open(file_path).convert("L")
        amp = np.asarray(img, dtype=np.float64) / 255.0
        phase = np.zeros_like(amp)

    amp_img = Image.fromarray(amp.astype(np.float32))
    amp_img = amp_img.resize((shape[1], shape[0]), Image.BILINEAR)
    amp = np.asarray(amp_img, dtype=np.float64)

    phase_img = Image.fromarray(phase.astype(np.float32))
    phase_img = phase_img.resize((shape[1], shape[0]), Image.BILINEAR)
    phase = np.asarray(phase_img, dtype=np.float64)
    return amp, phase


def _parse_int_list(value) -> List[int]:
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",") if v.strip()]
        return [int(v) for v in items] if items else []
    return [int(value)]


def generate_lg_field(shape: Tuple[int, int], w0: float, p, l) -> Tuple[np.ndarray, np.ndarray]:
    from functions import OAM_gen

    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    cx = width / 2
    cy = height / 2
    x = xx - cx
    y = yy - cy
    r = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    w0_px = max(float(w0), 1e-6)
    wavelength = 1.0
    zR = np.pi * w0_px ** 2 / wavelength
    p_list = _parse_int_list(p)
    l_list = _parse_int_list(l)
    if len(p_list) != len(l_list):
        raise ValueError("LG 模式: p 与 l 数量不一致")
    if not p_list:
        p_list = [0]
        l_list = [0]
    field = OAM_gen.LG_beam(r, phi, wavelength, w0_px, zR, 0.0, l_list, p_list, norm=True)
    amp = np.abs(field).astype(np.float64)
    phase = np.angle(field).astype(np.float64)
    return amp, phase


def render_letter_field(shape: Tuple[int, int], letter: str) -> Tuple[np.ndarray, np.ndarray]:
    height, width = shape
    img = Image.new("L", (width, height), color=0)
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=int(min(width, height) * 0.6))
    except Exception:
        font = ImageFont.load_default()
    text = (letter or "A")[0]
    try:
        draw.text((width / 2, height / 2), text, fill=255, font=font, anchor="mm")
    except TypeError:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        pos = ((width - text_w) // 2, (height - text_h) // 2)
        draw.text(pos, text, fill=255, font=font)
    amp = np.asarray(img, dtype=np.float64) / 255.0
    phase = np.zeros_like(amp)
    return amp, phase


def generate_focus_field(
    shape: Tuple[int, int],
    focus_mm: float,
    diameter_px: float,
    pixel_pitch_um: float = 8.0,
    wavelength_nm: float = 532.0,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = shape
    focus_mm = float(focus_mm)
    diameter_px = float(diameter_px)
    pixel_pitch_um = float(pixel_pitch_um)
    wavelength_nm = float(wavelength_nm)

    if focus_mm == 0:
        amp = np.ones((height, width), dtype=np.float64)
        if diameter_px > 0:
            yy, xx = np.mgrid[0:height, 0:width]
            dx = xx - (width / 2)
            dy = yy - (height / 2)
            radius_px = diameter_px / 2.0
            amp = ((dx ** 2 + dy ** 2) <= radius_px ** 2).astype(np.float64)
        phase = np.zeros((height, width), dtype=np.float64)
        return amp, phase

    pitch = pixel_pitch_um * 1e-6
    wavelength = wavelength_nm * 1e-9
    f = focus_mm * 1e-3

    yy, xx = np.mgrid[0:height, 0:width]
    cx = width / 2
    cy = height / 2
    dx = xx - cx
    dy = yy - cy
    x = dx * pitch
    y = dy * pitch
    r2 = x ** 2 + y ** 2

    phase = (-np.pi / (wavelength * f)) * r2
    if diameter_px > 0:
        radius_px = diameter_px / 2.0
        mask = (dx ** 2 + dy ** 2) <= radius_px ** 2
        amp = mask.astype(np.float64)
        phase = phase * mask
    else:
        amp = np.ones((height, width), dtype=np.float64)
    return amp, phase


def resize_and_embed(
    amp: np.ndarray,
    phase: np.ndarray,
    slm_shape: Tuple[int, int],
    target_size: Tuple[int, int] | None,
) -> Tuple[np.ndarray, np.ndarray]:
    if target_size is None:
        target_w, target_h = amp.shape[1], amp.shape[0]
    else:
        target_w, target_h = target_size
        if target_w <= 0 or target_h <= 0:
            target_w, target_h = amp.shape[1], amp.shape[0]
    target_w = int(target_w)
    target_h = int(target_h)
    if (target_h, target_w) == amp.shape and amp.shape == slm_shape:
        return amp, phase
    amp_img = Image.fromarray(amp.astype(np.float32))
    amp_img = amp_img.resize((target_w, target_h), Image.BILINEAR)
    phase_img = Image.fromarray(phase.astype(np.float32))
    phase_img = phase_img.resize((target_w, target_h), Image.BILINEAR)
    amp_resized = np.asarray(amp_img, dtype=np.float64)
    phase_resized = np.asarray(phase_img, dtype=np.float64)

    base_amp = np.zeros(slm_shape, dtype=np.float64)
    base_phase = np.zeros(slm_shape, dtype=np.float64)
    top = max((slm_shape[0] - target_h) // 2, 0)
    left = max((slm_shape[1] - target_w) // 2, 0)
    bottom = min(top + target_h, slm_shape[0])
    right = min(left + target_w, slm_shape[1])
    amp_crop = amp_resized[:bottom - top, :right - left]
    phase_crop = phase_resized[:bottom - top, :right - left]
    base_amp[top:bottom, left:right] = amp_crop
    base_phase[top:bottom, left:right] = phase_crop
    return base_amp, base_phase


def _download_file(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, dest.open("wb") as f:
        f.write(resp.read())


def _maybe_download_mnist(data_dir: Path, fashion: bool) -> None:
    bases = [
        "https://fashion-mnist.s3.amazonaws.com" if fashion else "https://storage.googleapis.com/cvdf-datasets/mnist",
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com" if fashion else "http://yann.lecun.com/exdb/mnist",
    ]
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    for name in files:
        dest = data_dir / name
        if dest.exists():
            continue
        last_exc = None
        for base in bases:
            try:
                _download_file(f"{base}/{name}", dest)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc


def _load_idx_images(path: Path) -> np.ndarray:
    import gzip

    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    if magic != 2051:
        raise RuntimeError(f"IDX magic 错误: {magic}")
    num = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")
    images = np.frombuffer(data, dtype=np.uint8, offset=16)
    return images.reshape(num, rows, cols)


def load_mnist_sample(index: int, fashion: bool = False, data_dir: str | None = None) -> np.ndarray:
    root = Path(data_dir or "data/mnist")
    dataset_dir = root / ("fashion" if fashion else "mnist")
    try:
        _maybe_download_mnist(dataset_dir, fashion)
        images = _load_idx_images(dataset_dir / "train-images-idx3-ubyte.gz")
    except Exception as exc:
        raise RuntimeError(f"MNIST 数据集不可用: {exc}") from exc
    idx = int(index) % images.shape[0]
    return images[idx]

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


def compose_layers(
    slm_shape: Tuple[int, int],
    window_size: Tuple[int, int],
    layers: List[Dict],
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


def generate_lg_field(shape: Tuple[int, int], w0: float, p: int, l: int) -> Tuple[np.ndarray, np.ndarray]:
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    cx = width / 2
    cy = height / 2
    x = xx - cx
    y = yy - cy
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    rho = np.sqrt(2) * r / max(w0, 1e-6)
    try:
        from scipy.special import genlaguerre

        lag = genlaguerre(p, abs(l))(rho ** 2)
    except Exception:
        lag = np.ones_like(rho)
    amp = (rho ** abs(l)) * lag * np.exp(-(rho ** 2) / 2)
    amp = amp / (np.max(amp) + 1e-9)
    phase = l * theta
    return amp.astype(np.float64), phase.astype(np.float64)


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
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pos = ((width - text_w) // 2, (height - text_h) // 2)
    draw.text(pos, text, fill=255, font=font)
    amp = np.asarray(img, dtype=np.float64) / 255.0
    phase = np.zeros_like(amp)
    return amp, phase


def load_mnist_sample(index: int, fashion: bool = False) -> np.ndarray:
    try:
        if fashion:
            from tensorflow.keras.datasets import fashion_mnist

            (x_train, _), _ = fashion_mnist.load_data()
        else:
            from tensorflow.keras.datasets import mnist

            (x_train, _), _ = mnist.load_data()
    except Exception as exc:
        raise RuntimeError(f"MNIST 数据集不可用: {exc}") from exc
    idx = int(index) % x_train.shape[0]
    return x_train[idx]

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
原位标定程序：开窗放行 + Zi 测量
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class ROI:
    x_min: int
    x_max: int
    y_min: int
    y_max: int

    def clamp(self, width: int, height: int) -> "ROI":
        return ROI(
            x_min=max(0, min(self.x_min, width - 1)),
            x_max=max(0, min(self.x_max, width - 1)),
            y_min=max(0, min(self.y_min, height - 1)),
            y_max=max(0, min(self.y_max, height - 1)),
        )

    def to_slices(self) -> Tuple[slice, slice]:
        return slice(self.y_min, self.y_max + 1), slice(self.x_min, self.x_max + 1)


def load_config(path: str) -> Dict:
    config_path = Path(path)
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        from importlib import import_module

        yaml = import_module("yaml")
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    if config_path.suffix.lower() == ".json":
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"不支持的配置文件格式: {config_path}")


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def roi_from_dict(data: Dict) -> ROI:
    return ROI(
        x_min=int(data["x_min"]),
        x_max=int(data["x_max"]),
        y_min=int(data["y_min"]),
        y_max=int(data["y_max"]),
    )


def phase_to_uint8(phase: np.ndarray) -> np.ndarray:
    phase_wrapped = np.mod(phase, 2 * np.pi)
    return np.round(phase_wrapped / (2 * np.pi) * 255).astype(np.uint8)


def load_compensation(path: str, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if not path:
        return None
    img = Image.open(path).convert("L")
    if img.size != (shape[1], shape[0]):
        img = img.resize((shape[1], shape[0]), Image.BILINEAR)
    comp_u8 = np.array(img, dtype=np.uint8)
    return comp_u8.astype(np.float32) / 255.0 * 2 * np.pi


def load_compensation_u8(path: str, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if not path:
        return None
    img = Image.open(path).convert("L")
    if img.size != (shape[1], shape[0]):
        img = img.resize((shape[1], shape[0]), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def apply_compensation(phase: np.ndarray, comp: Optional[np.ndarray]) -> np.ndarray:
    if comp is None:
        return phase
    return np.mod(phase + comp, 2 * np.pi)


def gaussian_pattern(shape: Tuple[int, int], sigma_px: float, peak: float) -> np.ndarray:
    height, width = shape
    y = np.arange(height) - height / 2
    x = np.arange(width) - width / 2
    yy, xx = np.meshgrid(y, x, indexing="ij")
    gauss = peak * np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_px ** 2))
    return np.clip(gauss, 0, 255).astype(np.uint8)


def blocking_phase(shape: Tuple[int, int], mode: str) -> np.ndarray:
    height, width = shape
    if mode == "checkerboard":
        y = np.arange(height)[:, None]
        x = np.arange(width)[None, :]
        return np.pi * ((x + y) % 2)
    if mode == "blaze":
        x = np.arange(width)[None, :]
        return 2 * np.pi * (x % 2)
    raise ValueError(f"未知阻断模式: {mode}")


def window_mask(shape: Tuple[int, int], center: Tuple[int, int], size_px: int, shape_type: str) -> np.ndarray:
    height, width = shape
    cx, cy = center
    yy, xx = np.ogrid[:height, :width]
    if shape_type == "circle":
        radius = size_px / 2
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    if shape_type == "square":
        half = int(size_px / 2)
        return (np.abs(xx - cx) <= half) & (np.abs(yy - cy) <= half)
    raise ValueError(f"未知窗口形状: {shape_type}")


def lens_phase(shape: Tuple[int, int], center: Tuple[int, int],
               pixel_pitch: Tuple[float, float], wavelength: float, f: float) -> np.ndarray:
    height, width = shape
    cx, cy = center
    px, py = pixel_pitch
    y = (np.arange(height) - cy) * py
    x = (np.arange(width) - cx) * px
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return -np.pi / (wavelength * f) * (xx ** 2 + yy ** 2)


def compute_metric(img: np.ndarray, roi_cam: ROI, metric_type: str) -> float:
    roi_img = img[roi_cam.to_slices()]
    if metric_type == "roi_sum":
        return float(np.sum(roi_img))
    full_sum = float(np.sum(img)) + 1e-9
    return float(np.sum(roi_img) / full_sum)


def focus_metric(img: np.ndarray, roi_focus: ROI, metric_type: str) -> float:
    roi_img = img[roi_focus.to_slices()].astype(np.float64)
    if metric_type == "peak":
        return float(np.max(roi_img))
    total = np.sum(roi_img) + 1e-9
    yy, xx = np.mgrid[0:roi_img.shape[0], 0:roi_img.shape[1]]
    cx = np.sum(xx * roi_img) / total
    cy = np.sum(yy * roi_img) / total
    moment = np.sum(((xx - cx) ** 2 + (yy - cy) ** 2) * roi_img) / total
    return float(moment)


class CameraBase:
    def capture(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class CameraGX(CameraBase):
    def __init__(self, config: Dict):
        from importlib import import_module

        gx = import_module("gxipy")
        self.gx = gx
        self.device_manager = gx.DeviceManager()
        self.device_manager.update_device_list()
        self.image_convert = self.device_manager.create_image_format_convert()
        device_sn = config.get("device_sn") or ""
        if device_sn:
            self.cam = self.device_manager.open_device_by_sn(device_sn)
        else:
            if self.device_manager.device_num == 0:
                raise RuntimeError("未检测到相机设备")
            self.cam = self.device_manager.open_device_by_index(1)
        self._setup_camera(config)
        self.cam.stream_on()
        self.avg_frames = int(config.get("avg_frames", 1))

    def _setup_camera(self, config: Dict) -> None:
        remote = self.cam
        if remote.get_remote_device_feature_control().is_implemented("ExposureTime"):
            exposure = float(config.get("exposure_us", 20000))
            remote.get_remote_device_feature_control().get_float_feature("ExposureTime").set(exposure)
        if remote.get_remote_device_feature_control().is_implemented("Gain"):
            gain = float(config.get("gain", 0.0))
            remote.get_remote_device_feature_control().get_float_feature("Gain").set(gain)

    def capture(self) -> np.ndarray:
        from DaHeng.GetImage import capture

        frames = []
        for _ in range(self.avg_frames):
            frame = capture(self.cam, self.image_convert)
            if frame is None:
                raise RuntimeError("相机采集失败")
            frames.append(frame.astype(np.float32))
        avg = np.mean(frames, axis=0)
        return avg.astype(np.float32)

    def close(self) -> None:
        self.cam.stream_off()
        self.cam.close_device()


class MockCamera(CameraBase):
    def __init__(self, config: Dict, slm2_shape: Tuple[int, int], noise_std: float = 5.0):
        self.height, self.width = config.get("camera_size", [800, 800])
        self.noise_std = noise_std
        self.slm2_shape = slm2_shape
        self.current_window: Optional[Tuple[int, int]] = None

    def update_window(self, center: Optional[Tuple[int, int]]) -> None:
        self.current_window = center

    def capture(self) -> np.ndarray:
        img = np.zeros((self.height, self.width), dtype=np.float32)
        if self.current_window is not None:
            cx, cy = self.current_window
            sx = int(self.width * 0.5 + (cx - self.slm2_shape[1] / 2) * 0.1)
            sy = int(self.height * 0.5 + (cy - self.slm2_shape[0] / 2) * 0.1)
            yy, xx = np.mgrid[0:self.height, 0:self.width]
            img += 200.0 * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * 20 ** 2))
        noise = np.random.normal(0, self.noise_std, size=img.shape)
        return np.clip(img + noise, 0, 255).astype(np.float32)


class SLM1Controller:
    def __init__(self, config: Dict, output_dir: Path):
        from UPO_SLM_80Rplus.SLM_UPOLabs import SLM_UP

        self.slm = SLM_UP()
        self.screen_num = int(config.get("screen_num", 1))
        self.slm.Open_window(screenNum=self.screen_num)
        self.width, self.height = self.slm.Get_size(self.screen_num)
        self.comp_u8 = load_compensation_u8(config.get("compensation_path", ""), (self.height, self.width))
        self.output_dir = output_dir
        ensure_dir(str(self.output_dir))

    def display_gray(self, img_u8: np.ndarray) -> None:
        if self.comp_u8 is not None:
            img_u8 = ((img_u8.astype(np.uint16) + self.comp_u8.astype(np.uint16)) % 256).astype(np.uint8)
        img = Image.fromarray(img_u8)
        path = self.output_dir / "slm1_temp.bmp"
        img.save(path)
        self.slm.Disp_ReadImage(path=str(path), screenNum=self.screen_num, bits=8)

    def close(self) -> None:
        self.slm.Close_window(self.screen_num)


class SLM2Controller:
    def __init__(self, config: Dict, output_dir: Path):
        from importlib import import_module

        heds = import_module("HEDS")
        self.heds = heds
        self.output_dir = output_dir
        ensure_dir(str(self.output_dir))

        heds.SDK.PrintVersion()
        err = heds.SDK.Init(4, 1)
        if err != heds.HEDSERR_NoError:
            raise RuntimeError(heds.SDK.ErrorString(err))
        self.slm = heds.SLM.Init()
        if self.slm.errorCode() != heds.HEDSERR_NoError:
            raise RuntimeError(heds.SDK.ErrorString(self.slm.errorCode()))

        self.width = self.slm.width_px()
        self.height = self.slm.height_px()
        self.comp = load_compensation(config.get("compensation_path", ""), (self.height, self.width))

    def display_phase(self, phase: np.ndarray) -> None:
        phase = apply_compensation(phase, self.comp)
        img_u8 = phase_to_uint8(phase)
        img = Image.fromarray(img_u8)
        path = self.output_dir / "slm2_temp.bmp"
        img.save(path)
        if hasattr(self.slm, "showDataFromImageFile"):
            err = self.slm.showDataFromImageFile(str(path))
        elif hasattr(self.slm, "showCGHFromImageFile"):
            err = self.slm.showCGHFromImageFile(str(path))
        else:
            raise RuntimeError("未找到可用的 SLM2 显示接口")
        if err != self.heds.HEDSERR_NoError:
            raise RuntimeError(self.heds.SDK.ErrorString(err))


class MockSLM2:
    def __init__(self, config: Dict):
        self.height, self.width = config.get("slm2_size", [1200, 1920])
        self.last_phase = None

    def display_phase(self, phase: np.ndarray) -> None:
        self.last_phase = phase


def build_scan_positions(roi: ROI, step: int) -> Tuple[List[int], List[int]]:
    xs = list(range(roi.x_min, roi.x_max + 1, step))
    ys = list(range(roi.y_min, roi.y_max + 1, step))
    return xs, ys


def scan_layer(camera: CameraBase, slm2, layer_idx: int, centers: List[Tuple[int, int]],
               roi: ROI, config: Dict, slm2_shape: Tuple[int, int], block_mode: str,
               output_dir: Path, mock_cam: Optional[MockCamera]) -> Tuple[int, int]:
    win_cfg = config["window"]
    shape_type = win_cfg["shape"]
    size_px = int(win_cfg["size_px"])
    step = int(win_cfg["scan_step_px"])
    roi_cam = roi_from_dict(config["roi_cam"])
    metric_type = config["metric"]["type"]

    xs, ys = build_scan_positions(roi, step)
    heatmap = np.zeros((len(ys), len(xs)), dtype=np.float32)

    base_block = blocking_phase(slm2_shape, block_mode)

    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            phase = np.array(base_block, copy=True)
            for cx, cy in centers:
                mask = window_mask(slm2_shape, (cx, cy), size_px, shape_type)
                phase[mask] = 0.0
            mask = window_mask(slm2_shape, (x, y), size_px, shape_type)
            phase[mask] = 0.0
            slm2.display_phase(phase)
            if mock_cam is not None:
                mock_cam.update_window((x, y))
            img = camera.capture()
            heatmap[yi, xi] = compute_metric(img, roi_cam, metric_type)

    best_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    best_y = ys[best_idx[0]]
    best_x = xs[best_idx[1]]

    fine_cfg = win_cfg.get("fine_scan", {})
    if fine_cfg.get("enabled", False):
        radius = int(fine_cfg.get("radius_px", step))
        fine_step = int(fine_cfg.get("step_px", max(1, step // 2)))
        fine_roi = ROI(
            x_min=max(roi.x_min, best_x - radius),
            x_max=min(roi.x_max, best_x + radius),
            y_min=max(roi.y_min, best_y - radius),
            y_max=min(roi.y_max, best_y + radius),
        )
        xs_f, ys_f = build_scan_positions(fine_roi, fine_step)
        fine_map = np.zeros((len(ys_f), len(xs_f)), dtype=np.float32)
        for yi, y in enumerate(ys_f):
            for xi, x in enumerate(xs_f):
                phase = np.array(base_block, copy=True)
                for cx, cy in centers:
                    mask = window_mask(slm2_shape, (cx, cy), size_px, shape_type)
                    phase[mask] = 0.0
                mask = window_mask(slm2_shape, (x, y), size_px, shape_type)
                phase[mask] = 0.0
                slm2.display_phase(phase)
                if mock_cam is not None:
                    mock_cam.update_window((x, y))
                img = camera.capture()
                fine_map[yi, xi] = compute_metric(img, roi_cam, metric_type)
        best_idx = np.unravel_index(np.argmax(fine_map), fine_map.shape)
        best_y = ys_f[best_idx[0]]
        best_x = xs_f[best_idx[1]]
        heatmap = fine_map
        xs = xs_f
        ys = ys_f

    ensure_dir(str(output_dir))
    np.savetxt(output_dir / f"layer_{layer_idx:02d}_heatmap.csv", heatmap, delimiter=",")
    meta = {
        "xs": xs,
        "ys": ys,
        "best_x": best_x,
        "best_y": best_y,
    }
    (output_dir / f"layer_{layer_idx:02d}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return best_x, best_y


def measure_focus(camera: CameraBase, slm2, layer_idx: int, center: Tuple[int, int],
                  centers: List[Tuple[int, int]], config: Dict, slm2_shape: Tuple[int, int],
                  block_mode: str,
                  output_dir: Path, mock_cam: Optional[MockCamera]) -> Tuple[float, List[Tuple[float, float]]]:
    roi_focus = roi_from_dict(config["roi_focus"])
    metric_type = config["focus_metric"]["type"]
    f_list = [float(f) for f in config["f_list_m"]]
    pixel_pitch = (
        float(config["pixel_pitch_x_m"]),
        float(config["pixel_pitch_y_m"]),
    )
    win_cfg = config["window"]
    shape_type = win_cfg["shape"]
    size_px = int(win_cfg["size_px"])

    base_block = blocking_phase(slm2_shape, block_mode)
    curves = []

    best_f = f_list[0]
    best_metric = None

    for f in f_list:
        phase = np.array(base_block, copy=True)
        for cx, cy in centers:
            mask = window_mask(slm2_shape, (cx, cy), size_px, shape_type)
            phase[mask] = 0.0
        lens = lens_phase(slm2_shape, center, pixel_pitch, config["lambda_m"], f)
        mask = window_mask(slm2_shape, center, size_px, shape_type)
        phase[mask] = np.mod(lens[mask], 2 * np.pi)
        slm2.display_phase(phase)
        if mock_cam is not None:
            mock_cam.update_window(center)
        img = camera.capture()
        metric = focus_metric(img, roi_focus, metric_type)
        curves.append((f, metric))

        if metric_type == "second_moment":
            if best_metric is None or metric < best_metric:
                best_metric = metric
                best_f = f
        else:
            if best_metric is None or metric > best_metric:
                best_metric = metric
                best_f = f

    ensure_dir(str(output_dir))
    curve_path = output_dir / f"layer_{layer_idx:02d}_focus_curve.csv"
    with curve_path.open("w", encoding="utf-8") as f:
        f.write("f_m,metric\n")
        for f_val, metric in curves:
            f.write(f"{f_val},{metric}\n")

    return best_f, curves


def auto_roi(center: Tuple[int, int], half_size: int, slm2_shape: Tuple[int, int]) -> ROI:
    cx, cy = center
    width = slm2_shape[1]
    height = slm2_shape[0]
    return ROI(
        x_min=max(0, cx - half_size),
        x_max=min(width - 1, cx + half_size),
        y_min=max(0, cy - half_size),
        y_max=min(height - 1, cy + half_size),
    )


def run_calibration(config: Dict, mock: bool) -> None:
    output_cfg = config["output"]
    ensure_dir(output_cfg["scan_maps_dir"])
    ensure_dir(output_cfg["focus_curves_dir"])

    slm1_temp_dir = Path(output_cfg["scan_maps_dir"]).parent / "slm1_tmp"
    slm2_temp_dir = Path(output_cfg["scan_maps_dir"]).parent / "slm2_tmp"

    if mock:
        slm2 = MockSLM2(config["mock"])
        slm2_shape = (slm2.height, slm2.width)
        camera = MockCamera(config["mock"], slm2_shape, config["mock"].get("noise_std", 5.0))
        slm1 = None
    else:
        slm1 = SLM1Controller(config["slm1"], slm1_temp_dir)
        slm2 = SLM2Controller(config["slm2"], slm2_temp_dir)
        slm2_shape = (slm2.height, slm2.width)
        camera = CameraGX(config["camera"])

    try:
        if slm1 is not None:
            gauss_cfg = config["slm1"].get("gaussian", {})
            if gauss_cfg.get("enabled", True):
                gauss = gaussian_pattern((slm1.height, slm1.width), gauss_cfg.get("sigma_px", 60),
                                        gauss_cfg.get("peak", 255))
                slm1.display_gray(gauss)

        centers: List[Tuple[int, int]] = []
        n_layers = int(config["calibration"]["n_layers"])
        roi_layers = config["calibration"].get("roi_layers", [])
        auto_cfg = config["calibration"].get("roi_auto", {})
        block_mode = config["slm2"].get("block_mode", "checkerboard")
        use_auto = bool(auto_cfg.get("enabled", True))
        half_size = int(auto_cfg.get("half_size_px", 120))

        for idx in range(n_layers):
            layer_i = idx + 1
            if idx < len(roi_layers):
                roi = roi_from_dict(roi_layers[idx]).clamp(slm2_shape[1], slm2_shape[0])
            elif use_auto and centers:
                roi = auto_roi(centers[-1], half_size, slm2_shape)
            else:
                roi = ROI(0, slm2_shape[1] - 1, 0, slm2_shape[0] - 1)

            best_x, best_y = scan_layer(
                camera,
                slm2,
                layer_i,
                centers,
                roi,
                config["calibration"],
                slm2_shape,
                block_mode,
                Path(output_cfg["scan_maps_dir"]),
                camera if isinstance(camera, MockCamera) else None,
            )
            centers.append((best_x, best_y))

        layers_result = []
        for idx, center in enumerate(centers):
            layer_i = idx + 1
            best_f, _ = measure_focus(
                camera,
                slm2,
                layer_i,
                center,
                centers[:layer_i],
                config["calibration"],
                slm2_shape,
                block_mode,
                Path(output_cfg["focus_curves_dir"]),
                camera if isinstance(camera, MockCamera) else None,
            )
            layers_result.append({
                "i": layer_i,
                "x": int(center[0]),
                "y": int(center[1]),
                "Z_m": float(best_f),
            })

        result = {
            "lambda_m": float(config["calibration"]["lambda_m"]),
            "N_layers": n_layers,
            "layers": layers_result,
        }
        result_path = Path(output_cfg["result_path"])
        ensure_dir(str(result_path.parent))
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        if isinstance(camera, CameraGX):
            camera.close()
        if slm1 is not None:
            slm1.close()


def check_devices(config: Dict) -> None:
    output_cfg = config["output"]
    check_dir = Path(output_cfg["scan_maps_dir"]).parent / "device_check"
    ensure_dir(str(check_dir))

    slm1 = SLM1Controller(config["slm1"], check_dir)
    slm2 = SLM2Controller(config["slm2"], check_dir)
    camera = CameraGX(config["camera"])
    try:
        gauss_cfg = config["slm1"].get("gaussian", {})
        gauss = gaussian_pattern((slm1.height, slm1.width), gauss_cfg.get("sigma_px", 60),
                                gauss_cfg.get("peak", 255))
        slm1.display_gray(gauss)

        block_mode = config["slm2"].get("block_mode", "checkerboard")
        block = blocking_phase((slm2.height, slm2.width), block_mode)
        slm2.display_phase(block)

        img = camera.capture()
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(check_dir / "camera_check.png")
    finally:
        camera.close()
        slm1.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="原位标定程序")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--mock", action="store_true", help="使用模拟数据运行")
    parser.add_argument("--check-devices", action="store_true", help="单独验证设备控制")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.check_devices:
        check_devices(config)
        return

    if args.mock:
        run_calibration(config, mock=True)
    else:
        run_calibration(config, mock=False)


if __name__ == "__main__":
    main()

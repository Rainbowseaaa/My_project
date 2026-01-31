import time
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6 import QtCore
from PIL import Image

from functions.bolduc_hologram import bolduc_phase_encoding
from functions.utils import phase_to_uint8
from ui.phase_utils import (
    apply_compensation,
    generate_lg_field,
    load_compensation,
    load_field_file,
    load_hologram_file,
    load_mnist_sample,
    load_phase_file,
    resize_and_embed,
    render_letter_field,
)


class CameraWorker(QtCore.QObject):
    frame_ready = QtCore.pyqtSignal(np.ndarray, float)
    error = QtCore.pyqtSignal(str)

    def __init__(self, camera, target_fps: float = 30.0, parent=None):
        super().__init__(parent)
        self._camera = camera
        self._running = False
        self._target_fps = max(1.0, float(target_fps))

    @QtCore.pyqtSlot()
    def start(self) -> None:
        self._running = True
        frame_count = 0
        fps = 0.0
        last_fps_time = time.time()
        interval = 1.0 / self._target_fps

        while self._running:
            start = time.time()
            try:
                frame = self._camera.capture()
            except Exception as exc:
                self.error.emit(str(exc))
                break

            frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = frame_count / (now - last_fps_time)
                frame_count = 0
                last_fps_time = now

            self.frame_ready.emit(frame, fps)
            elapsed = time.time() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    @QtCore.pyqtSlot()
    def stop(self) -> None:
        self._running = False

    @QtCore.pyqtSlot(float)
    def set_exposure(self, exposure_us: float) -> None:
        if hasattr(self._camera, "set_exposure"):
            try:
                self._camera.set_exposure(float(exposure_us))
            except Exception as exc:
                self.error.emit(str(exc))


# [ui/workers.py]
# 重点修改 SLM1Worker，其他类如 CameraWorker 无需大改

class SLM1Worker(QtCore.QObject):
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    image_ready = QtCore.pyqtSignal(object)

    def __init__(self, controller, slm_shape, period: int, parent=None):
        super().__init__(parent)
        self._slm = controller
        self._slm_shape = slm_shape
        self._period = period

    # 参数签名增加 flip_h, flip_v
    @QtCore.pyqtSlot(str, bool, str, str, object, int, float, bool, bool)
    def load_hologram(self, image_path: str, use_comp: bool, comp_path: str, input_type: str, field_params: dict,
                      period: int, amp_norm_max: float, flip_h: bool, flip_v: bool) -> None:
        try:
            period = int(period) if period else int(self._period)
            amp_norm_max = float(amp_norm_max)
            if amp_norm_max < 0.0:
                amp_norm_max = 0.0
            if amp_norm_max > 0.99:
                amp_norm_max = 0.99
            # 1. 生成或加载基础数据
            if input_type == "hologram":
                hologram_u8 = load_hologram_file(image_path, self._slm_shape)
                # 如果是直接全息图，直接翻转 uint8
                if flip_h: hologram_u8 = np.fliplr(hologram_u8)
                if flip_v: hologram_u8 = np.flipud(hologram_u8)
            else:
                # 这是一个生成光场的过程
                if field_params.get("mode") == "lg":
                    amp, phase = generate_lg_field(
                        self._slm_shape,
                        float(field_params.get("w0", 80.0)),
                        int(field_params.get("p", 0)),
                        int(field_params.get("l", 0)),
                    )
                elif field_params.get("mode") == "letter":
                    size = field_params.get("size")
                    if size and size[0] > 0 and size[1] > 0:
                        amp, phase = render_letter_field((int(size[1]), int(size[0])), field_params.get("letter", "A"))
                    else:
                        amp, phase = render_letter_field(self._slm_shape, field_params.get("letter", "A"))
                    amp, phase = resize_and_embed(amp, phase, self._slm_shape, size)
                elif field_params.get("mode") in ("mnist", "fashion_mnist"):
                    is_fashion = (field_params.get("mode") == "fashion_mnist")
                    sample = load_mnist_sample(
                        int(field_params.get("index", 0)),
                        fashion=is_fashion,
                        data_dir=field_params.get("data_dir"),
                    )
                    amp = np.asarray(sample, dtype=np.float64) / 255.0
                    phase = np.zeros_like(amp)
                    size = field_params.get("size")
                    amp, phase = resize_and_embed(amp, phase, self._slm_shape, size)
                else:
                    amp, phase = load_field_file(image_path, self._slm_shape)

                amp_max = float(np.max(amp)) if amp.size else 0.0
                if amp_max > 0:
                    amp = amp / amp_max * amp_norm_max
                else:
                    amp = np.zeros_like(amp)

                # 光场层面的翻转
                if flip_h:
                    amp = np.fliplr(amp)
                    phase = np.fliplr(phase)
                if flip_v:
                    amp = np.flipud(amp)
                    phase = np.flipud(phase)

                hologram_phase = bolduc_phase_encoding(amp, phase, period)
                hologram_u8 = phase_to_uint8(hologram_phase)

            if use_comp and comp_path:
                comp = load_compensation(comp_path, hologram_u8.shape, meaning="encoded_0_255")
                if comp is not None:
                    comp_u8 = phase_to_uint8(comp)
                    hologram_u8 = ((hologram_u8.astype(np.uint16) + comp_u8.astype(np.uint16)) % 256).astype(np.uint8)

            self._slm.display_gray(hologram_u8, use_comp=False)
            self.image_ready.emit(hologram_u8)
            name = Path(image_path).name if image_path else "generated"
            self.status.emit(f"SLM1 加载完成: {name}")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.error.emit(str(exc))


class SLM2Worker(QtCore.QObject):
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    phase_ready = QtCore.pyqtSignal(object, object)

    def __init__(self, controller, slm_shape, parent=None):
        super().__init__(parent)
        self._slm = controller
        self._slm_shape = slm_shape

    @QtCore.pyqtSlot(object, bool, str)
    def load_phase(self, phase: np.ndarray, use_comp: bool, comp_path: str) -> None:
        try:
            if use_comp and comp_path:
                comp = load_compensation(comp_path, phase.shape, meaning="encoded_0_255")
                phase = apply_compensation(phase, comp)
            self._slm.display_phase(phase, use_comp=False)
            returned = getattr(self._slm, "last_phase", None)
            self.phase_ready.emit(phase, returned)
            self.status.emit("SLM2 合成相位已加载")
        except Exception as exc:
            self.error.emit(str(exc))


class PhaseLoader:
    def __init__(self, mat_key: str = "phase"):
        self.mat_key = mat_key

    def load(self, path: str, meaning: str = "auto") -> np.ndarray:
        return load_phase_file(path, meaning=meaning, mat_key=self.mat_key)

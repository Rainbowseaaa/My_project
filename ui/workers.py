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
    load_compensation,
    load_phase_file,
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


class SLM1Worker(QtCore.QObject):
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(self, controller, slm_shape, period: int, parent=None):
        super().__init__(parent)
        self._slm = controller
        self._slm_shape = slm_shape
        self._period = period

    @QtCore.pyqtSlot(str, bool, str)
    def load_hologram(self, image_path: str, use_comp: bool, comp_path: str) -> None:
        try:
            img = Image.open(image_path).convert("L")
            img = img.resize((self._slm_shape[1], self._slm_shape[0]), Image.BILINEAR)
            amp = np.asarray(img, dtype=np.float64) / 255.0
            phase = np.zeros_like(amp)
            hologram_phase = bolduc_phase_encoding(amp, phase, self._period)

            if use_comp and comp_path:
                comp = load_compensation(comp_path, hologram_phase.shape, meaning="encoded_0_255")
                hologram_phase = apply_compensation(hologram_phase, comp)

            hologram_u8 = phase_to_uint8(hologram_phase)
            self._slm.display_gray(hologram_u8, use_comp=False)
            self.status.emit(f"SLM1 加载完成: {Path(image_path).name}")
        except Exception as exc:
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

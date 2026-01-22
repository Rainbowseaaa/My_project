import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PIL import Image
from PyQt6 import QtCore, QtGui, QtWidgets

from calibrate_insitu import CameraGX, MockCamera, MockSLM2, SLM1Controller, SLM2Controller
from ui.phase_utils import compose_layers, load_phase_file, make_preview_image
from ui.widgets import (
    ImageSourcePanel,
    LogPanel,
    PlayerControls,
    PreviewPanel,
    RoiStatsPanel,
    SLM2Panel,
    StatusPanel,
)
from ui.workers import CameraWorker, SLM1Worker, SLM2Worker


def load_config(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class MockSLM1:
    def __init__(self, size):
        self.height, self.width = size
        self.last_image = None

    def display_gray(self, img_u8: np.ndarray, use_comp: bool = True) -> None:
        self.last_image = img_u8


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config: dict, mock: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("SLM 控制 UI")
        self.config = config
        self.mock = mock

        self.slm1_controller = None
        self.slm2_controller = None
        self.camera = None

        self.image_paths: List[Path] = []
        self.image_index = 0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_timer_next)

        self.recording = False
        self.record_dir: Optional[Path] = None

        self._setup_ui()
        self._setup_threads()
        self._load_defaults()

    def _setup_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        self.image_panel = ImageSourcePanel()
        self.player_controls = PlayerControls()
        self.slm2_panel = SLM2Panel()
        self.preview_panel = PreviewPanel()
        self.status_panel = StatusPanel()
        self.log_panel = LogPanel()

        self.run_button = QtWidgets.QPushButton("Run")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.save_button = QtWidgets.QPushButton("保存当前帧")
        self.record_button = QtWidgets.QPushButton("开始连续保存")
        self.stop_record_button = QtWidgets.QPushButton("停止连续保存")

        self.roi_panel = RoiStatsPanel()

        self.image_panel.generate_button.clicked.connect(self.generate_hologram)
        self.image_panel.load_button.clicked.connect(self.load_slm1)
        self.player_controls.play_clicked.connect(self.start_playback)
        self.player_controls.pause_clicked.connect(self.pause_playback)
        self.player_controls.prev_clicked.connect(self.show_prev_image)
        self.player_controls.next_clicked.connect(self.show_next_image)

        for widget in self.slm2_panel.layer_widgets:
            widget.file_changed.connect(self.on_layer_change)
            widget.enabled_changed.connect(self.on_layer_change)
            widget.offset_changed.connect(self.on_layer_change)

        self.slm2_panel.apply_button.clicked.connect(self.apply_slm2)

        self.run_button.clicked.connect(self.run_all)
        self.stop_button.clicked.connect(self.stop_all)
        self.save_button.clicked.connect(self.save_frame)
        self.record_button.clicked.connect(self.start_recording)
        self.stop_record_button.clicked.connect(self.stop_recording)

        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(self.image_panel)
        left_layout.addWidget(self.player_controls)
        left_layout.addWidget(self.log_panel)

        center_layout = QtWidgets.QVBoxLayout()
        center_layout.addWidget(self.slm2_panel)
        center_layout.addWidget(self.preview_panel)

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(self._build_camera_view())
        right_layout.addWidget(self.roi_panel)
        right_layout.addWidget(self.save_button)
        right_layout.addWidget(self.record_button)
        right_layout.addWidget(self.stop_record_button)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(center_layout, 1)
        main_layout.addLayout(right_layout, 1)

        bottom_layout = QtWidgets.QHBoxLayout()
        bottom_layout.addWidget(self.run_button)
        bottom_layout.addWidget(self.stop_button)
        bottom_layout.addWidget(self.status_panel)

        wrapper = QtWidgets.QVBoxLayout(central)
        wrapper.addLayout(main_layout)
        wrapper.addLayout(bottom_layout)

    def _build_camera_view(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("相机显示")
        layout = QtWidgets.QVBoxLayout(group)

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.view_box = self.plot_widget.addViewBox(lockAspect=True)
        pan_mode = getattr(getattr(pg.ViewBox, "MouseMode", None), "PanMode", None)
        if pan_mode is None:
            pan_mode = getattr(pg.ViewBox, "PanMode", None)
        if pan_mode is not None:
            self.view_box.setMouseMode(pan_mode)
        self.image_item = pg.ImageItem()
        self.view_box.addItem(self.image_item)
        self.view_box.setBackgroundColor("k")

        self.roi = pg.RectROI([50, 50], [100, 100], pen=pg.mkPen("y", width=2))
        self.view_box.addItem(self.roi)
        self.roi.sigRegionChanged.connect(self.update_roi_stats)

        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_moved)

        layout.addWidget(self.plot_widget)
        return group

    def _setup_threads(self) -> None:
        self.camera_thread = QtCore.QThread(self)
        self.slm1_thread = QtCore.QThread(self)
        self.slm2_thread = QtCore.QThread(self)

        self.camera_worker = None
        self.slm1_worker = None
        self.slm2_worker = None

    def _load_defaults(self) -> None:
        slm1_comp = self.config.get("slm1", {}).get("compensation_path", "")
        slm2_comp = self.config.get("slm2", {}).get("compensation_path", "")
        if slm1_comp:
            self.image_panel.slm1_comp_edit.setText(slm1_comp)
        if slm2_comp:
            self.slm2_panel.slm2_comp_edit.setText(slm2_comp)

        interval_ms = int(self.config.get("slm1", {}).get("play_interval_ms", 500))
        self.image_panel.interval_spin.setValue(interval_ms)

        for idx, widget in enumerate(self.slm2_panel.layer_widgets):
            layer_cfg = self.config.get("slm2", {}).get("layers", [])
            if idx < len(layer_cfg):
                widget.file_edit.setText(layer_cfg[idx].get("path", ""))
                widget.dx_spin.setValue(int(layer_cfg[idx].get("dx", 0)))
                widget.dy_spin.setValue(int(layer_cfg[idx].get("dy", 0)))
                widget.enable_checkbox.setChecked(bool(layer_cfg[idx].get("enabled", True)))

    def log(self, message: str) -> None:
        self.log_panel.append(message)
        print(message)

    def _init_hardware(self) -> None:
        output_cfg = self.config.get("output", {})
        slm1_dir = Path(output_cfg.get("slm1_tmp_dir", "output/slm1_tmp"))
        slm2_dir = Path(output_cfg.get("slm2_tmp_dir", "output/slm2_tmp"))
        slm1_dir.mkdir(parents=True, exist_ok=True)
        slm2_dir.mkdir(parents=True, exist_ok=True)

        if self.mock:
            slm1_size = tuple(self.config.get("mock", {}).get("slm1_size", [1200, 1920]))
            slm2_size = tuple(self.config.get("mock", {}).get("slm2_size", [1200, 1920]))
            self.slm1_controller = MockSLM1(slm1_size)
            self.slm2_controller = MockSLM2({"slm2_size": slm2_size})
            self.camera = MockCamera(self.config.get("mock", {}), slm2_size)
        else:
            self.slm1_controller = SLM1Controller(self.config.get("slm1", {}), slm1_dir)
            self.slm2_controller = SLM2Controller(self.config.get("slm2", {}), slm2_dir)
            self.camera = CameraGX(self.config.get("camera", {}))

    def _start_workers(self) -> None:
        self._init_hardware()

        slm1_period = int(self.config.get("slm1", {}).get("bolduc_period", 8))
        slm1_shape = (self.slm1_controller.height, self.slm1_controller.width)
        slm2_shape = (self.slm2_controller.height, self.slm2_controller.width)

        self.camera_worker = CameraWorker(self.camera, self.config.get("camera", {}).get("target_fps", 30))
        self.slm1_worker = SLM1Worker(self.slm1_controller, slm1_shape, slm1_period)
        self.slm2_worker = SLM2Worker(self.slm2_controller, slm2_shape)

        self.camera_worker.moveToThread(self.camera_thread)
        self.slm1_worker.moveToThread(self.slm1_thread)
        self.slm2_worker.moveToThread(self.slm2_thread)

        self.camera_thread.started.connect(self.camera_worker.start)
        self.camera_worker.frame_ready.connect(self.update_frame)
        self.camera_worker.error.connect(self.on_camera_error)

        self.slm1_worker.status.connect(self.on_slm1_status)
        self.slm1_worker.error.connect(self.on_slm1_error)
        self.slm2_worker.status.connect(self.on_slm2_status)
        self.slm2_worker.error.connect(self.on_slm2_error)

        self.camera_thread.start()
        self.slm1_thread.start()
        self.slm2_thread.start()

    def run_all(self) -> None:
        if self.camera_thread.isRunning():
            return
        self.log("启动运行流程")
        self._start_workers()
        self.start_playback()

    def stop_all(self) -> None:
        self.log("停止运行流程")
        self.pause_playback()

        if self.camera_worker is not None:
            self.camera_worker.stop()
        if self.camera_thread.isRunning():
            self.camera_thread.quit()
            self.camera_thread.wait()
        if self.slm1_thread.isRunning():
            self.slm1_thread.quit()
            self.slm1_thread.wait()
        if self.slm2_thread.isRunning():
            self.slm2_thread.quit()
            self.slm2_thread.wait()

        if self.camera is not None and hasattr(self.camera, "close"):
            self.camera.close()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.stop_all()
        event.accept()

    def _collect_image_paths(self) -> None:
        folder = self.image_panel.folder_edit.text().strip()
        image = self.image_panel.image_edit.text().strip()
        paths = []
        if folder:
            folder_path = Path(folder)
            for ext in ("*.png", "*.bmp", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
                paths.extend(sorted(folder_path.glob(ext)))
        elif image:
            paths.append(Path(image))
        self.image_paths = [p for p in paths if p.exists()]
        self.image_index = 0

    def start_playback(self) -> None:
        self._collect_image_paths()
        if not self.image_paths:
            self.log("未找到输入图像")
            return
        interval = self.image_panel.interval_spin.value()
        self.timer.start(interval)
        self.show_current_image()

    def pause_playback(self) -> None:
        self.timer.stop()

    def show_current_image(self) -> None:
        if not self.image_paths:
            return
        path = self.image_paths[self.image_index]
        self.status_panel.update_status(image_name=path.name)
        self.generate_hologram()

    def show_prev_image(self) -> None:
        if not self.image_paths:
            return
        self.image_index = (self.image_index - 1) % len(self.image_paths)
        self.show_current_image()

    def show_next_image(self) -> None:
        if not self.image_paths:
            return
        self.image_index = (self.image_index + 1) % len(self.image_paths)
        self.show_current_image()

    def _on_timer_next(self) -> None:
        if not self.image_paths:
            return
        if self.image_index >= len(self.image_paths) - 1:
            if self.player_controls.loop_enabled():
                self.image_index = 0
            else:
                self.timer.stop()
                return
        else:
            self.image_index += 1
        self.show_current_image()

    def generate_hologram(self) -> None:
        if not self.image_paths:
            self._collect_image_paths()
        if not self.image_paths:
            self.log("未选择输入图像")
            return
        if self.slm1_worker is None:
            self.log("SLM1 未初始化，请先 Run")
            return
        self.load_slm1()

    def load_slm1(self) -> None:
        if not self.image_paths:
            return
        path = str(self.image_paths[self.image_index])
        use_comp = self.image_panel.slm1_comp_checkbox.isChecked()
        comp_path = self.image_panel.slm1_comp_edit.text().strip()
        QtCore.QMetaObject.invokeMethod(
            self.slm1_worker,
            "load_hologram",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(str, path),
            QtCore.Q_ARG(bool, use_comp),
            QtCore.Q_ARG(str, comp_path),
        )

    def on_layer_change(self, _: int) -> None:
        if self.slm2_panel.auto_apply_checkbox.isChecked():
            self.apply_slm2()
        else:
            self.update_preview()

    def apply_slm2(self) -> None:
        try:
            phase = self.build_slm2_phase()
        except Exception as exc:
            self.log(f"SLM2 合成失败: {exc}")
            return

        if self.slm2_worker is None:
            self.log("SLM2 未初始化，请先 Run")
            return

        use_comp = self.slm2_panel.slm2_comp_checkbox.isChecked()
        comp_path = self.slm2_panel.slm2_comp_edit.text().strip()

        QtCore.QMetaObject.invokeMethod(
            self.slm2_worker,
            "load_phase",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(object, phase),
            QtCore.Q_ARG(bool, use_comp),
            QtCore.Q_ARG(str, comp_path),
        )
        self.update_preview()

    def build_slm2_phase(self) -> np.ndarray:
        if self.slm2_controller is not None:
            slm2_shape = (self.slm2_controller.height, self.slm2_controller.width)
        else:
            slm2_shape = tuple(self.config.get("mock", {}).get("slm2_size", [1200, 1920]))
        window_size = tuple(self.config.get("slm2", {}).get("window_size_px", [400, 400]))
        center_defaults = self.config.get("slm2", {}).get("default_centers", [])

        layers = []
        centers = []
        enabled = []

        for idx, layer in enumerate(self.slm2_panel.get_layers()):
            if idx < len(center_defaults):
                base_center = center_defaults[idx]
            else:
                base_center = [slm2_shape[1] // 2, slm2_shape[0] // 2]
            cx = int(base_center[0] + layer.dx)
            cy = int(base_center[1] + layer.dy)

            phase = np.zeros(window_size, dtype=np.float64)
            if layer.file_path:
                phase = load_phase_file(layer.file_path, meaning="auto", mat_key=self.config.get("slm2", {}).get("mat_key", "phase"))

            layers.append({"phase": phase, "center": (cx, cy), "enabled": layer.enabled})
            centers.append((cx, cy))
            enabled.append(layer.enabled)

        outside_mode = self.config.get("slm2", {}).get("outside_mode", "zero")
        block_mode = self.config.get("slm2", {}).get("block_mode", "checkerboard")

        phase, rects = compose_layers(slm2_shape, window_size, layers, outside_mode, block_mode)
        self._latest_preview = (rects, centers, enabled)
        return phase

    def update_preview(self) -> None:
        if not hasattr(self, "_latest_preview"):
            try:
                _ = self.build_slm2_phase()
            except Exception:
                return
        rects, centers, enabled = self._latest_preview
        if self.slm2_controller is not None:
            slm2_shape = (self.slm2_controller.height, self.slm2_controller.width)
        else:
            slm2_shape = tuple(self.config.get("mock", {}).get("slm2_size", [1200, 1920]))
        img = make_preview_image(slm2_shape, rects, centers, enabled)
        qimage = QtGui.QImage(
            img.tobytes("raw", "RGB"),
            img.width,
            img.height,
            QtGui.QImage.Format.Format_RGB888,
        )
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.preview_panel.update_pixmap(pixmap)

    def update_frame(self, frame: np.ndarray, fps: float) -> None:
        self.image_item.setImage(frame, autoLevels=True)
        self.status_panel.update_status(fps=fps)
        if self.recording and self.record_dir:
            timestamp = QtCore.QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss_zzz")
            path = self.record_dir / f"frame_{timestamp}.png"
            img = np.clip(frame, 0, 255).astype(np.uint8)
            Image.fromarray(img).save(path)
        self.update_roi_stats()

    def update_roi_stats(self) -> None:
        if self.image_item.image is None:
            return
        img = self.image_item.image
        roi_bounds = self.roi.parentBounds()
        x0 = max(int(roi_bounds.left()), 0)
        y0 = max(int(roi_bounds.top()), 0)
        x1 = min(int(roi_bounds.right()), img.shape[1] - 1)
        y1 = min(int(roi_bounds.bottom()), img.shape[0] - 1)
        if x1 <= x0 or y1 <= y0:
            return
        roi = img[y0:y1, x0:x1]
        if roi.size == 0:
            return
        mean = float(np.mean(roi))
        min_val = float(np.min(roi))
        max_val = float(np.max(roi))
        sum_val = float(np.sum(roi))

        yy, xx = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
        total = np.sum(roi) + 1e-9
        cx = float(np.sum(xx * roi) / total) + x0
        cy = float(np.sum(yy * roi) / total) + y0

        self.roi_panel.update_stats(mean, min_val, max_val, sum_val, (cx, cy))

    def on_mouse_moved(self, pos) -> None:
        if self.image_item.image is None:
            return
        point = self.view_box.mapSceneToView(pos)
        x, y = int(point.x()), int(point.y())
        img = self.image_item.image
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            value = img[y, x]
            self.roi_panel.update_pixel(f"Pixel: ({x}, {y}) = {value:.1f}")
        else:
            self.roi_panel.update_pixel("Pixel: -")

    def save_frame(self) -> None:
        if self.image_item.image is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "保存当前帧",
            "frame.png",
            "PNG (*.png)",
        )
        if path:
            img = np.clip(self.image_item.image, 0, 255).astype(np.uint8)
            Image.fromarray(img).save(path)
            self.log(f"保存当前帧: {path}")

    def start_recording(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not folder:
            return
        self.record_dir = Path(folder)
        self.recording = True
        self.log(f"开始连续保存: {folder}")

    def stop_recording(self) -> None:
        self.recording = False
        self.log("停止连续保存")

    def on_slm1_status(self, message: str) -> None:
        self.status_panel.update_status(slm1=message)
        self.log(message)

    def on_slm1_error(self, message: str) -> None:
        self.status_panel.update_status(slm1=f"错误: {message}")
        self.log(f"SLM1 错误: {message}")

    def on_slm2_status(self, message: str) -> None:
        self.status_panel.update_status(slm2=message)
        self.log(message)

    def on_slm2_error(self, message: str) -> None:
        self.status_panel.update_status(slm2=f"错误: {message}")
        self.log(f"SLM2 错误: {message}")

    def on_camera_error(self, message: str) -> None:
        self.log(f"相机错误: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SLM UI 控制程序")
    parser.add_argument("--config", default="config_ui.yaml", help="配置文件路径")
    parser.add_argument("--mock", action="store_true", help="使用 Mock 模式")
    args = parser.parse_args()

    config = load_config(args.config)
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(config, mock=args.mock)
    window.resize(1600, 900)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

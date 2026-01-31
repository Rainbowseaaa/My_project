import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PIL import Image
from PyQt6 import QtCore, QtGui, QtWidgets

from calibrate_insitu import CameraGX, MockCamera, MockSLM2, SLM1Controller, SLM2Controller
from ui.phase_utils import apply_compensation, compose_layers, load_compensation, load_phase_file, make_preview_image
from ui.widgets import (
    CameraControlPanel,
    ImageSourcePanel,
    LogPanel,
    PreviewPanel,
    RoiStatsPanel,
    SLMPreviewPanel,
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


class CameraViewBox(pg.ViewBox):
    select_drag = QtCore.pyqtSignal(QtCore.QPointF, QtCore.QPointF, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = "pan"
        self._drag_start = None

    def set_mode(self, mode: str) -> None:
        self._mode = mode

    def mouseDragEvent(self, ev, axis=None):
        if self._mode == "select" and ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            if ev.isStart():
                self._drag_start = ev.buttonDownScenePos()
            if self._drag_start is None:
                self._drag_start = ev.buttonDownScenePos()
            end = ev.scenePos()
            self.select_drag.emit(self._drag_start, end, ev.isFinish())
            if ev.isFinish():
                self._drag_start = None
            return
        super().mouseDragEvent(ev, axis=axis)


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
        self.last_record_time = 0.0
        self.overexposed = False
        self._latest_slm2_loaded_phase: Optional[np.ndarray] = None
        self._latest_slm2_returned_phase: Optional[np.ndarray] = None
        self._latest_slm1_image: Optional[np.ndarray] = None

        self._setup_ui()
        self._setup_threads()
        self._load_defaults()

    def _setup_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        self.image_panel = ImageSourcePanel()
        self.slm2_panel = SLM2Panel()
        self.preview_panel = PreviewPanel()
        self.slm1_preview = SLMPreviewPanel("SLM1 预览")
        self.status_panel = StatusPanel()
        self.log_panel = LogPanel()
        self.camera_control = CameraControlPanel()

        self.run_button = QtWidgets.QPushButton("Run")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.save_button = QtWidgets.QPushButton("保存当前帧")
        self.record_button = QtWidgets.QPushButton("开始连续保存")
        self.stop_record_button = QtWidgets.QPushButton("停止连续保存")

        self.roi_panel = RoiStatsPanel()

        self.image_panel.single_button.clicked.connect(self.show_single_frame)
        self.image_panel.play_button.clicked.connect(self.start_playback)
        self.image_panel.stop_play_button.clicked.connect(self.pause_playback)
        self.image_panel.prev_button.clicked.connect(self.show_prev_image)
        self.image_panel.next_button.clicked.connect(self.show_next_image)
        self.image_panel.run_button.clicked.connect(self.start_slm1)
        self.image_panel.stop_button.clicked.connect(self.stop_slm1)
        self.image_panel.auto_apply_checkbox.stateChanged.connect(self.on_slm1_auto_apply_change)
        self.image_panel.slm1_comp_checkbox.stateChanged.connect(self.on_slm1_auto_apply_change)
        self.image_panel.slm1_comp_edit.textChanged.connect(self.on_slm1_auto_apply_change)

        for widget in self.slm2_panel.layer_widgets:
            widget.file_changed.connect(self.on_layer_change)
            widget.enabled_changed.connect(self.on_layer_change)
            widget.offset_changed.connect(self.on_layer_change)

        self.slm2_panel.apply_button.clicked.connect(self.apply_slm2)
        self.preview_panel.display_mode_changed.connect(lambda _: self.update_preview())
        self.slm2_panel.run_button.clicked.connect(self.start_slm2)
        self.slm2_panel.stop_button.clicked.connect(self.stop_slm2)
        self.slm2_panel.slm2_comp_checkbox.stateChanged.connect(self.on_layer_change)
        self.slm2_panel.slm2_comp_edit.textChanged.connect(self.on_layer_change)

        self.run_button.clicked.connect(self.run_all)
        self.stop_button.clicked.connect(self.stop_all)
        self.save_button.clicked.connect(self.save_frame)
        self.record_button.clicked.connect(self.start_recording)
        self.stop_record_button.clicked.connect(self.stop_recording)
        self.camera_control.run_button.clicked.connect(self.start_camera)
        self.camera_control.stop_button.clicked.connect(self.stop_camera)
        self.camera_control.apply_exposure_button.clicked.connect(self.apply_exposure)

        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(self.image_panel)
        left_layout.addWidget(self._build_left_split())

        center_layout = QtWidgets.QVBoxLayout()
        center_layout.addWidget(self._build_center_split())

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(self._build_camera_view())
        right_layout.addWidget(self.roi_panel)
        right_layout.addWidget(self.camera_control)
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
        self.camera_toolbar = QtWidgets.QHBoxLayout()
        self.camera_toolbar.setContentsMargins(0, 0, 0, 0)
        self.camera_mode_button = QtWidgets.QToolButton()
        self.camera_mode_button.setCheckable(True)
        self.camera_mode_button.setToolTip("拖动/框选切换")
        self.camera_mode_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowForward))
        self.camera_mode_button.clicked.connect(self.toggle_camera_mode)
        self.camera_zoom_button = QtWidgets.QToolButton()
        self.camera_zoom_button.setToolTip("缩放到 ROI")
        self.camera_zoom_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogYesButton))
        self.camera_zoom_button.clicked.connect(self.zoom_to_roi)
        self.camera_reset_button = QtWidgets.QToolButton()
        self.camera_reset_button.setToolTip("视图复位")
        self.camera_reset_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload))
        self.camera_reset_button.clicked.connect(self.reset_camera_view)
        self.camera_toolbar.addStretch(1)
        self.camera_toolbar.addWidget(self.camera_mode_button)
        self.camera_toolbar.addWidget(self.camera_zoom_button)
        self.camera_toolbar.addWidget(self.camera_reset_button)
        layout.addLayout(self.camera_toolbar)
        self.view_box = CameraViewBox()
        self.plot_widget.addItem(self.view_box)
        self._pan_mode = getattr(getattr(pg.ViewBox, "MouseMode", None), "PanMode", None)
        if self._pan_mode is None:
            self._pan_mode = getattr(pg.ViewBox, "PanMode", None)
        self._rect_mode = getattr(getattr(pg.ViewBox, "MouseMode", None), "RectMode", None)
        if self._rect_mode is None:
            self._rect_mode = getattr(pg.ViewBox, "RectMode", None)
        if self._pan_mode is not None:
            self.view_box.setMouseMode(self._pan_mode)
        self.view_box.set_mode("pan")
        self.image_item = pg.ImageItem()
        self.view_box.addItem(self.image_item)
        self.view_box.setBackgroundColor("k")

        self.roi = pg.RectROI([50, 50], [100, 100], pen=pg.mkPen("y", width=2))
        self.roi.addScaleHandle([0, 0], [1, 1])
        self.roi.addScaleHandle([1, 0], [0, 1])
        self.roi.addScaleHandle([0, 1], [1, 0])
        self.roi.addScaleHandle([1, 1], [0, 0])
        self.roi.addTranslateHandle([0.5, 0.5])
        self.roi.setSelectable(True)
        self.roi.setVisible(False)
        self.view_box.addItem(self.roi)
        self.roi.sigRegionChanged.connect(self.update_roi_stats)
        self.roi.sigRegionChangeFinished.connect(self.on_roi_changed)
        self.view_box.select_drag.connect(self.on_camera_select_drag)

        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_moved)

        layout.addWidget(self.plot_widget)
        return group

    def _build_left_split(self) -> QtWidgets.QWidget:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(self.slm1_preview)
        splitter.addWidget(self.log_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        return splitter

    def _build_center_split(self) -> QtWidgets.QWidget:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(self.slm2_panel)
        splitter.addWidget(self.preview_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        return splitter

    def _setup_threads(self) -> None:
        self.camera_thread = QtCore.QThread(self)
        self.slm1_thread = QtCore.QThread(self)
        self.slm2_thread = QtCore.QThread(self)

        self.camera_worker = None
        self.slm1_worker = None
        self.slm2_worker = None

    def start_slm1(self) -> None:
        if self.slm1_thread.isRunning():
            return
        try:
            self.slm1_controller = self._create_slm_controller("slm1")
            slm1_shape = (self.slm1_controller.height, self.slm1_controller.width)
            slm1_period = int(self.config.get("slm1", {}).get("bolduc_period", 8))
            self.slm1_worker = SLM1Worker(self.slm1_controller, slm1_shape, slm1_period)
            self.slm1_worker.moveToThread(self.slm1_thread)
            self.slm1_worker.status.connect(self.on_slm1_status)
            self.slm1_worker.error.connect(self.on_slm1_error)
            self.slm1_worker.image_ready.connect(self.on_slm1_image_ready)
            self.slm1_thread.start()
            self.log("SLM1 已启动")
        except Exception as exc:
            self.slm1_controller = None
            self.slm1_worker = None
            self.log(f"SLM1 初始化失败: {exc}")

    def stop_slm1(self) -> None:
        if self.slm1_thread.isRunning():
            self.slm1_thread.quit()
            self.slm1_thread.wait()
        if self.slm1_controller is not None and hasattr(self.slm1_controller, "close"):
            self.slm1_controller.close()
        self.slm1_controller = None
        self.slm1_worker = None
        self.log("SLM1 已停止")

    def start_slm2(self) -> None:
        if self.slm2_thread.isRunning():
            return
        try:
            self.slm2_controller = self._create_slm_controller("slm2")
            slm2_shape = (self.slm2_controller.height, self.slm2_controller.width)
            self.slm2_worker = SLM2Worker(self.slm2_controller, slm2_shape)
            self.slm2_worker.moveToThread(self.slm2_thread)
            self.slm2_worker.status.connect(self.on_slm2_status)
            self.slm2_worker.error.connect(self.on_slm2_error)
            self.slm2_worker.phase_ready.connect(self.on_slm2_phase_ready)
            self.slm2_thread.start()
            self.log("SLM2 已启动")
        except Exception as exc:
            self.slm2_controller = None
            self.slm2_worker = None
            self.log(f"SLM2 初始化失败: {exc}")

    def stop_slm2(self) -> None:
        if self.slm2_thread.isRunning():
            self.slm2_thread.quit()
            self.slm2_thread.wait()
        if self.slm2_controller is not None and hasattr(self.slm2_controller, "close"):
            self.slm2_controller.close()
        self.slm2_controller = None
        self.slm2_worker = None
        self.log("SLM2 已停止")

    def start_camera(self) -> None:
        if self.camera_thread.isRunning():
            return
        try:
            if self.mock:
                slm2_type = self.slm2_panel.device_combo.currentData() or self.config.get("slm2", {}).get("device_type", "holoeye")
                slm2_shape = self._slm_shape_for_type(slm2_type)
                self.camera = MockCamera(self.config.get("mock", {}), slm2_shape)
            else:
                self.camera = CameraGX(self.config.get("camera", {}))
            self.camera_worker = CameraWorker(self.camera, self.config.get("camera", {}).get("target_fps", 30))
            self.camera_worker.moveToThread(self.camera_thread)
            self.camera_thread.started.connect(self.camera_worker.start)
            self.camera_worker.frame_ready.connect(self.update_frame)
            self.camera_worker.error.connect(self.on_camera_error)
            self.camera_thread.start()
            self.apply_exposure()
            self.log("相机已启动")
        except Exception as exc:
            self.camera = None
            self.camera_worker = None
            self.log(f"相机初始化失败: {exc}")

    def stop_camera(self) -> None:
        if self.camera_worker is not None:
            self.camera_worker.stop()
        if self.camera_thread.isRunning():
            self.camera_thread.quit()
            self.camera_thread.wait()
        if self.camera is not None and hasattr(self.camera, "close"):
            self.camera.close()
        self.camera = None
        self.camera_worker = None
        self.log("相机已停止")

    def _load_defaults(self) -> None:
        slm1_comp = self.config.get("slm1", {}).get("compensation_path", "")
        slm2_comp = self.config.get("slm2", {}).get("compensation_path", "")
        slm1_type = self.config.get("slm1", {}).get("device_type", "upo")
        slm2_type = self.config.get("slm2", {}).get("device_type", "holoeye")
        if slm1_comp:
            self.image_panel.slm1_comp_edit.setText(slm1_comp)
        if slm2_comp:
            self.slm2_panel.slm2_comp_edit.setText(slm2_comp)
        slm1_index = self.image_panel.device_combo.findData(slm1_type)
        if slm1_index >= 0:
            self.image_panel.device_combo.setCurrentIndex(slm1_index)
        slm2_index = self.slm2_panel.device_combo.findData(slm2_type)
        if slm2_index >= 0:
            self.slm2_panel.device_combo.setCurrentIndex(slm2_index)

        interval_ms = int(self.config.get("slm1", {}).get("play_interval_ms", 500))
        self.image_panel.interval_spin.setValue(interval_ms)

        for idx, widget in enumerate(self.slm2_panel.layer_widgets):
            layer_cfg = self.config.get("slm2", {}).get("layers", [])
            if idx < len(layer_cfg):
                widget.file_edit.setText(layer_cfg[idx].get("path", ""))
                widget.dx_spin.setValue(int(layer_cfg[idx].get("dx", 0)))
                widget.dy_spin.setValue(int(layer_cfg[idx].get("dy", 0)))
                widget.enable_checkbox.setChecked(bool(layer_cfg[idx].get("enabled", True)))

        exposure_us = float(self.config.get("camera", {}).get("exposure_us", 20000))
        self.camera_control.exposure_spin.setValue(exposure_us)
        record_interval = int(self.config.get("camera", {}).get("record_interval_ms", 200))
        self.camera_control.record_interval_spin.setValue(record_interval)

    def log(self, message: str) -> None:
        self.log_panel.append(message)
        print(message)

    def _slm_shape_for_type(self, device_type: str) -> tuple[int, int]:
        if device_type == "holoeye":
            return (1080, 1920)
        return (1200, 1920)

    def _create_slm_controller(self, role: str):
        output_cfg = self.config.get("output", {})
        slm1_dir = Path(output_cfg.get("slm1_tmp_dir", "output/slm1_tmp"))
        slm2_dir = Path(output_cfg.get("slm2_tmp_dir", "output/slm2_tmp"))
        slm1_dir.mkdir(parents=True, exist_ok=True)
        slm2_dir.mkdir(parents=True, exist_ok=True)

        if role == "slm1":
            device_type = self.image_panel.device_combo.currentData() or self.config.get("slm1", {}).get("device_type", "upo")
            cfg = dict(self.config.get("slm1", {}))
            if device_type == "holoeye" and not cfg.get("sdk_path"):
                cfg["sdk_path"] = self.config.get("slm2", {}).get("sdk_path", "")
            if device_type == "upo":
                return MockSLM1(self._slm_shape_for_type(device_type)) if self.mock else SLM1Controller(cfg, slm1_dir)
            return MockSLM2({"slm2_size": self._slm_shape_for_type(device_type)}) if self.mock else SLM2Controller(cfg, slm1_dir)

        device_type = self.slm2_panel.device_combo.currentData() or self.config.get("slm2", {}).get("device_type", "holoeye")
        cfg = dict(self.config.get("slm2", {}))
        if device_type == "upo":
            if "screen_num" not in cfg:
                cfg["screen_num"] = self.config.get("slm1", {}).get("screen_num", 1)
            return MockSLM1(self._slm_shape_for_type(device_type)) if self.mock else SLM1Controller(cfg, slm2_dir)
        return MockSLM2({"slm2_size": self._slm_shape_for_type(device_type)}) if self.mock else SLM2Controller(cfg, slm2_dir)

    def _start_workers(self) -> None:
        self.start_slm1()
        self.start_slm2()
        self.start_camera()

    def run_all(self) -> None:
        if self.camera_thread.isRunning() or self.slm1_thread.isRunning() or self.slm2_thread.isRunning():
            return
        self.log("启动运行流程")
        self._start_workers()
        self.start_playback()

    def stop_all(self) -> None:
        self.log("停止运行流程")
        self.pause_playback()
        self.stop_camera()
        self.stop_slm1()
        self.stop_slm2()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.stop_all()
        event.accept()

    def _collect_image_paths(self) -> None:
        if (self.image_panel.load_mode_combo.currentData() or "file") != "file":
            self.image_paths = []
            self.image_index = 0
            return
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
        if (self.image_panel.play_mode_combo.currentData() or "single") != "continuous":
            return
        if (self.image_panel.load_mode_combo.currentData() or "file") != "file":
            self.log("连续播放仅支持文件模式")
            return
        self._collect_image_paths()
        if not self.image_paths:
            self.log("未找到输入图像")
            return
        interval = self.image_panel.interval_spin.value()
        self.timer.start(interval)
        self.show_current_image()

    def pause_playback(self) -> None:
        self.timer.stop()

    def show_single_frame(self) -> None:
        self.pause_playback()
        self.generate_hologram()

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
            if self.image_panel.loop_checkbox.isChecked():
                self.image_index = 0
            else:
                self.timer.stop()
                return
        else:
            self.image_index += 1
        self.show_current_image()

    def generate_hologram(self) -> None:
        load_mode = self.image_panel.load_mode_combo.currentData() or "file"
        if load_mode == "file":
            input_type = self.image_panel.input_type_combo.currentData() or "hologram"
            if not self.image_paths:
                self._collect_image_paths()
            if not self.image_paths:
                self.log("未选择输入图像")
                return
        else:
            input_type = "field"
        if self.slm1_worker is None:
            self.log("SLM1 未初始化，请先 Run")
            return
        self.load_slm1()

    def load_slm1(self) -> None:
        load_mode = self.image_panel.load_mode_combo.currentData() or "file"
        if load_mode == "file":
            input_type = self.image_panel.input_type_combo.currentData() or "hologram"
            if not self.image_paths:
                return
            path = str(self.image_paths[self.image_index])
        else:
            input_type = "field"
            path = ""
        use_comp = self.image_panel.slm1_comp_checkbox.isChecked()
        comp_path = self.image_panel.slm1_comp_edit.text().strip()
        field_params = {
            "mode": "file" if (load_mode == "file" and input_type == "field") else self.image_panel.field_mode_combo.currentData(),
            "w0": self.image_panel.lg_w0_spin.value(),
            "p": self.image_panel.lg_p_spin.value(),
            "l": self.image_panel.lg_l_spin.value(),
            "letter": self.image_panel.letter_edit.text().strip(),
            "index": self.image_panel.dataset_index_spin.value(),
            "size": (self.image_panel.field_width_spin.value(), self.image_panel.field_height_spin.value()),
            "data_dir": self.config.get("datasets", {}).get("mnist_dir", "data/mnist"),
        }
        QtCore.QMetaObject.invokeMethod(
            self.slm1_worker,
            "load_hologram",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(str, path),
            QtCore.Q_ARG(bool, use_comp),
            QtCore.Q_ARG(str, comp_path),
            QtCore.Q_ARG(str, input_type),
            QtCore.Q_ARG(object, field_params),
        )

    def on_layer_change(self, _: int) -> None:
        if self.slm2_panel.auto_apply_checkbox.isChecked():
            self.apply_slm2()
        else:
            self.update_preview()

    def on_slm1_auto_apply_change(self, *_args) -> None:
        if self.image_panel.auto_apply_checkbox.isChecked():
            self.generate_hologram()

    def apply_slm2(self) -> None:
        try:
            phase = self.build_slm2_phase()
        except Exception as exc:
            self.log(f"SLM2 合成失败: {exc}")
            return
        if self.slm2_worker is None:
            self.log("SLM2 未初始化，请先 Run")
            self.update_preview()
            return

        use_comp = self.slm2_panel.slm2_comp_checkbox.isChecked()
        comp_path = self.slm2_panel.slm2_comp_edit.text().strip()
        preview_phase = phase
        if use_comp and comp_path:
            comp = load_compensation(comp_path, phase.shape, meaning="encoded_0_255")
            preview_phase = apply_compensation(phase, comp)
        self._latest_slm2_loaded_phase = preview_phase

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
            slm2_type = self.slm2_panel.device_combo.currentData() or self.config.get("slm2", {}).get("device_type", "holoeye")
            slm2_shape = self._slm_shape_for_type(slm2_type)
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

    def on_slm2_phase_ready(self, loaded: Optional[np.ndarray], returned: Optional[np.ndarray]) -> None:
        self._latest_slm2_loaded_phase = loaded
        self._latest_slm2_returned_phase = returned
        self.update_preview()

    def on_slm1_image_ready(self, img_u8: np.ndarray) -> None:
        self._latest_slm1_image = img_u8
        img = Image.fromarray(img_u8, mode="L").convert("RGB")
        qimage = QtGui.QImage(
            img.tobytes("raw", "RGB"),
            img.width,
            img.height,
            QtGui.QImage.Format.Format_RGB888,
        )
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.slm1_preview.update_pixmap(pixmap)

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
            slm2_type = self.slm2_panel.device_combo.currentData() or self.config.get("slm2", {}).get("device_type", "holoeye")
            slm2_shape = self._slm_shape_for_type(slm2_type)
        mode = self.preview_panel.display_mode()
        base_phase = None
        if mode == "returned":
            base_phase = self._latest_slm2_returned_phase
            if base_phase is None:
                base_phase = self._latest_slm2_loaded_phase
        else:
            base_phase = self._latest_slm2_loaded_phase
        label_size = self.preview_panel.label.size()
        scale = min(label_size.width() / slm2_shape[1], label_size.height() / slm2_shape[0], 1.0)
        scale = max(scale, 0.05)
        img = make_preview_image(slm2_shape, rects, centers, enabled, base_phase=base_phase, scale=scale)
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
        self.view_box.setAspectLocked(True, ratio=frame.shape[1] / frame.shape[0])
        self.status_panel.update_status(fps=fps)
        self._check_overexposure(frame)
        if self.recording and self.record_dir:
            now = time.time()
            interval_s = self.camera_control.record_interval_spin.value() / 1000.0
            if now - self.last_record_time >= interval_s:
                if not (self.overexposed and self.camera_control.auto_reduce_checkbox.isChecked()):
                    timestamp = QtCore.QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss_zzz")
                    path = self.record_dir / f"frame_{timestamp}.png"
                    img = np.clip(frame, 0, 255).astype(np.uint8)
                    if self.camera_control.save_roi_checkbox.isChecked():
                        img = self._crop_to_roi(img)
                    Image.fromarray(img).save(path)
                self.last_record_time = now
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

    def on_roi_changed(self) -> None:
        if self.image_item.image is None:
            return
        self.update_roi_stats()

    def reset_camera_view(self) -> None:
        if self.image_item.image is None:
            self.view_box.autoRange()
            return
        img = self.image_item.image
        rect = QtCore.QRectF(0, 0, img.shape[1], img.shape[0])
        self.view_box.setRange(rect, padding=0.05)

    def zoom_to_roi(self) -> None:
        if self.image_item.image is None:
            return
        roi_bounds = self.roi.parentBounds()
        if roi_bounds.width() <= 1 or roi_bounds.height() <= 1:
            return
        self.view_box.setRange(roi_bounds, padding=0.05)

    def on_camera_mode_changed(self) -> None:
        mode = "select" if self.camera_mode_button.isChecked() else "pan"
        if mode == "select":
            self.roi.setVisible(True)
            self.view_box.set_mode("select")
            if self._rect_mode is not None:
                self.view_box.setMouseMode(self._rect_mode)
        else:
            self.roi.setVisible(False)
            self.view_box.set_mode("pan")
            if self._pan_mode is not None:
                self.view_box.setMouseMode(self._pan_mode)
        icon = self.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton if mode == "select" else QtWidgets.QStyle.StandardPixmap.SP_ArrowForward
        )
        self.camera_mode_button.setIcon(icon)

    def toggle_camera_mode(self) -> None:
        self.on_camera_mode_changed()

    def on_camera_select_drag(self, start: QtCore.QPointF, end: QtCore.QPointF, finished: bool) -> None:
        p1 = self.view_box.mapSceneToView(start)
        p2 = self.view_box.mapSceneToView(end)
        x0, x1 = sorted([p1.x(), p2.x()])
        y0, y1 = sorted([p1.y(), p2.y()])
        if x1 - x0 <= 1 or y1 - y0 <= 1:
            return
        self.roi.setPos((x0, y0))
        self.roi.setSize((x1 - x0, y1 - y0))
        if finished:
            self.on_roi_changed()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if self.roi.isVisible() and self.roi.isSelected():
            if event.key() in (QtCore.Qt.Key.Key_Escape, QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
                self.roi.setSelected(False)
                self.roi.setVisible(False)
                return
        super().keyPressEvent(event)

    def _crop_to_roi(self, img: np.ndarray) -> np.ndarray:
        roi_bounds = self.roi.parentBounds()
        x0 = max(int(roi_bounds.left()), 0)
        y0 = max(int(roi_bounds.top()), 0)
        x1 = min(int(roi_bounds.right()), img.shape[1] - 1)
        y1 = min(int(roi_bounds.bottom()), img.shape[0] - 1)
        if x1 <= x0 or y1 <= y0:
            return img
        return img[y0:y1, x0:x1]

    def _check_overexposure(self, frame: np.ndarray) -> None:
        threshold = float(self.config.get("camera", {}).get("overexposure_threshold", 250))
        ratio = float(self.config.get("camera", {}).get("overexposure_ratio", 0.01))
        over = np.mean(frame >= threshold)
        self.overexposed = over >= ratio
        self.status_panel.update_status(overexposure=f"{over:.2%}" if self.overexposed else "正常")
        if self.overexposed and self.camera_control.auto_reduce_checkbox.isChecked():
            new_exposure = max(100.0, self.camera_control.exposure_spin.value() * 0.8)
            self.camera_control.exposure_spin.setValue(new_exposure)
            self.apply_exposure()

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
            if self.camera_control.save_roi_checkbox.isChecked():
                img = self._crop_to_roi(img)
            Image.fromarray(img).save(path)
            self.log(f"保存当前帧: {path}")

    def start_recording(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not folder:
            return
        self.record_dir = Path(folder)
        self.recording = True
        self.last_record_time = 0.0
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

    def apply_exposure(self) -> None:
        if self.camera_worker is None:
            self.log("相机未初始化，无法设置曝光")
            return
        exposure_us = self.camera_control.exposure_spin.value()
        QtCore.QMetaObject.invokeMethod(
            self.camera_worker,
            "set_exposure",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(float, exposure_us),
        )
        self.log(f"设置曝光: {exposure_us:.0f} us")


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

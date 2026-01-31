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


# ... (Imports 和 Mock 类保持不变) ...

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

        # 确保窗口能接收键盘事件
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def _setup_ui(self) -> None:
        # ... (布局代码保持不变，与上一次一致，主要是连接 image_panel) ...
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        self.image_panel = ImageSourcePanel()
        self.slm2_panel = SLM2Panel()
        self.preview_panel = PreviewPanel()
        self.slm1_preview = SLMPreviewPanel("SLM1 预览")
        self.status_panel = StatusPanel()
        self.log_panel = LogPanel()
        self.camera_control = CameraControlPanel()

        self.run_button = QtWidgets.QPushButton("Run 全流程")
        self.stop_button = QtWidgets.QPushButton("Stop 全流程")
        self.save_button = QtWidgets.QPushButton("保存当前帧")
        self.record_button = QtWidgets.QPushButton("开始连续保存")
        self.stop_record_button = QtWidgets.QPushButton("停止连续保存")

        self.roi_panel = RoiStatsPanel()

        # 连接 image_panel 的新按钮 (修正为新控件名)
        self.image_panel.single_button.clicked.connect(self.show_single_frame)
        self.image_panel.play_button.clicked.connect(self.start_playback)
        self.image_panel.stop_play_button.clicked.connect(self.pause_playback)
        self.image_panel.prev_button.clicked.connect(self.show_prev_image)
        self.image_panel.next_button.clicked.connect(self.show_next_image)
        self.image_panel.run_button.clicked.connect(self.start_slm1)
        self.image_panel.stop_button.clicked.connect(self.stop_slm1)

        self.image_panel.param_changed.connect(self.on_slm1_auto_apply_change)
        self.image_panel.auto_apply_checkbox.stateChanged.connect(self.on_slm1_auto_apply_change)

        # SLM2 连接
        for widget in self.slm2_panel.layer_widgets:
            widget.file_changed.connect(self.on_layer_change)
            widget.enabled_changed.connect(self.on_layer_change)
            widget.offset_changed.connect(self.on_layer_change)
            widget.flip_changed.connect(self.on_layer_change)

        self.slm2_panel.apply_button.clicked.connect(self.apply_slm2)
        self.slm2_panel.global_flip_h.stateChanged.connect(self.on_layer_change)
        self.slm2_panel.global_flip_v.stateChanged.connect(self.on_layer_change)

        self.preview_panel.display_mode_changed.connect(lambda _: self.update_preview())
        self.slm2_panel.run_button.clicked.connect(self.start_slm2)
        self.slm2_panel.stop_button.clicked.connect(self.stop_slm2)
        self.slm2_panel.slm2_comp_checkbox.stateChanged.connect(self.on_layer_change)
        self.slm2_panel.slm2_comp_edit.textChanged.connect(self.on_layer_change)
        self.slm2_panel.auto_apply_checkbox.stateChanged.connect(self.on_layer_change)

        self.run_button.clicked.connect(self.run_all)
        self.stop_button.clicked.connect(self.stop_all)
        self.save_button.clicked.connect(self.save_frame)
        self.record_button.clicked.connect(self.start_recording)
        self.stop_record_button.clicked.connect(self.stop_recording)
        self.camera_control.run_button.clicked.connect(self.start_camera)
        self.camera_control.stop_button.clicked.connect(self.stop_camera)
        self.camera_control.apply_exposure_button.clicked.connect(self.apply_exposure)

        # 布局组装 (保持不变)
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

    # ... (_build_camera_view, _build_left_split, _build_center_split, _setup_threads, start_slm1 等保持不变) ...
    # ... 为了节省篇幅，这里假设你保留了 start_slm1, stop_slm1, start_slm2 等初始化方法 ...

    # 重点修正：文件路径收集
    def _collect_image_paths(self) -> None:
        load_mode = self.image_panel.load_mode_combo.currentData() or "file"
        if load_mode != "file":
            self.image_paths = []
            self.image_index = 0
            return

        # 修正：根据下拉框判断单图还是文件夹
        source_type = self.image_panel.file_source_type_combo.currentData()

        paths = []
        if source_type == "folder":
            folder = self.image_panel.folder_edit.text().strip()
            if folder:
                folder_path = Path(folder)
                # 兼容常见图片格式
                for ext in ("*.png", "*.bmp", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
                    paths.extend(sorted(folder_path.glob(ext)))
        else:  # single_file
            image = self.image_panel.image_edit.text().strip()
            if image:
                paths.append(Path(image))

        self.image_paths = [p for p in paths if p.exists()]
        self.image_index = 0
        self.log(f"已收集 {len(self.image_paths)} 张图像 (模式: {source_type})")

    # 重点修正：播放逻辑
    def start_playback(self) -> None:
        # 不再强制检查 play_mode_combo，用户点了开始就是想开始
        # 但如果是 generate 模式，也允许播放（可能只是循环刷新）
        # 这里主要针对 File 模式
        if (self.image_panel.load_mode_combo.currentData() == "file"):
            self._collect_image_paths()
            if not self.image_paths:
                self.log("未找到输入图像，请检查路径")
                return

        interval = self.image_panel.interval_spin.value()
        self.timer.start(interval)
        self.log(f"开始连续播放，间隔 {interval} ms")
        self.show_current_image()

    def pause_playback(self) -> None:
        self.timer.stop()
        self.log("停止播放")

    # 重点修正：ROI 删除逻辑
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        # 删除 ROI 的逻辑
        if event.key() in (QtCore.Qt.Key.Key_Escape, QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            # 检查 ROI 是否可见且被选中（或者仅仅是想取消它）
            if self.roi.isVisible():
                self.roi.setVisible(False)
                # 清除选中状态，如果有的话
                if hasattr(self.roi, "setSelected"):
                    self.roi.setSelected(False)
                # 清除测量面板的数据显示
                self.roi_panel.update_pixel("ROI 已移除")
                self.log("ROI 已移除")
                event.accept()
                return

        super().keyPressEvent(event)

    # ... (其余方法如 show_single_frame, generate_hologram 等保持不变，
    #      确保 generate_hologram 使用 self.image_panel.interval_spin 即可) ...

    # ... (load_slm1, apply_slm2, update_frame 等保持上一次修改后的带翻转逻辑的版本) ...

    # 为了确保完整性，这里再贴一次 load_slm1，因为控件名变了
    def load_slm1(self) -> None:
        load_mode = self.image_panel.load_mode_combo.currentData() or "file"
        if load_mode == "file":
            input_type = self.image_panel.input_type_combo.currentData() or "hologram"
            if not self.image_paths:
                self._collect_image_paths()  # 再次尝试收集
            if not self.image_paths:
                return
            # 防止索引越界
            if self.image_index >= len(self.image_paths):
                self.image_index = 0
            path = str(self.image_paths[self.image_index])
        else:
            input_type = "field"
            path = ""

        use_comp = self.image_panel.slm1_comp_checkbox.isChecked()
        comp_path = self.image_panel.slm1_comp_edit.text().strip()

        flip_h = self.image_panel.flip_h_checkbox.isChecked()
        flip_v = self.image_panel.flip_v_checkbox.isChecked()

        field_params = {
            "mode": "file" if (
                        load_mode == "file" and input_type == "field") else self.image_panel.field_mode_combo.currentData(),
            "w0": self.image_panel.lg_w0_spin.value(),
            "p": self.image_panel.lg_p_spin.value(),
            "l": self.image_panel.lg_l_spin.value(),
            "letter": self.image_panel.letter_edit.text().strip(),
            "index": self.image_panel.dataset_index_spin.value(),
            "size": (self.image_panel.field_width_spin.value(), self.image_panel.field_height_spin.value()),
            "data_dir": self.config.get("datasets", {}).get("mnist_dir", "data/mnist"),
        }

        if self.slm1_worker:
            QtCore.QMetaObject.invokeMethod(
                self.slm1_worker,
                "load_hologram",
                QtCore.Qt.ConnectionType.QueuedConnection,
                QtCore.Q_ARG(str, path),
                QtCore.Q_ARG(bool, use_comp),
                QtCore.Q_ARG(str, comp_path),
                QtCore.Q_ARG(str, input_type),
                QtCore.Q_ARG(object, field_params),
                QtCore.Q_ARG(bool, flip_h),
                QtCore.Q_ARG(bool, flip_v),
            )


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
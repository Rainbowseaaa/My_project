from dataclasses import dataclass
from typing import List

from PyQt6 import QtCore, QtGui, QtWidgets


@dataclass
class LayerConfig:
    enabled: bool
    file_path: str
    dx: int
    dy: int


class LayerWidget(QtWidgets.QGroupBox):
    file_changed = QtCore.pyqtSignal(int)
    enabled_changed = QtCore.pyqtSignal(int)
    offset_changed = QtCore.pyqtSignal(int)

    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.index = index
        self.setTitle(f"Layer {index + 1}")

        self.enable_checkbox = QtWidgets.QCheckBox("启用")
        self.enable_checkbox.setChecked(True)
        self.file_edit = QtWidgets.QLineEdit()
        self.file_button = QtWidgets.QPushButton("选择相位文件")
        self.dx_spin = QtWidgets.QSpinBox()
        self.dy_spin = QtWidgets.QSpinBox()
        self.dx_spin.setRange(-2000, 2000)
        self.dy_spin.setRange(-2000, 2000)
        self.dx_spin.setSingleStep(1)
        self.dy_spin.setSingleStep(1)

        offset_layout = QtWidgets.QFormLayout()
        offset_layout.addRow("dx (px)", self.dx_spin)
        offset_layout.addRow("dy (px)", self.dy_spin)

        file_layout = QtWidgets.QHBoxLayout()
        file_layout.addWidget(self.file_edit, 1)
        file_layout.addWidget(self.file_button)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.enable_checkbox)
        layout.addLayout(file_layout)
        layout.addLayout(offset_layout)
        self.setLayout(layout)

        self.file_button.clicked.connect(self._pick_file)
        self.enable_checkbox.stateChanged.connect(lambda _: self.enabled_changed.emit(self.index))
        self.dx_spin.valueChanged.connect(lambda _: self.offset_changed.emit(self.index))
        self.dy_spin.valueChanged.connect(lambda _: self.offset_changed.emit(self.index))

    def _pick_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择相位文件",
            "",
            "Phase Files (*.npy *.png *.bmp *.mat *.tif *.tiff)",
        )
        if path:
            self.file_edit.setText(path)
            self.file_changed.emit(self.index)

    def get_config(self) -> LayerConfig:
        return LayerConfig(
            enabled=self.enable_checkbox.isChecked(),
            file_path=self.file_edit.text().strip(),
            dx=self.dx_spin.value(),
            dy=self.dy_spin.value(),
        )


class LogPanel(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__("日志", parent)
        self.text = QtWidgets.QTextEdit()
        self.text.setReadOnly(True)
        self.text.setMinimumHeight(120)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.text)
        self.setLayout(layout)

    def append(self, message: str) -> None:
        self.text.append(message)


class StatusPanel(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__("状态", parent)
        self.image_label = QtWidgets.QLabel("当前图像: -")
        self.fps_label = QtWidgets.QLabel("FPS: -")
        self.slm1_label = QtWidgets.QLabel("SLM1: -")
        self.slm2_label = QtWidgets.QLabel("SLM2: -")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.fps_label)
        layout.addWidget(self.slm1_label)
        layout.addWidget(self.slm2_label)
        self.setLayout(layout)

    def update_status(self, image_name: str = None, fps: float = None, slm1: str = None, slm2: str = None) -> None:
        if image_name is not None:
            self.image_label.setText(f"当前图像: {image_name}")
        if fps is not None:
            self.fps_label.setText(f"FPS: {fps:.1f}")
        if slm1 is not None:
            self.slm1_label.setText(f"SLM1: {slm1}")
        if slm2 is not None:
            self.slm2_label.setText(f"SLM2: {slm2}")


class PreviewPanel(QtWidgets.QGroupBox):
    display_mode_changed = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__("SLM2 窗口示意", parent)
        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(240, 160)
        self.label.setStyleSheet("background-color: #222; border: 1px solid #444;")

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("显示加载结果", userData="loaded")
        self.mode_combo.addItem("显示 SLM2 返回", userData="returned")

        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.addWidget(QtWidgets.QLabel("显示源"))
        mode_layout.addWidget(self.mode_combo, 1)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(mode_layout)
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.mode_combo.currentIndexChanged.connect(self._emit_mode)

    def _emit_mode(self) -> None:
        self.display_mode_changed.emit(self.display_mode())

    def display_mode(self) -> str:
        data = self.mode_combo.currentData()
        return data if data else "loaded"

    def update_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        if pixmap is None:
            return
        scaled = pixmap.scaled(
            self.label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.label.setPixmap(scaled)


class RoiStatsPanel(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__("ROI 测量", parent)
        self.mean_label = QtWidgets.QLabel("Mean: -")
        self.min_label = QtWidgets.QLabel("Min: -")
        self.max_label = QtWidgets.QLabel("Max: -")
        self.sum_label = QtWidgets.QLabel("Sum: -")
        self.centroid_label = QtWidgets.QLabel("Centroid: -")
        self.pixel_label = QtWidgets.QLabel("Pixel: -")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.mean_label)
        layout.addWidget(self.min_label)
        layout.addWidget(self.max_label)
        layout.addWidget(self.sum_label)
        layout.addWidget(self.centroid_label)
        layout.addWidget(self.pixel_label)
        self.setLayout(layout)

    def update_stats(self, mean=None, min_val=None, max_val=None, sum_val=None, centroid=None) -> None:
        if mean is not None:
            self.mean_label.setText(f"Mean: {mean:.2f}")
        if min_val is not None:
            self.min_label.setText(f"Min: {min_val:.2f}")
        if max_val is not None:
            self.max_label.setText(f"Max: {max_val:.2f}")
        if sum_val is not None:
            self.sum_label.setText(f"Sum: {sum_val:.2f}")
        if centroid is not None:
            cx, cy = centroid
            self.centroid_label.setText(f"Centroid: ({cx:.1f}, {cy:.1f})")

    def update_pixel(self, text: str) -> None:
        self.pixel_label.setText(text)


class PlayerControls(QtWidgets.QGroupBox):
    play_clicked = QtCore.pyqtSignal()
    pause_clicked = QtCore.pyqtSignal()
    prev_clicked = QtCore.pyqtSignal()
    next_clicked = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("播放控制", parent)
        self.play_button = QtWidgets.QPushButton("播放")
        self.pause_button = QtWidgets.QPushButton("暂停")
        self.prev_button = QtWidgets.QPushButton("上一张")
        self.next_button = QtWidgets.QPushButton("下一张")
        self.loop_checkbox = QtWidgets.QCheckBox("循环")
        self.loop_checkbox.setChecked(True)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.play_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.prev_button)
        layout.addWidget(self.next_button)
        layout.addWidget(self.loop_checkbox)
        self.setLayout(layout)

        self.play_button.clicked.connect(self.play_clicked.emit)
        self.pause_button.clicked.connect(self.pause_clicked.emit)
        self.prev_button.clicked.connect(self.prev_clicked.emit)
        self.next_button.clicked.connect(self.next_clicked.emit)

    def loop_enabled(self) -> bool:
        return self.loop_checkbox.isChecked()


class ImageSourcePanel(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__("SLM1 输入图像", parent)
        self.image_edit = QtWidgets.QLineEdit()
        self.image_button = QtWidgets.QPushButton("选择单张图")
        self.folder_edit = QtWidgets.QLineEdit()
        self.folder_button = QtWidgets.QPushButton("选择文件夹")
        self.interval_spin = QtWidgets.QSpinBox()
        self.interval_spin.setRange(50, 5000)
        self.interval_spin.setValue(500)
        self.interval_spin.setSuffix(" ms")

        self.generate_button = QtWidgets.QPushButton("生成全息图")
        self.load_button = QtWidgets.QPushButton("加载到 SLM1")

        self.slm1_comp_checkbox = QtWidgets.QCheckBox("叠加 SLM1 补偿")
        self.slm1_comp_edit = QtWidgets.QLineEdit()
        self.slm1_comp_button = QtWidgets.QPushButton("选择补偿文件")

        top_layout = QtWidgets.QGridLayout()
        top_layout.addWidget(self.image_edit, 0, 0)
        top_layout.addWidget(self.image_button, 0, 1)
        top_layout.addWidget(self.folder_edit, 1, 0)
        top_layout.addWidget(self.folder_button, 1, 1)
        top_layout.addWidget(QtWidgets.QLabel("播放间隔"), 2, 0)
        top_layout.addWidget(self.interval_spin, 2, 1)

        comp_layout = QtWidgets.QHBoxLayout()
        comp_layout.addWidget(self.slm1_comp_checkbox)
        comp_layout.addWidget(self.slm1_comp_edit, 1)
        comp_layout.addWidget(self.slm1_comp_button)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addWidget(self.generate_button)
        layout.addWidget(self.load_button)
        layout.addLayout(comp_layout)
        self.setLayout(layout)

        self.image_button.clicked.connect(self._pick_image)
        self.folder_button.clicked.connect(self._pick_folder)
        self.slm1_comp_button.clicked.connect(self._pick_slm1_comp)

    def _pick_image(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择输入图像",
            "",
            "Images (*.png *.bmp *.jpg *.jpeg *.tif *.tiff)",
        )
        if path:
            self.image_edit.setText(path)

    def _pick_folder(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if path:
            self.folder_edit.setText(path)

    def _pick_slm1_comp(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择 SLM1 补偿文件",
            "",
            "Images (*.png *.bmp *.tif *.tiff)",
        )
        if path:
            self.slm1_comp_edit.setText(path)


class SLM2Panel(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__("SLM2 四层相位")
        self.layer_widgets: List[LayerWidget] = [LayerWidget(i) for i in range(4)]
        self.apply_button = QtWidgets.QPushButton("合成并加载 SLM2")
        self.auto_apply_checkbox = QtWidgets.QCheckBox("Auto Apply")
        self.auto_apply_checkbox.setChecked(False)

        self.slm2_comp_checkbox = QtWidgets.QCheckBox("叠加 SLM2 补偿")
        self.slm2_comp_edit = QtWidgets.QLineEdit()
        self.slm2_comp_button = QtWidgets.QPushButton("选择补偿文件")

        comp_layout = QtWidgets.QHBoxLayout()
        comp_layout.addWidget(self.slm2_comp_checkbox)
        comp_layout.addWidget(self.slm2_comp_edit, 1)
        comp_layout.addWidget(self.slm2_comp_button)

        layout = QtWidgets.QVBoxLayout()
        for widget in self.layer_widgets:
            layout.addWidget(widget)
        layout.addLayout(comp_layout)
        layout.addWidget(self.auto_apply_checkbox)
        layout.addWidget(self.apply_button)
        self.setLayout(layout)

        self.slm2_comp_button.clicked.connect(self._pick_slm2_comp)

    def _pick_slm2_comp(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择 SLM2 补偿文件",
            "",
            "Images (*.png *.bmp *.tif *.tiff)",
        )
        if path:
            self.slm2_comp_edit.setText(path)

    def get_layers(self) -> List[LayerConfig]:
        return [widget.get_config() for widget in self.layer_widgets]

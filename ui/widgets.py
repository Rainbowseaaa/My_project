from dataclasses import dataclass
from typing import List

from PyQt6 import QtCore, QtGui, QtWidgets


@dataclass
class LayerConfig:
    enabled: bool
    file_path: str
    cx: int  # 绝对坐标 X
    cy: int  # 绝对坐标 Y
    flip_h: bool
    flip_v: bool


class LayerWidget(QtWidgets.QGroupBox):
    file_changed = QtCore.pyqtSignal(int)
    enabled_changed = QtCore.pyqtSignal(int)
    center_changed = QtCore.pyqtSignal(int)
    flip_changed = QtCore.pyqtSignal(int)

    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.index = index
        self.setTitle(f"Layer {index + 1}")

        self.enable_checkbox = QtWidgets.QCheckBox("启用")
        self.enable_checkbox.setChecked(True)
        self.file_edit = QtWidgets.QLineEdit()
        self.file_button = QtWidgets.QPushButton("...")

        # 使用 cx/cy (绝对坐标)
        self.cx_spin = QtWidgets.QSpinBox()
        self.cy_spin = QtWidgets.QSpinBox()

        # 范围设大
        self.cx_spin.setRange(-2000, 4000)
        self.cy_spin.setRange(-2000, 4000)

        # 开启加速 (长按跳得快)
        self.cx_spin.setAccelerated(True)
        self.cy_spin.setAccelerated(True)

        # 【核心修复】通过样式表强制加宽按钮，确保肉眼可见！
        spin_style = """
            QSpinBox {
                min-width: 100px;  /* 保证足够宽度 */
                min-height: 25px;  /* 稍微高一点 */
                font-size: 14px;   /* 字号适中 */
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 25px;       /* 【重点】把加减按钮加宽！ */
                border: 1px solid #555; /* 加个边框更明显 */
                background: #ddd;       /* 给按钮一点背景色 */
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background: #bbb;       /* 悬停变色 */
            }
        """
        self.cx_spin.setStyleSheet(spin_style)
        self.cy_spin.setStyleSheet(spin_style)

        self.flip_h_check = QtWidgets.QCheckBox("H翻转")
        self.flip_v_check = QtWidgets.QCheckBox("V翻转")

        # 布局
        ctrl_layout = QtWidgets.QHBoxLayout()
        ctrl_layout.addWidget(QtWidgets.QLabel("X:"))
        ctrl_layout.addWidget(self.cx_spin)
        ctrl_layout.addWidget(QtWidgets.QLabel("Y:"))
        ctrl_layout.addWidget(self.cy_spin)
        ctrl_layout.addWidget(self.flip_h_check)
        ctrl_layout.addWidget(self.flip_v_check)
        ctrl_layout.addStretch(1)

        file_layout = QtWidgets.QHBoxLayout()
        file_layout.addWidget(self.file_edit, 1)
        file_layout.addWidget(self.file_button)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.enable_checkbox)
        layout.addLayout(file_layout)
        layout.addLayout(ctrl_layout)
        self.setLayout(layout)

        self.file_button.clicked.connect(self._pick_file)
        self.enable_checkbox.stateChanged.connect(lambda _: self.enabled_changed.emit(self.index))
        self.cx_spin.valueChanged.connect(lambda _: self.center_changed.emit(self.index))
        self.cy_spin.valueChanged.connect(lambda _: self.center_changed.emit(self.index))
        self.flip_h_check.stateChanged.connect(lambda _: self.flip_changed.emit(self.index))
        self.flip_v_check.stateChanged.connect(lambda _: self.flip_changed.emit(self.index))

    def _pick_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择相位文件", "", "Phase Files (*.npy *.png *.bmp *.mat *.tif *.tiff)"
        )
        if path:
            self.file_edit.setText(path)
            self.file_changed.emit(self.index)

    def set_center(self, x: int, y: int) -> None:
        self.cx_spin.setValue(int(x))
        self.cy_spin.setValue(int(y))

    def get_config(self) -> LayerConfig:
        return LayerConfig(
            enabled=self.enable_checkbox.isChecked(),
            file_path=self.file_edit.text().strip(),
            cx=self.cx_spin.value(),
            cy=self.cy_spin.value(),
            flip_h=self.flip_h_check.isChecked(),
            flip_v=self.flip_v_check.isChecked()
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
        self.overexposure_label = QtWidgets.QLabel("过曝: -")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.fps_label)
        layout.addWidget(self.slm1_label)
        layout.addWidget(self.slm2_label)
        layout.addWidget(self.overexposure_label)
        self.setLayout(layout)

    def update_status(
            self,
            image_name: str = None,
            fps: float = None,
            slm1: str = None,
            slm2: str = None,
            overexposure: str = None,
    ) -> None:
        if image_name is not None:
            self.image_label.setText(f"当前图像: {image_name}")
        if fps is not None:
            self.fps_label.setText(f"FPS: {fps:.1f}")
        if slm1 is not None:
            self.slm1_label.setText(f"SLM1: {slm1}")
        if slm2 is not None:
            self.slm2_label.setText(f"SLM2: {slm2}")
        if overexposure is not None:
            self.overexposure_label.setText(f"过曝: {overexposure}")


class PreviewPanel(QtWidgets.QGroupBox):
    display_mode_changed = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__("SLM2 窗口示意", parent)
        self.label = ZoomLabel()
        self.label.setMinimumSize(240, 160)

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
        self.label.set_base_pixmap(pixmap)


class ZoomLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._base_pixmap: QtGui.QPixmap | None = None
        self._scale = 1.0
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #222; border: 1px solid #444;")

    def set_base_pixmap(self, pixmap: QtGui.QPixmap | None) -> None:
        self._base_pixmap = pixmap
        self._scale = 1.0
        self._update_pixmap()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self._base_pixmap is None:
            return
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 1 / 1.1
        self._scale = max(0.1, min(5.0, self._scale * factor))
        self._update_pixmap()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._base_pixmap is None:
            self.clear()
            return
        target = self.size() * self._scale
        scaled = self._base_pixmap.scaled(
            target,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


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
    # 定义 Auto Apply 信号，方便外部连接
    param_changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("SLM1 输入图像", parent)

        # --- 1. 设备选择 ---
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItem("UPO (1920x1200)", userData="upo")
        self.device_combo.addItem("Holoeye (1920x1080)", userData="holoeye")
        self.screen_combo = QtWidgets.QComboBox()
        self._populate_screen_combo(self.screen_combo)
        self.heds_selector_combo = QtWidgets.QComboBox()
        self.heds_refresh_button = QtWidgets.QToolButton()
        self.heds_refresh_button.setText("刷新")

        # --- 2. 加载模式 (顶层) ---
        self.load_mode_combo = QtWidgets.QComboBox()
        self.load_mode_combo.addItem("从文件加载", userData="file")
        self.load_mode_combo.addItem("自动生成", userData="generate")

        # ====== 文件加载区域 ======
        self.file_source_type_combo = QtWidgets.QComboBox()
        self.file_source_type_combo.addItem("单张图片文件", userData="single_file")
        self.file_source_type_combo.addItem("文件夹 (序列)", userData="folder")

        self.input_type_combo = QtWidgets.QComboBox()
        self.input_type_combo.addItem("作为全息图(直接加载)", userData="hologram")
        self.input_type_combo.addItem("作为光场(生成全息图)", userData="field")

        self.image_edit = QtWidgets.QLineEdit()
        self.image_button = QtWidgets.QPushButton("选择文件")
        self.folder_edit = QtWidgets.QLineEdit()
        self.folder_button = QtWidgets.QPushButton("选择文件夹")

        # 布局：文件设置
        file_layout = QtWidgets.QGridLayout()
        file_layout.addWidget(QtWidgets.QLabel("来源类型:"), 0, 0)
        file_layout.addWidget(self.file_source_type_combo, 0, 1)
        file_layout.addWidget(QtWidgets.QLabel("数据含义:"), 1, 0)
        file_layout.addWidget(self.input_type_combo, 1, 1)

        # 单文件控件行
        self.label_single = QtWidgets.QLabel("文件路径:")
        file_layout.addWidget(self.label_single, 2, 0)
        file_layout.addWidget(self.image_edit, 2, 1)
        file_layout.addWidget(self.image_button, 2, 2)

        # 文件夹控件行
        self.label_folder = QtWidgets.QLabel("文件夹路径:")
        file_layout.addWidget(self.label_folder, 3, 0)
        file_layout.addWidget(self.folder_edit, 3, 1)
        file_layout.addWidget(self.folder_button, 3, 2)

        file_widget = QtWidgets.QWidget()
        file_widget.setLayout(file_layout)

        # ====== 生成模式区域 ======
        self.field_mode_combo = QtWidgets.QComboBox()
        self.field_mode_combo.addItem("LG 光束", userData="lg")
        self.field_mode_combo.addItem("聚焦相位", userData="focus")
        self.field_mode_combo.addItem("字母", userData="letter")
        self.field_mode_combo.addItem("MNIST", userData="mnist")
        self.field_mode_combo.addItem("FashionMNIST", userData="fashion_mnist")

        self.lg_w0_spin = QtWidgets.QDoubleSpinBox()
        self.lg_w0_spin.setRange(1.0, 1000.0)
        self.lg_w0_spin.setValue(80.0)
        self.lg_p_edit = QtWidgets.QLineEdit("0")
        self.lg_l_edit = QtWidgets.QLineEdit("0")
        self.letter_edit = QtWidgets.QLineEdit("A")
        self.focus_distance_spin = QtWidgets.QDoubleSpinBox()
        self.focus_distance_spin.setRange(-10000.0, 10000.0)
        self.focus_distance_spin.setValue(245.0)
        self.focus_distance_spin.setSuffix(" mm")
        self.focus_diameter_spin = QtWidgets.QDoubleSpinBox()
        self.focus_diameter_spin.setRange(0.0, 10000.0)
        self.focus_diameter_spin.setValue(350.0)
        self.focus_diameter_spin.setSuffix(" px")
        self.dataset_index_spin = QtWidgets.QSpinBox()
        self.dataset_index_spin.setRange(0, 9999)
        self.field_width_spin = QtWidgets.QSpinBox()
        self.field_width_spin.setRange(0, 4000)
        self.field_height_spin = QtWidgets.QSpinBox()
        self.field_height_spin.setRange(0, 4000)
        self._field_size_enabled = False

        lg_form = QtWidgets.QFormLayout()
        lg_form.addRow("LG w0", self.lg_w0_spin)
        lg_form.addRow("LG p(逗号)", self.lg_p_edit)
        lg_form.addRow("LG l(逗号)", self.lg_l_edit)
        lg_widget = QtWidgets.QWidget()
        lg_widget.setLayout(lg_form)

        letter_form = QtWidgets.QFormLayout()
        letter_form.addRow("字母", self.letter_edit)
        letter_widget = QtWidgets.QWidget()
        letter_widget.setLayout(letter_form)

        mnist_form = QtWidgets.QFormLayout()
        mnist_form.addRow("索引", self.dataset_index_spin)
        mnist_widget = QtWidgets.QWidget()
        mnist_widget.setLayout(mnist_form)

        focus_form = QtWidgets.QFormLayout()
        focus_form.addRow("焦距", self.focus_distance_spin)
        focus_form.addRow("光场直径(px)", self.focus_diameter_spin)
        focus_widget = QtWidgets.QWidget()
        focus_widget.setLayout(focus_form)

        self.field_stack = QtWidgets.QStackedWidget()
        self.field_stack.addWidget(lg_widget)
        self.field_stack.addWidget(focus_widget)
        self.field_stack.addWidget(letter_widget)
        self.field_stack.addWidget(mnist_widget)

        gen_layout = QtWidgets.QVBoxLayout()
        gen_layout.addWidget(self.field_mode_combo)
        gen_layout.addWidget(self.field_stack)
        gen_widget = QtWidgets.QWidget()
        gen_widget.setLayout(gen_layout)

        # 堆叠布局切换
        self.load_stack = QtWidgets.QStackedWidget()
        self.load_stack.addWidget(file_widget)
        self.load_stack.addWidget(gen_widget)

        size_layout = QtWidgets.QHBoxLayout()
        size_layout.addWidget(QtWidgets.QLabel("Size:"))
        size_layout.addWidget(self.field_width_spin)
        size_layout.addWidget(QtWidgets.QLabel("x"))
        size_layout.addWidget(self.field_height_spin)
        self.size_widget = QtWidgets.QWidget()
        self.size_widget.setLayout(size_layout)

        # --- Hologram params ---
        self.period_spin = QtWidgets.QSpinBox()
        self.period_spin.setRange(-200, 200)
        self.period_spin.setValue(8)
        self.period_spin.setToolTip("Bolduc grating period (sign controls direction)")

        self.amp_norm_spin = QtWidgets.QDoubleSpinBox()
        self.amp_norm_spin.setRange(0.0, 0.99)
        self.amp_norm_spin.setDecimals(2)
        self.amp_norm_spin.setSingleStep(0.01)
        self.amp_norm_spin.setValue(0.90)
        self.amp_norm_spin.setToolTip("Amplitude normalization max (0..0.99)")

        # --- 3. 播放控制 ---
        self.play_mode_combo = QtWidgets.QComboBox()
        self.play_mode_combo.addItem("单帧模式", userData="single")
        self.play_mode_combo.addItem("连续播放模式", userData="continuous")

        self.interval_spin = QtWidgets.QSpinBox()
        self.interval_spin.setRange(10, 5000)
        self.interval_spin.setValue(500)
        self.interval_spin.setSuffix(" ms")
        self.interval_spin.setToolTip("连续播放时的帧间隔")

        self.prev_button = QtWidgets.QPushButton("上")
        self.next_button = QtWidgets.QPushButton("下")
        self.loop_checkbox = QtWidgets.QCheckBox("循环")
        self.loop_checkbox.setChecked(True)

        self.single_button = QtWidgets.QPushButton("显示单帧")
        self.play_button = QtWidgets.QPushButton("开始播放")
        self.stop_play_button = QtWidgets.QPushButton("停止播放")

        # 播放控制布局
        play_mode_layout = QtWidgets.QHBoxLayout()
        play_mode_layout.addWidget(QtWidgets.QLabel("模式:"))
        play_mode_layout.addWidget(self.play_mode_combo)
        play_mode_layout.addWidget(QtWidgets.QLabel("间隔:"))
        play_mode_layout.addWidget(self.interval_spin)

        play_btn_layout = QtWidgets.QHBoxLayout()
        play_btn_layout.addWidget(self.prev_button)
        play_btn_layout.addWidget(self.next_button)
        play_btn_layout.addWidget(self.loop_checkbox)
        play_btn_layout.addWidget(self.single_button)
        play_btn_layout.addWidget(self.play_button)
        play_btn_layout.addWidget(self.stop_play_button)

        # --- 4. 变换与运行 ---
        self.run_button = QtWidgets.QPushButton("Run SLM1")
        self.stop_button = QtWidgets.QPushButton("Stop SLM1")

        self.slm1_comp_checkbox = QtWidgets.QCheckBox("叠加补偿")
        self.slm1_comp_edit = QtWidgets.QLineEdit()
        self.slm1_comp_button = QtWidgets.QPushButton("...")
        self.slm1_comp_flip_h = QtWidgets.QCheckBox("补偿H翻转")
        self.slm1_comp_flip_v = QtWidgets.QCheckBox("补偿V翻转")
        self.auto_apply_checkbox = QtWidgets.QCheckBox("Auto Apply")

        self.flip_h_checkbox = QtWidgets.QCheckBox("H翻转")
        self.flip_v_checkbox = QtWidgets.QCheckBox("V翻转")

        trans_layout = QtWidgets.QHBoxLayout()
        trans_layout.addWidget(self.flip_h_checkbox)
        trans_layout.addWidget(self.flip_v_checkbox)
        trans_layout.addWidget(self.slm1_comp_checkbox)
        trans_layout.addWidget(self.slm1_comp_flip_h)
        trans_layout.addWidget(self.slm1_comp_flip_v)
        trans_layout.addWidget(self.slm1_comp_edit, 1)
        trans_layout.addWidget(self.slm1_comp_button)

        run_layout = QtWidgets.QHBoxLayout()
        run_layout.addWidget(self.run_button)
        run_layout.addWidget(self.stop_button)

        # --- 主布局组装 ---
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(QtWidgets.QLabel("SLM1 类型"))
        main_layout.addWidget(self.device_combo)
        screen_layout = QtWidgets.QHBoxLayout()
        screen_layout.addWidget(QtWidgets.QLabel("屏幕ID"))
        screen_layout.addWidget(self.screen_combo, 1)
        main_layout.addLayout(screen_layout)

        heds_layout = QtWidgets.QHBoxLayout()
        heds_layout.addWidget(QtWidgets.QLabel("Holoeye 设备"))
        heds_layout.addWidget(self.heds_selector_combo, 1)
        heds_layout.addWidget(self.heds_refresh_button)
        main_layout.addLayout(heds_layout)

        source_header = QtWidgets.QHBoxLayout()
        source_header.addWidget(QtWidgets.QLabel("加载设置"))
        source_header.addWidget(self.load_mode_combo)
        source_header.addWidget(self.auto_apply_checkbox)
        main_layout.addLayout(source_header)
        main_layout.addWidget(self.load_stack)
        main_layout.addWidget(self.size_widget)

        hologram_layout = QtWidgets.QHBoxLayout()
        hologram_layout.addWidget(QtWidgets.QLabel("光栅周期(±)"))
        hologram_layout.addWidget(self.period_spin)
        hologram_layout.addWidget(QtWidgets.QLabel("幅度上限"))
        hologram_layout.addWidget(self.amp_norm_spin)
        main_layout.addLayout(hologram_layout)

        main_layout.addLayout(play_mode_layout)
        main_layout.addLayout(play_btn_layout)

        main_layout.addLayout(trans_layout)
        main_layout.addLayout(run_layout)

        self.setLayout(main_layout)

        # --- 信号连接 ---
        self.image_button.clicked.connect(self._pick_image)
        self.folder_button.clicked.connect(self._pick_folder)
        self.slm1_comp_button.clicked.connect(self._pick_slm1_comp)
        self.load_mode_combo.currentIndexChanged.connect(self._update_load_mode)
        self.input_type_combo.currentIndexChanged.connect(self._update_size_ui)
        self.play_mode_combo.currentIndexChanged.connect(self._update_play_mode)
        self.field_mode_combo.currentIndexChanged.connect(self._update_field_mode)
        self.file_source_type_combo.currentIndexChanged.connect(self._update_file_source_ui)
        self.device_combo.currentIndexChanged.connect(self._update_device_ui)

        self._connect_auto_apply_signals()

        self._update_load_mode()
        self._update_play_mode()
        self._update_field_mode()
        self._update_file_source_ui()
        self._update_size_ui()
        self._update_device_ui()

    def _populate_screen_combo(self, combo: QtWidgets.QComboBox) -> None:
        combo.clear()
        screens = QtGui.QGuiApplication.screens()
        count = max(len(screens), 3)
        for idx in range(1, count + 1):
            combo.addItem(str(idx), userData=idx)

    def _update_device_ui(self) -> None:
        is_upo = (self.device_combo.currentData() == "upo")
        self.screen_combo.setEnabled(is_upo)
        self.heds_selector_combo.setEnabled(not is_upo)
        self.heds_refresh_button.setEnabled(not is_upo)

    def selected_screen_num(self) -> int:
        data = self.screen_combo.currentData()
        return int(data) if data is not None else 1

    def set_heds_options(self, items: list[str]) -> None:
        self.heds_selector_combo.clear()
        if not items:
            self.heds_selector_combo.addItem("auto")
            return
        for item in items:
            self.heds_selector_combo.addItem(item)

    def heds_selector(self) -> str:
        data = self.heds_selector_combo.currentText().strip()
        return "" if data == "auto" else data

    def _connect_auto_apply_signals(self):
        self.load_mode_combo.currentIndexChanged.connect(self.param_changed)
        self.file_source_type_combo.currentIndexChanged.connect(self.param_changed)
        self.input_type_combo.currentIndexChanged.connect(self.param_changed)
        self.image_edit.textChanged.connect(self.param_changed)
        self.field_mode_combo.currentIndexChanged.connect(self.param_changed)
        self.lg_w0_spin.valueChanged.connect(self.param_changed)
        self.lg_p_edit.textChanged.connect(self.param_changed)
        self.lg_l_edit.textChanged.connect(self.param_changed)
        self.focus_distance_spin.valueChanged.connect(self.param_changed)
        self.focus_diameter_spin.valueChanged.connect(self.param_changed)
        self.letter_edit.textChanged.connect(self.param_changed)
        self.dataset_index_spin.valueChanged.connect(self.param_changed)
        self.field_width_spin.valueChanged.connect(self.param_changed)
        self.field_height_spin.valueChanged.connect(self.param_changed)
        self.period_spin.valueChanged.connect(self.param_changed)
        self.amp_norm_spin.valueChanged.connect(self.param_changed)
        self.flip_h_checkbox.stateChanged.connect(self.param_changed)
        self.flip_v_checkbox.stateChanged.connect(self.param_changed)
        self.slm1_comp_checkbox.stateChanged.connect(self.param_changed)
        self.slm1_comp_flip_h.stateChanged.connect(self.param_changed)
        self.slm1_comp_flip_v.stateChanged.connect(self.param_changed)
        self.slm1_comp_edit.textChanged.connect(self.param_changed)

    def _pick_image(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择图像", "", "Images (*.png *.bmp *.jpg *.tif)")
        if path:
            self.image_edit.setText(path)

    def _pick_folder(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择文件夹")
        if path:
            self.folder_edit.setText(path)

    def _pick_slm1_comp(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "补偿文件", "", "Images (*.png *.bmp *.tif)")
        if path:
            self.slm1_comp_edit.setText(path)

    def _update_load_mode(self) -> None:
        index = 0 if self.load_mode_combo.currentData() == "file" else 1
        self.load_stack.setCurrentIndex(index)
        self._update_size_ui()

    def _update_file_source_ui(self) -> None:
        is_folder = (self.file_source_type_combo.currentData() == "folder")
        self.label_single.setVisible(not is_folder)
        self.image_edit.setVisible(not is_folder)
        self.image_button.setVisible(not is_folder)
        self.label_folder.setVisible(is_folder)
        self.folder_edit.setVisible(is_folder)
        self.folder_button.setVisible(is_folder)

    def _update_play_mode(self) -> None:
        is_continuous = (self.play_mode_combo.currentData() == "continuous")
        self.interval_spin.setEnabled(is_continuous)
        self.play_button.setEnabled(is_continuous)
        self.stop_play_button.setEnabled(is_continuous)
        self.single_button.setEnabled(True)

    def _update_field_mode(self) -> None:
        mode = self.field_mode_combo.currentData() or "lg"
        if mode == "lg":
            index = 0;
            enable_size = False
        elif mode == "focus":
            index = 1;
            enable_size = False
        elif mode == "letter":
            index = 2;
            enable_size = True
        elif mode in {"mnist", "fashion_mnist"}:
            index = 3;
            enable_size = True
        else:
            index = 0;
            enable_size = False
        self._field_size_enabled = enable_size
        self.field_stack.setCurrentIndex(index)
        self._update_size_ui()

    def _update_size_ui(self) -> None:
        load_mode = self.load_mode_combo.currentData()
        input_type = self.input_type_combo.currentData()
        if load_mode == "file":
            visible = (input_type == "field")
        else:
            visible = self._field_size_enabled
        self.size_widget.setVisible(visible)
        self.field_width_spin.setEnabled(visible)
        self.field_height_spin.setEnabled(visible)


class SLM2Panel(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__("SLM2 四层相位")
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("加载", userData="load")
        self.mode_combo.addItem("标定", userData="calib")
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItem("Holoeye (1920x1080)", userData="holoeye")
        self.device_combo.addItem("UPO (1920x1200)", userData="upo")
        self.screen_combo = QtWidgets.QComboBox()
        self._populate_screen_combo(self.screen_combo)
        self.heds_selector_combo = QtWidgets.QComboBox()
        self.heds_refresh_button = QtWidgets.QToolButton()
        self.heds_refresh_button.setText("刷新")
        self.layer_widgets: List[LayerWidget] = [LayerWidget(i) for i in range(4)]
        self.apply_button = QtWidgets.QPushButton("合成并加载 SLM2")
        self.auto_apply_checkbox = QtWidgets.QCheckBox("Auto Apply")
        self.auto_apply_checkbox.setChecked(False)
        self.run_button = QtWidgets.QPushButton("Run SLM2")
        self.stop_button = QtWidgets.QPushButton("Stop SLM2")

        self.slm2_comp_checkbox = QtWidgets.QCheckBox("叠加补偿")
        self.slm2_comp_edit = QtWidgets.QLineEdit()
        self.slm2_comp_button = QtWidgets.QPushButton("...")
        self.slm2_comp_flip_h = QtWidgets.QCheckBox("补偿H翻转")
        self.slm2_comp_flip_v = QtWidgets.QCheckBox("补偿V翻转")

        self.global_flip_h = QtWidgets.QCheckBox("全局H翻转")
        self.global_flip_v = QtWidgets.QCheckBox("全局V翻转")

        comp_layout = QtWidgets.QHBoxLayout()
        comp_layout.addWidget(self.slm2_comp_checkbox)
        comp_layout.addWidget(self.slm2_comp_edit, 1)
        comp_layout.addWidget(self.slm2_comp_button)

        flip_layout = QtWidgets.QHBoxLayout()
        flip_layout.addWidget(self.global_flip_h)
        flip_layout.addWidget(self.global_flip_v)
        flip_layout.addWidget(self.slm2_comp_flip_h)
        flip_layout.addWidget(self.slm2_comp_flip_v)
        flip_layout.addStretch(1)

        self.calib_layer_center_combo = QtWidgets.QComboBox()
        self.calib_layer_distance_combo = QtWidgets.QComboBox()
        for i in range(1, 5):
            self.calib_layer_center_combo.addItem(f"Layer {i}", userData=i)
            self.calib_layer_distance_combo.addItem(f"Layer {i}", userData=i)
        self.calib_dist_cx = QtWidgets.QSpinBox()
        self.calib_dist_cx.setRange(0, 4000)
        self.calib_dist_cy = QtWidgets.QSpinBox()
        self.calib_dist_cy.setRange(0, 4000)

        self.calib_center_cx = QtWidgets.QSpinBox()
        self.calib_center_cx.setRange(0, 4000)
        self.calib_center_cy = QtWidgets.QSpinBox()
        self.calib_center_cy.setRange(0, 4000)
        self.calib_center_range_x = QtWidgets.QSpinBox()
        self.calib_center_range_x.setRange(0, 2000)
        self.calib_center_range_x.setValue(200)
        self.calib_center_range_y = QtWidgets.QSpinBox()
        self.calib_center_range_y.setRange(0, 2000)
        self.calib_center_range_y.setValue(200)
        self.calib_center_step = QtWidgets.QSpinBox()
        self.calib_center_step.setRange(1, 500)
        self.calib_center_step.setValue(20)
        self.calib_window_size = QtWidgets.QSpinBox()
        self.calib_window_size.setRange(10, 2000)
        self.calib_window_size.setValue(200)
        self.calib_window_shape = QtWidgets.QComboBox()
        self.calib_window_shape.addItem("方窗", userData="square")
        self.calib_window_shape.addItem("圆窗", userData="circle")
        self.calib_grating_period = QtWidgets.QSpinBox()
        self.calib_grating_period.setRange(-500, 500)
        self.calib_grating_period.setValue(8)
        self.calib_center_start = QtWidgets.QPushButton("开始中心标定")
        self.calib_center_stop = QtWidgets.QPushButton("停止中心标定")
        self.calib_comp_checkbox = QtWidgets.QCheckBox("叠加补偿")
        self.calib_comp_checkbox.setChecked(True)
        self.calib_center_pause = QtWidgets.QPushButton("暂停")
        self.calib_center_resume = QtWidgets.QPushButton("继续")
        self.calib_center_status = QtWidgets.QLabel("当前窗口中心: -")
        self.calib_roi_x1 = QtWidgets.QSpinBox()
        self.calib_roi_x1.setRange(0, 4000)
        self.calib_roi_x2 = QtWidgets.QSpinBox()
        self.calib_roi_x2.setRange(0, 4000)
        self.calib_roi_y1 = QtWidgets.QSpinBox()
        self.calib_roi_y1.setRange(0, 4000)
        self.calib_roi_y2 = QtWidgets.QSpinBox()
        self.calib_roi_y2.setRange(0, 4000)

        self.calib_dist_init = QtWidgets.QDoubleSpinBox()
        self.calib_dist_init.setRange(-2000.0, 2000.0)
        self.calib_dist_init.setValue(200.0)
        self.calib_dist_range = QtWidgets.QDoubleSpinBox()
        self.calib_dist_range.setRange(1.0, 2000.0)
        self.calib_dist_range.setValue(100.0)
        self.calib_dist_step = QtWidgets.QDoubleSpinBox()
        self.calib_dist_step.setRange(1.0, 500.0)
        self.calib_dist_step.setValue(10.0)
        self.calib_dist_start = QtWidgets.QPushButton("开始距离标定")
        self.calib_dist_stop = QtWidgets.QPushButton("停止距离标定")
        self.calib_dist_pause = QtWidgets.QPushButton("暂停")
        self.calib_dist_resume = QtWidgets.QPushButton("继续")
        self.calib_dist_status = QtWidgets.QLabel("当前距离: -")
        self.calib_dist_comp_checkbox = QtWidgets.QCheckBox("叠加补偿")
        self.calib_dist_comp_checkbox.setChecked(True)
        self.calib_dist_roi_x1 = QtWidgets.QSpinBox()
        self.calib_dist_roi_x1.setRange(0, 4000)
        self.calib_dist_roi_x2 = QtWidgets.QSpinBox()
        self.calib_dist_roi_x2.setRange(0, 4000)
        self.calib_dist_roi_y1 = QtWidgets.QSpinBox()
        self.calib_dist_roi_y1.setRange(0, 4000)
        self.calib_dist_roi_y2 = QtWidgets.QSpinBox()
        self.calib_dist_roi_y2.setRange(0, 4000)

        self.calib_log_list = QtWidgets.QListWidget()

        load_layout = QtWidgets.QVBoxLayout()
        load_layout.addWidget(QtWidgets.QLabel("SLM2 类型"))
        load_layout.addWidget(self.device_combo)
        screen_layout = QtWidgets.QHBoxLayout()
        screen_layout.addWidget(QtWidgets.QLabel("屏幕ID"))
        screen_layout.addWidget(self.screen_combo, 1)
        load_layout.addLayout(screen_layout)

        heds_layout = QtWidgets.QHBoxLayout()
        heds_layout.addWidget(QtWidgets.QLabel("Holoeye 设备"))
        heds_layout.addWidget(self.heds_selector_combo, 1)
        heds_layout.addWidget(self.heds_refresh_button)
        load_layout.addLayout(heds_layout)
        for widget in self.layer_widgets:
            load_layout.addWidget(widget)
        load_layout.addLayout(comp_layout)
        load_layout.addLayout(flip_layout)
        load_layout.addWidget(self.auto_apply_checkbox)
        load_layout.addWidget(self.apply_button)

        run_layout = QtWidgets.QHBoxLayout()
        run_layout.addWidget(self.run_button)
        run_layout.addWidget(self.stop_button)
        load_layout.addLayout(run_layout)
        load_widget = QtWidgets.QWidget()
        load_widget.setLayout(load_layout)

        center_grid = QtWidgets.QGridLayout()
        center_grid.addWidget(QtWidgets.QLabel("标定层"), 0, 0)
        center_grid.addWidget(self.calib_layer_center_combo, 0, 1)
        center_grid.addWidget(QtWidgets.QLabel("初始中心 X"), 1, 0)
        center_grid.addWidget(self.calib_center_cx, 1, 1)
        center_grid.addWidget(QtWidgets.QLabel("初始中心 Y"), 2, 0)
        center_grid.addWidget(self.calib_center_cy, 2, 1)
        center_grid.addWidget(QtWidgets.QLabel("扫描范围 X"), 0, 2)
        center_grid.addWidget(self.calib_center_range_x, 0, 3)
        center_grid.addWidget(QtWidgets.QLabel("扫描范围 Y"), 1, 2)
        center_grid.addWidget(self.calib_center_range_y, 1, 3)
        center_grid.addWidget(QtWidgets.QLabel("扫描步长"), 2, 2)
        center_grid.addWidget(self.calib_center_step, 2, 3)
        center_grid.addWidget(QtWidgets.QLabel("窗口大小"), 3, 0)
        center_grid.addWidget(self.calib_window_size, 3, 1)
        center_grid.addWidget(QtWidgets.QLabel("窗口形状"), 3, 2)
        center_grid.addWidget(self.calib_window_shape, 3, 3)
        center_grid.addWidget(QtWidgets.QLabel("光栅周期"), 4, 0)
        center_grid.addWidget(self.calib_grating_period, 4, 1)
        center_grid.addWidget(QtWidgets.QLabel("ROI x1"), 4, 2)
        center_grid.addWidget(self.calib_roi_x1, 4, 3)
        center_grid.addWidget(QtWidgets.QLabel("ROI x2"), 5, 0)
        center_grid.addWidget(self.calib_roi_x2, 5, 1)
        center_grid.addWidget(QtWidgets.QLabel("ROI y1"), 5, 2)
        center_grid.addWidget(self.calib_roi_y1, 5, 3)
        center_grid.addWidget(QtWidgets.QLabel("ROI y2"), 6, 0)
        center_grid.addWidget(self.calib_roi_y2, 6, 1)
        center_btn_layout = QtWidgets.QHBoxLayout()
        center_btn_layout.addWidget(self.calib_center_start)
        center_btn_layout.addWidget(self.calib_center_stop)
        center_btn_layout.addWidget(self.calib_center_pause)
        center_btn_layout.addWidget(self.calib_center_resume)
        center_btn_layout.addWidget(self.calib_comp_checkbox)
        center_group = QtWidgets.QGroupBox("中心标定")
        center_layout = QtWidgets.QVBoxLayout()
        center_layout.addLayout(center_grid)
        center_layout.addWidget(self.calib_center_status)
        center_layout.addLayout(center_btn_layout)
        center_group.setLayout(center_layout)

        dist_form = QtWidgets.QFormLayout()
        dist_form.addRow("标定层", self.calib_layer_distance_combo)
        dist_form.addRow("中心 X", self.calib_dist_cx)
        dist_form.addRow("中心 Y", self.calib_dist_cy)
        dist_form.addRow("初始距离(mm)", self.calib_dist_init)
        dist_form.addRow("扫描范围(mm)", self.calib_dist_range)
        dist_form.addRow("扫描步长(mm)", self.calib_dist_step)
        dist_btn_layout = QtWidgets.QHBoxLayout()
        dist_btn_layout.addWidget(self.calib_dist_start)
        dist_btn_layout.addWidget(self.calib_dist_stop)
        dist_btn_layout.addWidget(self.calib_dist_pause)
        dist_btn_layout.addWidget(self.calib_dist_resume)
        dist_group = QtWidgets.QGroupBox("距离标定")
        dist_grid = QtWidgets.QGridLayout()
        dist_grid.addWidget(QtWidgets.QLabel("标定层"), 0, 0)
        dist_grid.addWidget(self.calib_layer_distance_combo, 0, 1)
        dist_grid.addWidget(QtWidgets.QLabel("中心 X"), 1, 0)
        dist_grid.addWidget(self.calib_dist_cx, 1, 1)
        dist_grid.addWidget(QtWidgets.QLabel("中心 Y"), 2, 0)
        dist_grid.addWidget(self.calib_dist_cy, 2, 1)
        dist_grid.addWidget(QtWidgets.QLabel("初始距离(mm)"), 0, 2)
        dist_grid.addWidget(self.calib_dist_init, 0, 3)
        dist_grid.addWidget(QtWidgets.QLabel("扫描范围(mm)"), 1, 2)
        dist_grid.addWidget(self.calib_dist_range, 1, 3)
        dist_grid.addWidget(QtWidgets.QLabel("扫描步长(mm)"), 2, 2)
        dist_grid.addWidget(self.calib_dist_step, 2, 3)
        dist_grid.addWidget(QtWidgets.QLabel("ROI x1"), 3, 0)
        dist_grid.addWidget(self.calib_dist_roi_x1, 3, 1)
        dist_grid.addWidget(QtWidgets.QLabel("ROI x2"), 3, 2)
        dist_grid.addWidget(self.calib_dist_roi_x2, 3, 3)
        dist_grid.addWidget(QtWidgets.QLabel("ROI y1"), 4, 0)
        dist_grid.addWidget(self.calib_dist_roi_y1, 4, 1)
        dist_grid.addWidget(QtWidgets.QLabel("ROI y2"), 4, 2)
        dist_grid.addWidget(self.calib_dist_roi_y2, 4, 3)
        dist_grid.addWidget(self.calib_dist_status, 5, 0, 1, 4)

        dist_layout = QtWidgets.QVBoxLayout()
        dist_layout.addLayout(dist_grid)
        dist_layout.addWidget(self.calib_dist_comp_checkbox)
        dist_layout.addLayout(dist_btn_layout)
        dist_group.setLayout(dist_layout)

        log_group = QtWidgets.QGroupBox("标定日志")
        log_layout = QtWidgets.QVBoxLayout()
        log_layout.addWidget(self.calib_log_list)
        log_group.setLayout(log_layout)

        calib_layout = QtWidgets.QVBoxLayout()
        calib_layout.addWidget(center_group)
        calib_layout.addWidget(dist_group)
        calib_layout.addWidget(log_group)
        calib_widget = QtWidgets.QWidget()
        calib_widget.setLayout(calib_layout)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(load_widget)
        self.stack.addWidget(calib_widget)

        layout = QtWidgets.QVBoxLayout()
        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.addWidget(QtWidgets.QLabel("模式"))
        mode_layout.addWidget(self.mode_combo, 1)
        layout.addLayout(mode_layout)
        layout.addWidget(self.stack)
        self.setLayout(layout)

        self.slm2_comp_button.clicked.connect(self._pick_slm2_comp)
        self.device_combo.currentIndexChanged.connect(self._update_device_ui)
        self.mode_combo.currentIndexChanged.connect(self._update_mode_ui)
        self._update_device_ui()
        self._update_mode_ui()

    def _pick_slm2_comp(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "补偿文件", "", "Images (*.png *.bmp *.tif)")
        if path:
            self.slm2_comp_edit.setText(path)

    def _populate_screen_combo(self, combo: QtWidgets.QComboBox) -> None:
        combo.clear()
        screens = QtGui.QGuiApplication.screens()
        count = max(len(screens), 3)
        for idx in range(1, count + 1):
            combo.addItem(str(idx), userData=idx)

    def _update_device_ui(self) -> None:
        is_upo = (self.device_combo.currentData() == "upo")
        self.screen_combo.setEnabled(is_upo)
        self.heds_selector_combo.setEnabled(not is_upo)
        self.heds_refresh_button.setEnabled(not is_upo)

    def _update_mode_ui(self) -> None:
        index = 0 if (self.mode_combo.currentData() == "load") else 1
        self.stack.setCurrentIndex(index)

    def selected_screen_num(self) -> int:
        data = self.screen_combo.currentData()
        return int(data) if data is not None else 1

    def set_heds_options(self, items: list[str]) -> None:
        self.heds_selector_combo.clear()
        if not items:
            self.heds_selector_combo.addItem("auto")
            return
        for item in items:
            self.heds_selector_combo.addItem(item)

    def heds_selector(self) -> str:
        data = self.heds_selector_combo.currentText().strip()
        return "" if data == "auto" else data

    def get_layers(self) -> List[LayerConfig]:
        return [widget.get_config() for widget in self.layer_widgets]


class CameraControlPanel(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__("相机控制", parent)
        self.run_button = QtWidgets.QPushButton("Run")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.exposure_spin = QtWidgets.QDoubleSpinBox()
        self.exposure_spin.setRange(10.0, 200000.0)
        self.exposure_spin.setValue(20000.0)
        self.apply_exposure_button = QtWidgets.QPushButton("Set")
        self.auto_exposure_checkbox = QtWidgets.QCheckBox("自动曝光")

        self.save_roi_checkbox = QtWidgets.QCheckBox("仅保存ROI")
        self.record_interval_spin = QtWidgets.QSpinBox()
        self.record_interval_spin.setRange(10, 5000)
        self.record_interval_spin.setValue(200)
        self.record_interval_spin.setSuffix(" ms")

        self.auto_reduce_checkbox = QtWidgets.QCheckBox("过曝自动降曝光")

        self.flip_h = QtWidgets.QCheckBox("H翻转")
        self.flip_v = QtWidgets.QCheckBox("V翻转")

        run_layout = QtWidgets.QHBoxLayout()
        run_layout.addWidget(self.run_button)
        run_layout.addWidget(self.stop_button)

        exposure_layout = QtWidgets.QHBoxLayout()
        exposure_layout.addWidget(QtWidgets.QLabel("Exp(us)"))
        exposure_layout.addWidget(self.exposure_spin, 1)
        exposure_layout.addWidget(self.apply_exposure_button)

        flip_layout = QtWidgets.QHBoxLayout()
        flip_layout.addWidget(self.flip_h)
        flip_layout.addWidget(self.flip_v)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(run_layout)
        layout.addWidget(self.auto_exposure_checkbox)
        layout.addLayout(exposure_layout)
        layout.addLayout(flip_layout)
        layout.addWidget(self.save_roi_checkbox)
        layout.addWidget(self.auto_reduce_checkbox)
        self.setLayout(layout)


class SLMPreviewPanel(QtWidgets.QGroupBox):
    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.label = ZoomLabel()
        self.label.setMinimumSize(200, 140)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

    def update_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        self.label.set_base_pixmap(pixmap)

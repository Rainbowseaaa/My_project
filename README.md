# SLM & Camera Control

该仓库包含 SLM1(UPO) / SLM2(Holoeye) SDK、相机控制代码，以及示例程序。

## 新增 UI 程序

新增了基于 PyQt6 的桌面 UI，用于完成：输入图像 → SLM1 全息图 → SLM2 四层相位 → 相机实时显示与测量的完整流程。

### 安装依赖

```bash
pip install -r requirements.txt
```

> 注意：当前依赖使用 `numpy<2`，避免部分编译扩展（如 matplotlib 及其依赖）与 NumPy 2.x 产生 ABI 不兼容。

### 运行 UI

```bash
python ui_app.py
```

使用模拟模式（无硬件）运行：

```bash
python ui_app.py --mock
```

### 校准可视化

如需在 `calibrate_insitu.py` 运行时显示校准层数、中心位置、相机画面与热图，可使用：

```bash
python calibrate_insitu.py --visualize
```

### 配置文件

默认配置文件：`config_ui.yaml`，包含以下内容：

- SLM1/SLM2 分辨率、窗口尺寸、补偿图路径（可选 `slm2.sdk_path` 指向 HEDS SDK 目录）
- 四层相位默认中心与偏移
- 相机参数、播放间隔等

你可以根据实际硬件修改对应路径与参数。

# 项目脚本运行命令说明

本文档列出 `src` 目录下各个 Python 脚本的独立运行命令及其执行后果。

## 环境准备

```bash
# 激活虚拟环境（如果使用）
source /home/eureka/tilt_checker/.venv/bin/activate

# 进入项目目录
cd /home/eureka/tilt_checker
```

---

## 1. `run_tilt_check.py` - 倾斜检测主脚本

### 功能
检测圆点标定板，计算相机姿态和倾斜角，生成可视化结果图。

### 命令示例

#### 1.1 处理单张图片（使用 YAML 内参）
```bash
python src/run_tilt_check.py --image data/board.png
```

#### 1.2 处理单张图片（指定行列数）
```bash
python src/run_tilt_check.py --image data/board.png --rows 15 --cols 15
```

#### 1.3 处理单张图片（指定相机内参 YAML）
```bash
python src/run_tilt_check.py --image data/board.png --camera-yaml config/camera_info.yaml
```

#### 1.4 自动搜索网格尺寸
```bash
python src/run_tilt_check.py --image data/board.png --auto
```

#### 1.5 批量处理目录
```bash
python src/run_tilt_check.py --dir data --rows 15 --cols 15
```

#### 1.6 使用原版检测算法
```bash
python src/run_tilt_check.py --image data/board.png --use-original
```

### 执行后果
- **终端输出**：
  - 检测状态（`ok` 值：是否成功匹配网格）
  - 检测到的 blob 数量
  - 成功匹配的网格点数
  - Roll/Pitch/Yaw 角度（板子相对于相机）
  - 相机倾斜角（假设板子水平，相机相对于水平面）
  - 歪斜判断结果
- **生成文件**：
  - `outputs/{图片名}_result.png`：网格检测结果图（带坐标轴、投影等，不含绿色 blob 点）
  - `outputs/{图片名}_result_with_blobs.png`：带绿色 blob 点的结果图（用于调试检测参数）

### 参数说明
- `--image`：输入图像路径
- `--dir`：输入目录（批量处理）
- `--rows`：圆点行数（默认 15）
- `--cols`：圆点列数（默认 15）
- `--asymmetric`：使用非对称圆点网格
- `--spacing`：圆点间距（mm，默认 10.0）
- `--auto`：自动搜索 rows/cols/对称性
- `--save`：结果图保存路径（默认 `outputs/result.png`）
- `--use-original`：使用原版检测算法
- `--camera-yaml`：相机内参 YAML 文件路径（默认 `config/camera_info.yaml`）

---

## 2. `camera_rectifier.py` - ROS2 相机内参提取脚本

### 功能
从 ROS2 `CameraInfo` 话题读取相机内参，保存为 OpenCV 兼容的 YAML 文件。可选：实时去畸变并保存图像。

### 前置条件
```bash
source /opt/ros/humble/setup.bash
```

### 命令示例

#### 2.1 只提取一次内参并保存 YAML（推荐）
```bash
python src/camera_rectifier.py --camera_info_topic /camera/camera_info --output config/camera_info.yaml
```

#### 2.2 提取内参并保存去畸变图像
```bash
python src/camera_rectifier.py --camera_info_topic /camera/camera_info --output config/camera_info.yaml --save_image
```

#### 2.3 指定图像话题（用于去畸变）
```bash
python src/camera_rectifier.py --camera_info_topic /camera/camera_info --image_topic /camera/image_raw --output config/camera_info.yaml --save_image
```

### 执行后果
- **终端输出**：
  - ROS2 节点启动信息
  - 等待 `CameraInfo` 消息的提示
  - 成功保存 YAML 文件的确认信息
  - 如果启用 `--save_image`，会显示保存的去畸变图像路径
- **生成文件**：
  - `config/camera_info.yaml`：相机内参文件（包含 K 矩阵、畸变系数 D、图像尺寸）
  - 如果启用 `--save_image`：`outputs/rectified/` 目录下的去畸变图像

### 参数说明
- `--camera_info_topic`：CameraInfo 话题名称（默认 `/camera/color/camera_info`）
- `--image_topic`：图像话题名称（如果启用 `--save_image`）
- `--output`：输出 YAML 文件路径（默认 `config/camera_info.yaml`）
- `--save_image`：是否保存去畸变图像（需要同时订阅图像话题）

### 注意事项
- 脚本会持续运行，直到接收到 `CameraInfo` 消息并保存 YAML 后退出（如果只提取一次）
- 如果启用 `--save_image`，脚本会持续订阅图像并保存去畸变结果，按 `Ctrl+C` 退出

---

## 3. `calibration_and_reprojection.py` - 标定和重投影误差分析

### 功能
1. 使用 YAML 内参对 `data` 目录中的所有图像进行畸变矫正
2. 使用矫正后的图像进行相机标定
3. 计算所有图像的重投影误差
4. 绘制重投影误差分布图（包括零中心残差散点图）

### 命令示例

#### 3.1 使用默认参数
```bash
python src/calibration_and_reprojection.py
```

#### 3.2 指定数据目录和输出目录
```bash
python src/calibration_and_reprojection.py --data-dir data --undistorted-dir outputs/undistorted
```

#### 3.3 指定相机内参 YAML
```bash
python src/calibration_and_reprojection.py --camera-yaml config/camera_info.yaml
```

#### 3.4 指定网格尺寸
```bash
python src/calibration_and_reprojection.py --rows 15 --cols 15
```

#### 3.5 指定圆点间距
```bash
python src/calibration_and_reprojection.py --spacing 10.0
```

#### 3.6 指定输出路径
```bash
python src/calibration_and_reprojection.py --output outputs/my_errors.png
```

#### 3.7 完整参数示例
```bash
python src/calibration_and_reprojection.py --data-dir data --undistorted-dir outputs/undistorted --camera-yaml config/camera_info.yaml --rows 15 --cols 15 --spacing 10.0 --output outputs/reprojection_errors.png
```

### 执行后果
- **终端输出**：
  - 步骤 1：畸变矫正进度（每张图像的处理状态）
  - 步骤 2：相机标定进度（每张图像的检测状态）
  - 步骤 3：重投影误差计算（每张图像的平均误差）
  - 步骤 4：绘图完成确认
  - 步骤 5：零中心残差散点图完成确认
  - 统计信息（图像数量、总点数、平均误差、误差范围等）
- **生成文件**：
  - `outputs/undistorted/`（或指定目录）：所有矫正后的图像（`{原文件名}_undistorted.png`）
  - `outputs/reprojection_errors.png`：误差分布图（包含两张子图：每张图像的平均误差散点图、所有点的误差直方图）
  - `outputs/reprojection_residual_scatter.png`：零中心残差散点图（dx, dy 分布，颜色表示误差大小）
  - `outputs/reprojection_residuals.csv`：所有点的残差数据（dx, dy, magnitude）

### 参数说明
- `--data-dir`：输入图像目录（默认 `data`）
- `--undistorted-dir`：矫正后的图像保存目录（默认 `data_undistorted`）
- `--camera-yaml`：相机内参 YAML 文件路径（默认 `config/camera_info.yaml`）
- `--rows`：圆点行数（默认 15）
- `--cols`：圆点列数（默认 15）
- `--spacing`：圆点间距（mm，默认 10.0）
- `--output`：误差分布图保存路径（默认 `outputs/reprojection_errors.png`）

---

## 4. `undistort_demo.py` - 畸变矫正演示工具

### 功能
对单张图像进行畸变矫正，并对比矫正前后的效果。可选：显示内参和投影矩阵的详细说明。

### 命令示例

#### 4.1 矫正单张图像
```bash
python src/undistort_demo.py --image data/board.png
```

#### 4.2 指定相机内参 YAML
```bash
python src/undistort_demo.py --image data/board.png --camera-yaml config/camera_info.yaml
```

#### 4.3 指定保存目录
```bash
python src/undistort_demo.py --image data/board.png --save-dir outputs
```

#### 4.4 显示内参和投影矩阵说明
```bash
python src/undistort_demo.py --explain
```

#### 4.5 矫正图像并显示说明
```bash
python src/undistort_demo.py --image data/board.png --explain
```

### 执行后果
- **终端输出**：
  - 加载的内参信息（K 矩阵、畸变系数 D、图像尺寸）
  - 如果启用 `--explain`：详细的内参和投影矩阵说明
  - 矫正完成确认
- **生成文件**：
  - `outputs/`（或指定目录）：矫正前后的对比图像

### 参数说明
- `--image`：输入图像路径
- `--camera-yaml`：相机内参 YAML 文件路径（默认 `config/camera_info.yaml`）
- `--save-dir`：保存目录（默认 `outputs`）
- `--explain`：显示内参和投影矩阵的详细说明

---

## 5. 其他脚本说明

### `adjust_blob_params.py`
这是一个工具模块，提供不同严格程度的 blob 检测器参数配置，不是独立运行的脚本。如需调整 blob 检测参数，请修改 `src/utils.py` 中的 `BLOB_DETECTOR_PARAMS` 字典。

### `detect_grid_improved.py` 和 `detect_grid.py`
这些是检测模块，被 `run_tilt_check.py` 调用，通常不直接运行。

### `estimate_tilt.py` 和 `utils.py`
这些是工具模块，被其他脚本调用，通常不直接运行。

---

## 快速参考

### 最常用的命令组合

1. **单张图片倾斜检测**：
   ```bash
   python src/run_tilt_check.py --image data/board.png
   ```

2. **从 ROS2 提取相机内参**：
   ```bash
   source /opt/ros/humble/setup.bash
   python src/camera_rectifier.py --camera_info_topic /camera/camera_info --output config/camera_info.yaml
   ```

3. **批量标定和误差分析**：
   ```bash
   python src/calibration_and_reprojection.py
   ```

4. **单张图片去畸变演示**：
   ```bash
   python src/undistort_demo.py --image data/board.png
   ```

---

## 注意事项

1. **路径问题**：所有命令应在项目根目录（`/home/eureka/tilt_checker`）执行，或使用绝对路径。

2. **依赖问题**：确保已安装所有依赖（`pip install -r requirements.txt`）。

3. **ROS2 环境**：只有 `camera_rectifier.py` 需要 ROS2 环境，其他脚本不需要。

4. **默认值**：大多数参数都有默认值，可以根据需要省略。

5. **输出目录**：脚本会自动创建输出目录（如 `outputs/`），无需手动创建。


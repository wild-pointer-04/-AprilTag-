# 相机内参功能实现说明

## 概述

本项目实现了从 ROS2 CameraInfo 话题提取相机内参，并在倾斜检测中使用真实内参进行更准确的位姿估计。

## 实现架构

### 1. 内参提取工具 (`src/camera_rectifier.py`)

**功能**: 从 ROS2 CameraInfo 话题订阅相机内参，并保存为 OpenCV 兼容的 YAML 格式。

**主要类**: `CameraInfoExtractor`

**关键方法**:
- `camera_info_callback()`: 处理 CameraInfo 消息，提取内参
- `save_camera_info_yaml()`: 保存内参为 YAML 格式

**使用方式**:
```bash
# 确保 ROS2 环境已激活
source /opt/ros/humble/setup.bash  # 替换 humble 为你的发行版名称

# 直接运行 Python 脚本
python src/camera_rectifier.py --camera_info_topic /camera/color/camera_info --output config/camera_info.yaml

# 作为 ROS2 节点运行
ros2 run tilt_checker camera_rectifier --ros-args -p camera_info_topic:=/camera/color/camera_info
```

### 2. 内参加载模块 (`src/utils.py`)

**功能**: 从 YAML 文件加载相机内参，提供自动回退机制。

**主要函数**:
- `load_camera_intrinsics(yaml_path)`: 从 YAML 文件加载内参（失败返回 None）
- `get_camera_intrinsics(h, w, yaml_path, f_scale)`: 获取内参（优先从 YAML，失败则使用默认值）
- `default_intrinsics(h, w, f_scale)`: 生成默认近似内参

**特性**:
- 自动路径查找（当前目录、项目根目录）
- 图像尺寸验证
- 错误处理和日志输出
- 自动回退到默认内参

### 3. 位姿估计模块 (`src/estimate_tilt.py`)

**功能**: 使用真实内参进行 PnP 位姿估计。

**修改**:
- `solve_pose_with_guess()` 函数新增 `camera_yaml_path` 参数
- 自动使用 `get_camera_intrinsics()` 加载内参

### 4. 主程序 (`src/run_tilt_check.py`)

**功能**: 命令行界面，支持指定相机内参 YAML 文件。

**新增参数**:
- `--camera-yaml`: 指定相机内参 YAML 文件路径（默认: `config/camera_info.yaml`）

**使用方式**:
```bash
# 使用默认路径的内参文件
python src/run_tilt_check.py --image data/board.png

# 使用自定义路径的内参文件
python src/run_tilt_check.py --image data/board.png --camera-yaml /path/to/camera_info.yaml
```

## 文件格式

### YAML 文件格式 (`config/camera_info.yaml`)

```yaml
image_width: 640
image_height: 480
camera_name: camera
distortion_model: plumb_bob
camera_matrix:
  rows: 3
  cols: 3
  dt: d
  data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
distortion_coefficients:
  rows: 1
  cols: 5
  dt: d
  data: [k1, k2, p1, p2, k3]
rectification_matrix:
  rows: 3
  cols: 3
  dt: d
  data: [r11, r12, r13, r21, r22, r23, r31, r32, r33]
projection_matrix:
  rows: 3
  cols: 4
  dt: d
  data: [fx, 0, cx, tx, 0, fy, cy, ty, 0, 0, 1, tz]
```

**格式说明**:
- 兼容 OpenCV `cv::FileStorage` YAML 格式
- 内参矩阵 `K` 为 3×3 矩阵，按行展开为 9 个元素
- 畸变系数 `D` 为 1×5 向量（OpenCV 标准：k1, k2, p1, p2, k3）
- 图像尺寸用于验证内参与实际图像是否匹配

## 工作流程

### 1. 提取内参（一次性）

```
ROS2 CameraInfo 话题
    ↓
CameraInfoExtractor 节点
    ↓
提取内参矩阵 K 和畸变系数 D
    ↓
保存为 config/camera_info.yaml
```

### 2. 使用内参（每次运行）

```
运行 run_tilt_check.py
    ↓
get_camera_intrinsics() 尝试加载 YAML
    ↓
成功？ → 使用真实内参 → PnP 位姿估计
    ↓
失败？ → 使用默认内参 → PnP 位姿估计（近似）
```

## 错误处理

### 1. YAML 文件不存在

**行为**: 自动回退到默认内参

**日志**:
```
[WARN] 相机内参文件不存在: config/camera_info.yaml
[INFO] 使用默认内参（近似值）
```

### 2. YAML 文件格式错误

**行为**: 返回 None，回退到默认内参

**日志**:
```
[ERROR] 解析 YAML 文件失败: <error_message>
[INFO] 使用默认内参（近似值）
```

### 3. 图像尺寸不匹配

**行为**: 继续使用 YAML 中的内参，但显示警告

**日志**:
```
[WARN] 图像尺寸不匹配: YAML中为(640, 480), 实际为(1280, 720)
      继续使用 YAML 中的内参，但可能不够准确
```

## 测试

### 测试脚本 (`test_camera_intrinsics.py`)

**功能**: 验证内参加载功能是否正常工作

**测试项**:
1. YAML 文件加载测试
2. `get_camera_intrinsics()` 函数测试（自动回退）
3. 默认内参生成测试
4. YAML 文件格式验证

**运行方式**:
```bash
python test_camera_intrinsics.py
```

## 依赖

### Python 包（pip）
- `opencv-python`: OpenCV Python 绑定
- `numpy`: 数值计算
- `pyyaml`: YAML 文件解析

### ROS2 包（rosdep，仅 camera_rectifier.py 需要）
- `rclpy`: ROS2 Python 客户端库
- `sensor_msgs`: ROS2 传感器消息类型
- `cv_bridge`: OpenCV 和 ROS 图像消息转换

**安装 ROS2 依赖**:
```bash
sudo apt install ros-<distro>-rclpy ros-<distro>-sensor-msgs ros-<distro>-cv-bridge
```

## 使用建议

### 1. 何时需要重新提取内参

- 更换相机
- 改变图像分辨率
- 调整相机焦距（变焦镜头）
- 重新标定相机

### 2. 多相机支持

为每个相机创建独立的 YAML 文件：
```
config/
├── camera_info_front.yaml
├── camera_info_back.yaml
└── camera_info_left.yaml
```

使用时指定对应的文件：
```bash
python src/run_tilt_check.py --image data/board.png --camera-yaml config/camera_info_front.yaml
```

### 3. 内参精度

- **真实内参**: 更准确的位姿估计，考虑镜头畸变，适用于精确测量
- **默认内参**: 近似值（假设无畸变，焦距近似），仅适用于粗略判断

## 相关文档

- `docs/camera_intrinsics_usage.md`: 详细使用文档
- `docs/camera_intrinsics_quickstart.md`: 快速开始指南
- `config/README.md`: 配置目录说明

## 未来改进

1. **支持多种 YAML 格式**: 支持 OpenCV calibration 工具生成的 YAML 格式
2. **动态内参**: 支持从 ROS2 CameraInfo 话题实时获取内参（不保存文件）
3. **内参验证**: 添加内参合理性检查（焦距范围、主点位置等）
4. **多分辨率支持**: 自动缩放内参以适应不同图像分辨率


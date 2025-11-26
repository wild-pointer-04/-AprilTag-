# 快速开始指南 - 从零开始运行项目

本文档为交付方提供完整的项目运行指南，从环境配置到实际使用。

---

## 目录

1. [环境准备](#环境准备)
2. [项目安装](#项目安装)
3. [获取相机内参](#获取相机内参)
4. [单张图片测试](#单张图片测试)
5. [批量处理](#批量处理)
6. [ROS2 节点使用](#ros2-节点使用)
7. [从 rosbag 分析](#从-rosbag-分析)
8. [故障排除](#故障排除)

---

## 环境准备

### 1.1 系统要求

- **操作系统**: Ubuntu 20.04 / 22.04 / 24.04
- **Python**: Python 3.8 或更高版本
- **ROS2**: Humble / Foxy / Iron / Jazzy（可选，仅用于 ROS2 功能）

### 1.2 检查 Python 版本

```bash
python3 --version
# 应该显示 Python 3.8 或更高版本
```

### 1.3 安装系统依赖

```bash
# 更新包管理器
sudo apt update

# 安装基础依赖
sudo apt install -y python3-pip python3-venv git

# 如果使用 ROS2 功能，安装 ROS2 依赖
# 首先确定你的 ROS2 发行版（常见的有: humble, foxy, iron, jazzy）
ls /opt/ros/

# 假设是 humble，安装 ROS2 依赖
sudo apt install -y \
    ros-humble-rclpy \
    ros-humble-sensor-msgs \
    ros-humble-cv-bridge \
    ros-humble-rosbag2
```

---

## 项目安装

### 2.1 克隆或获取项目

```bash
# 如果项目在 Git 仓库中
git clone <repository_url>
cd tilt_checker

# 或者如果项目已经存在，直接进入目录
cd /path/to/tilt_checker
```

### 2.2 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 升级 pip
pip install --upgrade pip
```

### 2.3 安装 Python 依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 如果使用 ROS2 功能，还需要安装 ROS2 Python 包
# （通常通过系统包管理器安装，见 1.3）
```

### 2.4 验证安装

```bash
# 测试 OpenCV
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"

# 测试 NumPy
python3 -c "import numpy; print('NumPy version:', numpy.__version__)"

# 如果使用 ROS2，测试 ROS2
source /opt/ros/humble/setup.bash  # 替换为你的发行版
python3 -c "import rclpy; print('ROS2 Python 可用')"
```

---

## 获取相机内参

### 3.1 方法 1: 从 ROS2 CameraInfo 话题提取（推荐）

**前提条件**: 相机正在发布 ROS2 话题

```bash
# 激活 ROS2 环境
source /opt/ros/humble/setup.bash  # 替换为你的发行版

# 激活虚拟环境（如果使用）
source .venv/bin/activate

# 检查相机话题是否可用
ros2 topic list | grep camera_info

# 提取相机内参（替换话题名称为实际的话题）
python src/camera_rectifier.py \
    --camera_info_topic /camera/color/camera_info \
    --output config/camera_info.yaml

# 等待几秒钟，直到看到 "✅ 已保存 YAML 文件" 消息
# 按 Ctrl+C 退出
```

**输出**: `config/camera_info.yaml` 文件

### 3.2 方法 2: 手动创建 YAML 文件

如果无法从 ROS2 获取，可以手动创建 `config/camera_info.yaml`：

```yaml
%YAML:1.0
---
camera_matrix:
   rows: 3
   cols: 3
   data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
distortion_coefficients:
   rows: 1
   cols: 5
   data: [k1, k2, p1, p2, k3]
image_width: 1920
image_height: 1080
```

**注意**: 将 `fx, fy, cx, cy, k1, k2, p1, p2, k3` 替换为实际的相机内参值。

### 3.3 验证内参文件

```bash
python test_camera_intrinsics.py --camera-yaml config/camera_info.yaml
```

---

## 单张图片测试

### 4.1 准备测试图片

确保 `data/` 目录中有标定板图片（圆点网格，推荐 15×15）。

### 4.2 运行检测

```bash
# 激活虚拟环境（如果使用）
source .venv/bin/activate

# 运行检测
python src/run_tilt_check.py \
    --image data/board.png \
    --rows 15 \
    --cols 15 \
    --camera-yaml config/camera_info.yaml
```

### 4.3 查看结果

**终端输出**:
- 检测状态（是否成功）
- 板子中心像素坐标
- Roll/Pitch/Yaw 角度（板子相对于相机）
- 相机倾斜角（假设板子水平）
- 歪斜判断结果

**生成文件**:
- `outputs/board_result.png`: 检测结果图（带坐标轴、网格点）
- `outputs/board_result_with_blobs.png`: 带绿色 blob 点的结果图（用于调试）

### 4.4 理解输出

- **Roll(前后仰)**: 相机前后倾斜角度
- **Pitch(平面旋)**: 相机平面旋转角度
- **Yaw(左右歪)**: 相机左右倾斜角度

**歪斜判断**: 如果 Roll 或 Pitch 的绝对值超过 0.5°，系统会提示"存在歪斜"。

---

## 批量处理

### 5.1 批量处理目录中的所有图片

```bash
python src/run_tilt_check.py \
    --dir data \
    --rows 15 \
    --cols 15 \
    --camera-yaml config/camera_info.yaml
```

**输出**: `outputs/` 目录下每张图片对应两个结果文件。

### 5.2 自动搜索网格尺寸

如果不知道确切的网格尺寸，可以使用自动搜索：

```bash
python src/run_tilt_check.py \
    --image data/board.png \
    --auto
```

---

## ROS2 节点使用

### 6.1 实时检测节点

使用 ROS2 节点进行实时检测（从相机话题读取图像）：

```bash
# 激活 ROS2 环境
source /opt/ros/humble/setup.bash

# 激活虚拟环境（如果使用）
source .venv/bin/activate

# 运行节点
python src/tilt_checker_node.py \
    --image-topic /camera/image_raw \
    --camera-yaml config/camera_info.yaml \
    --rows 15 \
    --cols 15
```

**输出**: 
- 发布检测结果到 `/tilt_checker/result` 话题
- 在终端显示检测信息

### 6.2 使用 Launch 文件（推荐）

```bash
# 激活 ROS2 环境
source /opt/ros/humble/setup.bash

# 激活虚拟环境（如果使用）
source .venv/bin/activate

# 运行 launch 文件
ros2 launch tilt_checker tilt_checker.launch.py \
    image_topic:=/camera/image_raw \
    camera_yaml:=config/camera_info.yaml \
    rows:=15 \
    cols:=15
```

---

## 从 rosbag 分析

### 7.1 从 rosbag 读取并分析

```bash
# 激活 ROS2 环境
source /opt/ros/humble/setup.bash

# 激活虚拟环境（如果使用）
source .venv/bin/activate

# 从 rosbag 分析
python src/tilt_checker_node.py \
    --rosbag /path/to/your.bag \
    --image-topic /camera/image_raw \
    --camera-yaml config/camera_info.yaml \
    --rows 15 \
    --cols 15 \
    --output-dir outputs/rosbag_results
```

**输出**:
- 每帧的检测结果（JSON 或 CSV 格式）
- 可视化结果图
- 统计报告（平均误差、歪斜情况等）

### 7.2 批量分析多个 rosbag

```bash
# 创建脚本
cat > analyze_rosbags.sh << 'EOF'
#!/bin/bash
for bag in /path/to/bags/*.bag; do
    echo "处理: $bag"
    python src/tilt_checker_node.py \
        --rosbag "$bag" \
        --image-topic /camera/image_raw \
        --camera-yaml config/camera_info.yaml \
        --rows 15 \
        --cols 15 \
        --output-dir "outputs/$(basename $bag .bag)"
done
EOF

chmod +x analyze_rosbags.sh
./analyze_rosbags.sh
```

---

## 故障排除

### 8.1 常见错误

#### 错误 1: `ModuleNotFoundError: No module named 'src'`

**原因**: Python 路径问题

**解决**:
```bash
# 确保在项目根目录运行
cd /path/to/tilt_checker
python src/run_tilt_check.py ...
```

#### 错误 2: `无法加载相机内参`

**原因**: YAML 文件不存在或格式错误

**解决**:
```bash
# 检查文件是否存在
ls -l config/camera_info.yaml

# 验证文件格式
python test_camera_intrinsics.py --camera-yaml config/camera_info.yaml
```

#### 错误 3: `未检测到网格`

**原因**: 
- 图片中没有标定板
- 网格尺寸不匹配
- 图片质量太差

**解决**:
```bash
# 尝试自动搜索
python src/run_tilt_check.py --image data/board.png --auto

# 检查图片
# 使用图像查看器打开图片，确认标定板清晰可见
```

#### 错误 4: ROS2 相关错误

**原因**: ROS2 环境未激活

**解决**:
```bash
# 激活 ROS2 环境
source /opt/ros/humble/setup.bash  # 替换为你的发行版

# 检查 ROS2 是否可用
ros2 --version
```

### 8.2 获取帮助

```bash
# 查看脚本帮助
python src/run_tilt_check.py --help
python src/tilt_checker_node.py --help
python src/calibration_and_reprojection.py --help
```

### 8.3 调试模式

如果遇到问题，可以查看详细日志：

```bash
# 设置 Python 日志级别
export PYTHONUNBUFFERED=1

# 运行脚本，查看详细输出
python src/run_tilt_check.py --image data/board.png --rows 15 --cols 15
```

---

## 下一步

- 查看 [运行命令文档](run_commands.md) 了解所有脚本的详细用法
- 查看 [相机内参使用指南](camera_intrinsics_usage.md) 了解内参管理
- 查看 [标定和重投影误差分析](calibration_and_reprojection_usage.md) 了解误差分析

---

## 联系支持

如果遇到问题，请提供：
1. 错误信息（完整输出）
2. 使用的命令
3. 系统信息（`uname -a`）
4. Python 版本（`python3 --version`）
5. ROS2 版本（如果使用，`ros2 --version`）


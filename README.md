# 相机倾斜检测工具 (Tilt Checker)

一个用于检测相机安装倾斜度的工具，通过分析圆点标定板图像，计算相机倾斜角度和重投影误差。

---

## 功能特性

- ✅ **圆点网格检测**: 自动检测 15×15（或其他尺寸）圆点标定板
- ✅ **畸变矫正**: 使用相机内参对图像进行去畸变
- ✅ **倾斜角度计算**: 计算 Roll/Pitch/Yaw 角度（板子相对于相机，以及相机相对于水平面）
- ✅ **重投影误差分析**: 计算并可视化重投影误差分布
- ✅ **ROS2 集成**: 支持从 ROS2 话题或 rosbag 读取图像
- ✅ **批量处理**: 支持批量处理多张图片或多个 rosbag
- ✅ **结果导出**: 导出 JSON/CSV 格式的检测结果和统计报告

---

## 快速开始

### 1. 环境准备

```bash
# 克隆或获取项目
cd /path/to/tilt_checker

# 创建虚拟环境（推荐）
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 获取相机内参

```bash
# 从 ROS2 CameraInfo 话题提取（如果使用 ROS2）
source /opt/ros/humble/setup.bash  # 替换为你的发行版
python src/camera_rectifier.py \
    --camera_info_topic /camera/color/camera_info \
    --output config/camera_info.yaml
```

### 3. 单张图片测试

```bash
python src/run_tilt_check.py \
    --image data/board.png \
    --rows 15 \
    --cols 15 \
    --camera-yaml config/camera_info.yaml
```

### 4. 从 rosbag 分析

```bash
source /opt/ros/humble/setup.bash
python src/tilt_checker_node.py \
    --rosbag /path/to/your.bag \
    --image-topic /camera/image_raw \
    --camera-yaml config/camera_info.yaml \
    --rows 15 \
    --cols 15 \
    --save-images \
    --output-dir outputs/rosbag_results
```

---

## 文档

- **[快速开始指南](docs/QUICKSTART.md)**: 从零开始的完整使用指南
- **[ROS2 节点使用](docs/ROS2_NODE_USAGE.md)**: ROS2 节点的详细使用说明
- **[运行命令文档](docs/run_commands.md)**: 所有脚本的命令行参数说明
- **[相机内参使用指南](docs/camera_intrinsics_usage.md)**: 相机内参的获取和使用

---

## 项目结构

```
tilt_checker/
├── config/              # 配置文件
│   └── camera_info.yaml  # 相机内参（需要从 ROS2 或手动创建）
├── data/                 # 测试图像
├── outputs/              # 输出结果
├── src/                  # 源代码
│   ├── run_tilt_check.py           # 主检测脚本
│   ├── tilt_checker_node.py        # ROS2 节点
│   ├── camera_rectifier.py         # 相机内参提取工具
│   ├── calibration_and_reprojection.py  # 标定和误差分析
│   ├── detect_grid_improved.py     # 网格检测（改进版）
│   ├── estimate_tilt.py            # 倾斜角度计算
│   └── utils.py                    # 工具函数
├── launch/               # ROS2 launch 文件
│   └── tilt_checker.launch.py
├── docs/                 # 文档
└── requirements.txt      # Python 依赖
```

---

## 主要脚本

### 1. `run_tilt_check.py` - 单张/批量图片检测

```bash
python src/run_tilt_check.py --image data/board.png --rows 15 --cols 15
```

### 2. `tilt_checker_node.py` - ROS2 节点（支持 rosbag）

```bash
python src/tilt_checker_node.py --rosbag /path/to/bag --image-topic /camera/image_raw
```

### 3. `camera_rectifier.py` - 提取相机内参

```bash
python src/camera_rectifier.py --camera_info_topic /camera/camera_info --output config/camera_info.yaml
```

### 4. `calibration_and_reprojection.py` - 标定和误差分析

```bash
python src/calibration_and_reprojection.py --data-dir data --camera-yaml config/camera_info.yaml
```

---

## 输出结果

### 单张图片检测

- `outputs/{图片名}_result.png`: 检测结果图（带坐标轴、网格点）
- `outputs/{图片名}_result_with_blobs.png`: 带绿色 blob 点的结果图

### ROS2 节点 / rosbag 分析

- `outputs/{output_dir}/results.json`: 所有帧的详细结果（JSON）
- `outputs/{output_dir}/results.csv`: 结果表格（CSV）
- `outputs/{output_dir}/summary_report.txt`: 统计报告
- `outputs/{output_dir}/images/`: 每帧的检测结果图像（如果启用）

---

## 角度说明

### Roll (前后仰)
相机前后倾斜角度。正值表示相机向前倾斜，负值表示向后倾斜。

### Pitch (平面旋)
相机平面旋转角度。正值表示顺时针旋转，负值表示逆时针旋转。

### Yaw (左右歪)
相机左右倾斜角度。正值表示向右倾斜，负值表示向左倾斜。

**注意**: 
- **标准欧拉角**: 板子相对于相机的旋转（用于描述板子姿态）
- **相机倾斜角**: 假设板子水平，相机相对于水平面的倾斜（用于检测相机安装问题）

---

## 系统要求

- **操作系统**: Ubuntu 20.04 / 22.04 / 24.04
- **Python**: 3.8+
- **ROS2**: Humble / Foxy / Iron / Jazzy（可选，仅用于 ROS2 功能）

### 依赖安装

```bash
# Python 依赖
pip install -r requirements.txt

# ROS2 依赖（如果使用 ROS2 功能）
sudo apt install -y \
    ros-humble-rclpy \
    ros-humble-sensor-msgs \
    ros-humble-cv-bridge \
    ros-humble-rosbag2-py
```

---

## 故障排除

### 常见问题

1. **无法检测到网格**
   - 检查图片中是否有清晰的标定板
   - 尝试使用 `--auto` 自动搜索网格尺寸
   - 调整 blob 检测参数（修改 `src/utils.py`）

2. **无法加载相机内参**
   - 确保 `config/camera_info.yaml` 存在
   - 验证 YAML 文件格式是否正确

3. **ROS2 相关错误**
   - 确保已激活 ROS2 环境: `source /opt/ros/humble/setup.bash`
   - 检查 ROS2 发行版是否正确

更多问题请参考 [快速开始指南](docs/QUICKSTART.md) 中的"故障排除"部分。

---

## 许可证

[根据项目实际情况填写]

---

## 联系方式

[根据项目实际情况填写]


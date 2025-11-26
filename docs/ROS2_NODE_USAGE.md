# ROS2 节点使用指南

本文档说明如何使用 ROS2 节点进行相机倾斜检测。

---

## 目录

1. [节点功能](#节点功能)
2. [安装和配置](#安装和配置)
3. [从实时话题检测](#从实时话题检测)
4. [从 rosbag 分析](#从-rosbag-分析)
5. [输出结果](#输出结果)
6. [Launch 文件使用](#launch-文件使用)
7. [故障排除](#故障排除)

---

## 节点功能

`tilt_checker_node.py` 是一个完整的 ROS2 节点，提供以下功能：

1. **图像读取**: 从 ROS2 话题或 rosbag 读取图像
2. **畸变矫正**: 使用相机内参对图像进行去畸变
3. **网格检测**: 检测圆点标定板网格（15×15 或其他尺寸）
4. **倾斜计算**: 计算相机倾斜角度（Roll/Pitch/Yaw）
5. **误差分析**: 计算重投影误差
6. **结果输出**: 保存检测结果（JSON/CSV/图像）和统计报告

---

## 安装和配置

### 1. 环境准备

```bash
# 激活 ROS2 环境
source /opt/ros/humble/setup.bash  # 替换为你的发行版

# 激活虚拟环境（如果使用）
cd /path/to/tilt_checker
source .venv/bin/activate
```

### 2. 安装依赖

```bash
# 安装 ROS2 依赖
sudo apt install -y \
    ros-humble-rclpy \
    ros-humble-sensor-msgs \
    ros-humble-cv-bridge \
    ros-humble-rosbag2-py

# 安装 Python 依赖
pip install -r requirements.txt
```

### 3. 准备相机内参

确保 `config/camera_info.yaml` 文件存在且正确。如果不存在，参考 [快速开始指南](QUICKSTART.md) 中的"获取相机内参"部分。

---

## 从实时话题检测

### 基本用法

```bash
# 激活环境
source /opt/ros/humble/setup.bash
source .venv/bin/activate

# 运行节点
python src/tilt_checker_node.py \
    --image-topic /camera/image_raw \
    --camera-yaml config/camera_info.yaml \
    --rows 15 \
    --cols 15
```

### 参数说明

- `--image-topic`: 图像话题名称（默认: `/camera/image_raw`）
- `--camera-yaml`: 相机内参 YAML 文件路径（默认: `config/camera_info.yaml`）
- `--rows`: 圆点行数（默认: 15）
- `--cols`: 圆点列数（默认: 15）
- `--spacing`: 圆点间距（mm，默认: 10.0）
- `--save-images`: 保存检测结果图像
- `--output-dir`: 输出目录（默认: `outputs/rosbag_results`）

### 示例：保存结果图像

```bash
python src/tilt_checker_node.py \
    --image-topic /camera/image_raw \
    --camera-yaml config/camera_info.yaml \
    --rows 15 \
    --cols 15 \
    --save-images \
    --output-dir outputs/realtime_results
```

---

## 从 rosbag 分析

### 基本用法

```bash
# 激活环境
source /opt/ros/humble/setup.bash
source .venv/bin/activate

# 从 rosbag 分析
python src/tilt_checker_node.py \
    --rosbag /path/to/your.bag \
    --image-topic /camera/image_raw \
    --camera-yaml config/camera_info.yaml \
    --rows 15 \
    --cols 15 \
    --save-images \
    --output-dir outputs/rosbag_analysis
```

### 批量处理多个 rosbag

创建脚本 `analyze_bags.sh`:

```bash
#!/bin/bash

source /opt/ros/humble/setup.bash
source /path/to/tilt_checker/.venv/bin/activate

BAG_DIR="/path/to/rosbags"
OUTPUT_BASE="outputs/bag_analysis"

for bag in "$BAG_DIR"/*.bag; do
    if [ -f "$bag" ]; then
        bag_name=$(basename "$bag" .bag)
        output_dir="$OUTPUT_BASE/$bag_name"
        
        echo "=========================================="
        echo "处理: $bag"
        echo "输出目录: $output_dir"
        echo "=========================================="
        
        python src/tilt_checker_node.py \
            --rosbag "$bag" \
            --image-topic /camera/image_raw \
            --camera-yaml config/camera_info.yaml \
            --rows 15 \
            --cols 15 \
            --save-images \
            --output-dir "$output_dir"
        
        echo ""
    fi
done

echo "✅ 所有 rosbag 处理完成"
```

运行脚本:

```bash
chmod +x analyze_bags.sh
./analyze_bags.sh
```

---

## 输出结果

节点会在指定的输出目录生成以下文件：

### 1. `results.json`

包含所有帧的详细检测结果：

```json
{
  "summary": {
    "total_frames": 100,
    "success_count": 95,
    "failure_count": 5,
    "success_rate": 0.95
  },
  "results": [
    {
      "frame_id": "frame_000000",
      "timestamp": 1234567890.123,
      "success": true,
      "board_center_px": {
        "mean": {"u": 960.0, "v": 540.0},
        "mid": {"u": 960.0, "v": 540.0}
      },
      "euler_angles": {
        "roll": 1.23,
        "pitch": -0.45,
        "yaw": 0.12
      },
      "camera_tilt_angles": {
        "roll": 1.23,
        "pitch": -0.45,
        "yaw": 0.12
      },
      "reprojection_error": {
        "mean": 0.123,
        "max": 0.456,
        "point_count": 225
      },
      "tilt_detection": {
        "has_tilt": true,
        "roll_offset": 1.23,
        "pitch_offset": -0.45,
        "threshold": 0.5
      }
    }
  ]
}
```

### 2. `results.csv`

CSV 格式的结果，便于在 Excel 或其他工具中分析：

| frame_id | timestamp | success | center_u | center_v | roll_tilt | pitch_tilt | yaw_tilt | reprojection_error_mean | has_tilt |
|----------|-----------|---------|----------|----------|-----------|------------|----------|------------------------|----------|
| frame_000000 | 1234567890.123 | True | 960.0 | 540.0 | 1.23 | -0.45 | 0.12 | 0.123 | True |

### 3. `summary_report.txt`

统计报告，包含：
- 总帧数、成功率
- 角度统计（平均值、最大值）
- 重投影误差统计
- 歪斜检测统计

### 4. `images/` 目录（如果启用 `--save-images`）

每帧的检测结果图像，文件名格式：`{frame_id}_result.png`

---

## Launch 文件使用

### 使用 Launch 文件启动节点

```bash
# 激活环境
source /opt/ros/humble/setup.bash
source /path/to/tilt_checker/.venv/bin/activate

# 运行 launch 文件
ros2 launch tilt_checker tilt_checker.launch.py \
    image_topic:=/camera/image_raw \
    camera_yaml:=config/camera_info.yaml \
    rows:=15 \
    cols:=15 \
    save_images:=true
```

### Launch 文件参数

- `image_topic`: 图像话题名称
- `camera_yaml`: 相机内参 YAML 文件路径
- `rows`: 圆点行数
- `cols`: 圆点列数
- `spacing`: 圆点间距（mm）
- `output_dir`: 输出目录
- `save_images`: 是否保存图像（true/false）

---

## 故障排除

### 问题 1: 无法找到图像话题

**错误**: `在 rosbag 中未找到话题: /camera/image_raw`

**解决**:
```bash
# 检查 rosbag 中的话题
ros2 bag info /path/to/your.bag

# 使用正确的话题名称
python src/tilt_checker_node.py \
    --rosbag /path/to/your.bag \
    --image-topic /actual/image/topic \
    ...
```

### 问题 2: rosbag2_py 不可用

**错误**: `需要安装 rosbag2_py`

**解决**:
```bash
sudo apt install ros-humble-rosbag2-py
```

### 问题 3: 检测失败率高

**可能原因**:
- 图像质量差
- 网格尺寸不匹配
- 标定板未完全可见

**解决**:
- 检查图像质量
- 尝试自动搜索网格尺寸（需要修改代码）
- 调整 blob 检测参数（修改 `src/utils.py` 中的 `BLOB_DETECTOR_PARAMS`）

### 问题 4: 内存不足

**解决**:
- 处理大型 rosbag 时，可以分批处理
- 禁用 `--save-images` 以减少内存使用
- 使用 `--no-save-results` 如果只需要实时输出

---

## 性能优化

### 1. 跳过某些帧

如果需要处理大量帧，可以修改代码跳过某些帧（例如每 5 帧处理一次）。

### 2. 并行处理

对于多个 rosbag，可以使用并行处理脚本：

```bash
# 使用 GNU parallel（如果已安装）
parallel -j 4 python src/tilt_checker_node.py \
    --rosbag {} \
    --image-topic /camera/image_raw \
    --camera-yaml config/camera_info.yaml \
    --rows 15 --cols 15 \
    --output-dir outputs/bag_{/} \
    ::: /path/to/bags/*.bag
```

---

## 下一步

- 查看 [快速开始指南](QUICKSTART.md) 了解基础使用
- 查看 [运行命令文档](run_commands.md) 了解所有脚本的用法
- 查看 [标定和重投影误差分析](calibration_and_reprojection_usage.md) 了解误差分析


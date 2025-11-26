# 交付方使用指南

本文档为交付方提供完整的项目使用流程，从环境配置到实际检测。

---

## 一、快速开始（5分钟上手）

### 步骤 1: 环境准备

```bash
# 1. 进入项目目录
cd /path/to/tilt_checker

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

### 步骤 2: 获取相机内参

**方法 A: 从 ROS2 话题提取（推荐）**

```bash
# 激活 ROS2 环境
source /opt/ros/humble/setup.bash  # 替换为你的发行版

# 提取内参（需要相机正在发布话题）
python src/camera_rectifier.py \
    --camera_info_topic /camera/color/camera_info \
    --output config/camera_info.yaml
```

**方法 B: 手动创建 YAML 文件**

如果无法从 ROS2 获取，手动创建 `config/camera_info.yaml`（参考 `config/README.md`）

### 步骤 3: 测试单张图片

```bash
python src/run_tilt_check.py \
    --image data/board.png \
    --rows 15 \
    --cols 15 \
    --camera-yaml config/camera_info.yaml
```

**查看结果**: `outputs/board_result.png` 和 `outputs/board_result_with_blobs.png`

---

## 二、从 rosbag 分析（推荐工作流程）

### 完整流程

```bash
# 1. 激活环境
source /opt/ros/humble/setup.bash
source .venv/bin/activate

# 2. 从 rosbag 分析
python src/tilt_checker_node.py \
    --rosbag /path/to/your.bag \
    --image-topic /camera/image_raw \
    --camera-yaml config/camera_info.yaml \
    --rows 15 \
    --cols 15 \
    --save-images \
    --output-dir outputs/rosbag_analysis
```

### 输出结果

分析完成后，在 `outputs/rosbag_analysis/` 目录下会生成：

1. **`results.json`**: 所有帧的详细检测结果
2. **`results.csv`**: 结果表格（可用 Excel 打开）
3. **`summary_report.txt`**: 统计报告（包含平均角度、误差、歪斜情况等）
4. **`images/`**: 每帧的检测结果图像（如果启用 `--save-images`）

### 查看统计报告

```bash
cat outputs/rosbag_analysis/summary_report.txt
```

报告包含：
- 总帧数、成功率
- 相机倾斜角度统计（Roll/Pitch/Yaw 的平均值、最大值）
- 重投影误差统计
- 歪斜检测统计（多少帧存在歪斜）

---

## 三、批量处理多个 rosbag

### 创建批量处理脚本

创建文件 `analyze_all_bags.sh`:

```bash
#!/bin/bash

# 激活环境
source /opt/ros/humble/setup.bash
source /path/to/tilt_checker/.venv/bin/activate

# 设置路径
BAG_DIR="/path/to/your/rosbags"
OUTPUT_BASE="outputs/bag_analysis"

# 遍历所有 rosbag
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
chmod +x analyze_all_bags.sh
./analyze_all_bags.sh
```

---

## 四、结果解读

### 1. 角度含义

- **Roll (前后仰)**: 相机前后倾斜
  - 正值 = 向前倾斜
  - 负值 = 向后倾斜
  - 阈值: ±0.5°（超过此值认为存在歪斜）

- **Pitch (平面旋)**: 相机平面旋转
  - 正值 = 顺时针旋转
  - 负值 = 逆时针旋转
  - 阈值: ±0.5°

- **Yaw (左右歪)**: 相机左右倾斜
  - 正值 = 向右倾斜
  - 负值 = 向左倾斜

### 2. 重投影误差

- **平均误差**: 所有点的平均重投影误差（像素）
- **最大误差**: 最大重投影误差（像素）
- **正常范围**: 通常 < 0.5 像素表示标定质量良好

### 3. 歪斜判断

系统会自动判断是否存在歪斜：
- **正常**: Roll 和 Pitch 的绝对值都 < 0.5°
- **存在歪斜**: Roll 或 Pitch 的绝对值 ≥ 0.5°

---

## 五、常见使用场景

### 场景 1: 检测单个摄像头安装

```bash
# 从 rosbag 分析
python src/tilt_checker_node.py \
    --rosbag camera1.bag \
    --image-topic /camera/image_raw \
    --camera-yaml config/camera_info.yaml \
    --rows 15 --cols 15 \
    --save-images \
    --output-dir outputs/camera1_check
```

**查看结果**: 
- 打开 `outputs/camera1_check/summary_report.txt` 查看统计
- 检查 `outputs/camera1_check/results.csv` 查看每帧的详细数据

### 场景 2: 批量检测多个摄像头

```bash
# 使用批量处理脚本（见"三、批量处理多个 rosbag"）
./analyze_all_bags.sh
```

### 场景 3: 实时检测（从 ROS2 话题）

```bash
# 运行节点，订阅实时图像话题
python src/tilt_checker_node.py \
    --image-topic /camera/image_raw \
    --camera-yaml config/camera_info.yaml \
    --rows 15 --cols 15 \
    --save-images \
    --output-dir outputs/realtime_check
```

按 `Ctrl+C` 退出时，会自动保存结果。

---

## 六、故障排除

### 问题 1: 无法检测到网格

**症状**: 终端显示 "未检测到网格"

**可能原因**:
- 图片中没有标定板
- 网格尺寸不匹配（不是 15×15）
- 图片质量太差

**解决方法**:
```bash
# 尝试自动搜索网格尺寸
python src/run_tilt_check.py --image data/board.png --auto
```

### 问题 2: rosbag 中找不到话题

**症状**: "在 rosbag 中未找到话题: /camera/image_raw"

**解决方法**:
```bash
# 查看 rosbag 中的话题
ros2 bag info /path/to/your.bag

# 使用正确的话题名称
python src/tilt_checker_node.py \
    --rosbag /path/to/your.bag \
    --image-topic /actual/image/topic \
    ...
```

### 问题 3: 无法加载相机内参

**症状**: "无法加载相机内参"

**解决方法**:
```bash
# 检查文件是否存在
ls -l config/camera_info.yaml

# 验证文件格式
python test_camera_intrinsics.py --camera-yaml config/camera_info.yaml
```

### 问题 4: ROS2 相关错误

**症状**: "rclpy" 或 "rosbag2_py" 相关错误

**解决方法**:
```bash
# 安装 ROS2 依赖
sudo apt install -y \
    ros-humble-rclpy \
    ros-humble-sensor-msgs \
    ros-humble-cv-bridge \
    ros-humble-rosbag2-py

# 激活 ROS2 环境
source /opt/ros/humble/setup.bash
```

---

## 七、最佳实践

### 1. 标定板准备

- 使用 15×15 圆点标定板（推荐）
- 确保标定板平整、清晰
- 标定板应占图像的大部分区域

### 2. 图像质量

- 确保图像清晰，对焦准确
- 避免过曝或欠曝
- 确保标定板完全可见

### 3. 批量处理

- 对于大量 rosbag，使用批量处理脚本
- 定期检查 `summary_report.txt` 了解整体情况
- 对于异常帧，查看 `images/` 目录中的结果图

### 4. 结果分析

- 重点关注 `summary_report.txt` 中的统计信息
- 使用 Excel 打开 `results.csv` 进行进一步分析
- 对于存在歪斜的摄像头，检查 `roll_offset` 和 `pitch_offset`

---

## 八、技术支持

如果遇到问题，请提供：

1. **错误信息**: 完整的终端输出
2. **使用的命令**: 完整的命令行
3. **系统信息**: 
   ```bash
   uname -a
   python3 --version
   ros2 --version  # 如果使用 ROS2
   ```
4. **相关文件**: 
   - `config/camera_info.yaml`（如果存在）
   - 问题图片或 rosbag 的路径

---

## 九、快速参考

### 最常用命令

```bash
# 单张图片检测
python src/run_tilt_check.py --image data/board.png --rows 15 --cols 15

# 从 rosbag 分析
python src/tilt_checker_node.py \
    --rosbag /path/to/bag \
    --image-topic /camera/image_raw \
    --camera-yaml config/camera_info.yaml \
    --rows 15 --cols 15 \
    --save-images

# 提取相机内参
python src/camera_rectifier.py \
    --camera_info_topic /camera/color/camera_info \
    --output config/camera_info.yaml
```

### 输出文件位置

- 单张图片: `outputs/{图片名}_result.png`
- rosbag 分析: `outputs/{output_dir}/results.json`, `results.csv`, `summary_report.txt`

---

## 十、下一步

- 查看 [快速开始指南](QUICKSTART.md) 了解更多细节
- 查看 [ROS2 节点使用](ROS2_NODE_USAGE.md) 了解 ROS2 功能
- 查看 [运行命令文档](run_commands.md) 了解所有参数


# 相机倾斜检测系统 - 完整使用指南

一个基于AprilTag和圆点标定板的相机倾斜检测系统，用于精确测量相机的安装角度和位姿。


直接使用：

用rosbag进行检测
python robust_tilt_checker_node.py   --max-error 3.0   --rosbag rosbags/testbag   --image-topic /left/color/image_raw   --camera-yaml config/camera_info.yaml   --rows 15 --cols 15   --tag-family tagStandard41h12   --tag-size 0.0071   --board-spacing 0.065   --publish-results   --save-images   --output-dir outputs/robust_apriltag_recording_final_result

从data数据集图片进行检测
python robust_tilt_checker_node.py   --max-error 20.0   --image-dir data   --camera-yaml config/camera_info.yaml   --rows 15   --cols 15   --tag-family tagStandard41h12   --tag-size 0.0071   --board-spacing 0.065   --save-images   --output-dir outputs/robust_apriltag_recording_final_resl

计算由于重投影误差造成的3D世界误差
python3 calculate_3d_error.py   --reprojection-error 0.8   --distance 1600   --camera-yaml config/camera_info.yaml   --rows 15   --cols 15   --spacing 65.0   --gripper-offset 0

计算像素角分辨率，PPD和DPP
python compute_pixel_angular_resolution.py --camera-yaml config/camera_info.yaml --image-dir data

---


## 📋 目录

- [项目简介](#项目简介)
- [主要功能](#主要功能)
- [快速开始](#快速开始)
- [系统要求](#系统要求)
- [安装步骤](#安装步骤)
- [使用方法](#使用方法)
- [输出结果](#输出结果)
- [常见问题](#常见问题)
- [项目结构](#项目结构)
- [技术原理](#技术原理)

---

## 🎯 项目简介

本项目是一个专业的相机倾斜检测工具，主要用于：
- **相机安装质量检测**：检测相机是否水平安装
- **机器人视觉标定**：为机器人视觉系统提供精确的相机位姿
- **AprilTag定位**：使用AprilTag建立统一的坐标系
- **重投影误差分析**：评估标定质量

### 核心特点

✅ **鲁棒的AprilTag系统**：解决了传统方法中247像素重投影误差问题  
✅ **多种PnP方法交叉验证**：提高位姿估计的准确性  
✅ **自动网格检测**：支持15×15圆点标定板的自动检测  
✅ **ROS2集成**：完整支持ROS2生态系统  
✅ **批量处理**：支持从rosbag批量处理图像  
✅ **详细可视化**：生成带有坐标轴、角点、误差信息的结果图像

---

## 🚀 主要功能

### 1. 相机倾斜角度检测
- **Roll（横滚角）**：相机前后倾斜，绕X轴旋转
- **Pitch（俯仰角）**：相机平面旋转，绕Z轴旋转
- **Yaw（偏航角）**：相机左右倾斜，绕Y轴旋转

### 2. AprilTag坐标系建立
- 自动检测AprilTag标签（支持tagStandard41h12家族）
- 建立统一的世界坐标系
- 重新排列圆点网格，确保坐标一致性

### 3. 重投影误差分析
- 计算每个角点的重投影误差
- 自动淘汰高误差帧（可配置阈值）
- 生成详细的误差统计报告

### 4. ROS2集成
- 从ROS2话题实时读取图像
- 从rosbag批量处理历史数据
- 发布相机到标定板的变换参数

---

## 💻 系统要求

### 操作系统
- Ubuntu 20.04 / 22.04 / 24.04（推荐）
- 其他Linux发行版（需要ROS2支持）

### 软件依赖
- **Python**: 3.8 或更高版本
- **ROS2**: Humble / Foxy / Iron / Jazzy（可选，用于ROS2功能）
- **OpenCV**: < 4.10.0
- **NumPy**: < 2.0.0

### 硬件要求
- **相机**：任何支持ROS2的相机（如Intel RealSense、Orbbec等）
- **标定板**：15×15圆点标定板 + AprilTag标签
- **计算机**：建议4GB以上内存

---

## 📦 安装步骤

### 第一步：克隆项目

```bash
# 进入你的工作目录
cd ~/your_workspace

# 如果项目已存在，直接进入
cd /path/to/tilt_checker
```

### 第二步：创建Python虚拟环境（推荐）

```bash
# 创建虚拟环境
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 你会看到命令行前面出现 (.venv) 标记
```

### 第三步：安装Python依赖

```bash
# 安装所有Python包
pip install -r requirements.txt
```

**依赖说明**：
- `opencv-python`: 图像处理
- `numpy`: 数值计算
- `pyyaml`: 配置文件读取
- `apriltag`: AprilTag检测

### 第四步：安装ROS2依赖（如果使用ROS2功能）

```bash
# 确保ROS2环境已安装
source /opt/ros/humble/setup.bash  # 替换为你的ROS2版本

# 安装ROS2 Python包
sudo apt install -y \
    ros-humble-rclpy \
    ros-humble-sensor-msgs \
    ros-humble-cv-bridge \
    ros-humble-rosbag2-py
```

### 第五步：准备相机内参文件

相机内参文件是必需的，它包含相机的焦距、畸变系数等信息。

#### 方法1：从ROS2话题提取（推荐）

```bash
# 1. 启动你的相机（新终端）
source /opt/ros/humble/setup.bash
ros2 launch orbbec_camera gemini_330_series.launch.py  # 替换为你的相机启动命令

# 2. 提取相机内参（另一个终端）
source /opt/ros/humble/setup.bash
source .venv/bin/activate  # 如果使用虚拟环境
python src/camera_rectifier.py \
    --camera_info_topic /camera/color/camera_info \
    --output config/camera_info.yaml
```

#### 方法2：手动创建

如果你已经有相机标定数据，可以手动创建 `config/camera_info.yaml`：

```yaml
image_width: 848
image_height: 480
camera_name: camera
distortion_model: plumb_bob
camera_matrix:
  rows: 3
  cols: 3
  data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
distortion_coefficients:
  rows: 1
  cols: 8
  data: [k1, k2, p1, p2, k3, 0, 0, 0]
```

### 第六步：验证安装

```bash
# 检查Python依赖
python -c "import cv2, numpy, yaml, apriltag; print('✅ 所有依赖已安装')"

# 检查相机内参文件
ls config/camera_info.yaml && echo "✅ 相机内参文件存在"

# 检查主程序
python robust_tilt_checker_node.py --help
```

如果所有命令都成功执行，说明安装完成！

---

## 🎮 使用方法

### 核心程序：robust_tilt_checker_node.py

这是本项目的主要应用程序，提供最鲁棒的检测结果。

### 使用场景1：从ROS2话题实时检测

适用于：实时监控相机倾斜情况

```bash
# 终端1：启动相机
source /opt/ros/humble/setup.bash
ros2 launch orbbec_camera gemini_330_series.launch.py

# 终端2：运行检测节点
source /opt/ros/humble/setup.bash
source .venv/bin/activate  # 如果使用虚拟环境
python robust_tilt_checker_node.py \
    --image-topic /camera/color/image_raw \
    --camera-yaml config/camera_info.yaml \
    --rows 15 \
    --cols 15 \
    --save-images \
    --output-dir outputs/realtime_results
```

**参数说明**：
- `--image-topic`: ROS2图像话题名称
- `--camera-yaml`: 相机内参文件路径
- `--rows`: 标定板行数（默认15）
- `--cols`: 标定板列数（默认15）
- `--save-images`: 保存检测结果图像
- `--output-dir`: 输出目录

### 使用场景2：从rosbag批量处理

适用于：分析历史数据，批量处理多帧图像

```bash
source /opt/ros/humble/setup.bash
source .venv/bin/activate
python robust_tilt_checker_node.py \
    --rosbag /path/to/your_rosbag \
    --image-topic /camera/color/image_raw \
    --camera-yaml config/camera_info.yaml \
    --rows 15 \
    --cols 15 \
    --save-images \
    --output-dir outputs/rosbag_results \
    --max-error 1.0
```

**额外参数**：
- `--rosbag`: rosbag文件路径
- `--max-error`: 最大允许重投影误差（像素，默认1.0）
- `--skip-frames`: 跳帧处理（默认1，处理所有帧）
- `--max-frames`: 最大处理帧数（可选）

### 使用场景3：发布变换参数到ROS2

适用于：与机器人系统集成，实时发布相机位姿

```bash
python robust_tilt_checker_node.py \
    --image-topic /camera/color/image_raw \
    --camera-yaml config/camera_info.yaml \
    --publish-results \
    --output-dir outputs/with_publish
```

程序会发布到话题：`/tilt_checker/camera_to_board_transform`

**消息格式**：`Float64MultiArray`，包含6个元素：
```
[δx, δy, δz, γ, α, β]
```
- δx, δy, δz: 平移向量（米）
- γ, α, β: ZYX欧拉角（弧度）

### 完整参数列表

```bash
python robust_tilt_checker_node.py --help
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--image-topic` | str | `/camera/color/image_raw` | ROS2图像话题 |
| `--camera-yaml` | str | `config/camera_info.yaml` | 相机内参文件 |
| `--rows` | int | 15 | 标定板行数 |
| `--cols` | int | 15 | 标定板列数 |
| `--tag-family` | str | `tagStandard41h12` | AprilTag家族 |
| `--tag-size` | float | 0.071 | AprilTag尺寸（米） |
| `--board-spacing` | float | 0.065 | 圆点间距（米） |
| `--max-error` | float | 1.0 | 最大重投影误差（像素） |
| `--rosbag` | str | None | rosbag文件路径 |
| `--output-dir` | str | `outputs/robust_apriltag_results` | 输出目录 |
| `--save-images` | flag | False | 保存结果图像 |
| `--no-save-results` | flag | False | 不保存结果文件 |
| `--publish-results` | flag | False | 发布到ROS话题 |
| `--skip-frames` | int | 1 | 跳帧处理 |
| `--max-frames` | int | None | 最大处理帧数 |

---

## 📊 输出结果

### 输出目录结构

```
outputs/robust_apriltag_results/
├── images/                          # 可视化图像（如果启用--save-images）
│   ├── frame_000001_robust_result.png
│   ├── frame_000002_robust_result.png
│   └── ...
├── robust_results.json              # 详细结果（JSON格式）
├── detailed_results.csv             # 结果表格（CSV格式）
└── summary_report.txt               # 统计报告
```

### 1. 可视化图像

每张图像包含：
- **黄色圆点**：检测到的标定板角点
- **绿色圆圈**：blob检测到的所有点
- **绿色方框**：AprilTag检测框
- **红色箭头**：X轴（向右）
- **绿色箭头**：Y轴（向前）
- **蓝色箭头**：Z轴（向上）
- **左上角信息**：
  - 帧名称
  - AprilTag状态和ID
  - 重投影误差
  - Roll/Pitch/Yaw角度

### 2. JSON结果文件（robust_results.json）

```json
{
  "system_info": {
    "tag_family": "tagStandard41h12",
    "tag_size_mm": 0.071,
    "board_spacing_mm": 0.065,
    "max_reprojection_error_px": 1.0,
    "grid_size": "15x15"
  },
  "summary": {
    "total_frames": 100,
    "success_count": 95,
    "failure_count": 5,
    "rejected_by_error_count": 2,
    "apriltag_success_count": 93,
    "success_rate": 0.95,
    "apriltag_success_rate": 0.979
  },
  "results": [
    {
      "frame_id": "frame_000001",
      "timestamp": 1234567890.123,
      "success": true,
      "apriltag_success": true,
      "method_used": "ITERATIVE_WITH_APRILTAG",
      "camera_tilt_angles": {
        "roll": 0.15,
        "pitch": -0.08,
        "yaw": 0.03
      },
      "reprojection_error": {
        "mean": 0.45,
        "method": "ITERATIVE_WITH_APRILTAG"
      },
      "apriltag_info": {
        "tag_id": 0,
        "origin_idx": 0
      }
    }
  ]
}
```

### 3. CSV结果文件（detailed_results.csv）

可以用Excel或其他表格软件打开，包含每一帧的详细数据：

| frame_id | timestamp | success | apriltag_success | roll | pitch | yaw | reprojection_error_mean |
|----------|-----------|---------|------------------|------|-------|-----|------------------------|
| frame_000001 | 1234567890.123 | True | True | 0.15 | -0.08 | 0.03 | 0.45 |
| frame_000002 | 1234567890.223 | True | True | 0.12 | -0.09 | 0.02 | 0.38 |


---

## 🔧 常见问题

### Q1: 程序提示"无法检测到网格"

**原因**：
- 标定板不在视野内或太模糊
- 光照条件不好
- 标定板尺寸设置错误

**解决方法**：
```bash
# 1. 检查图像质量
# 查看保存的失败帧图像：outputs/*/images/*_FAILED.png

# 2. 调整网格尺寸（如果不是15×15）
python robust_tilt_checker_node.py --rows 10 --cols 10 ...

# 3. 改善光照条件
# - 增加环境光
# - 避免强光直射
# - 确保标定板清晰可见
```

### Q2: AprilTag检测失败

**原因**：
- AprilTag不在视野内
- AprilTag家族设置错误
- AprilTag尺寸设置错误

**解决方法**：
```bash
# 1. 确认AprilTag家族（查看标定板上的标签）
python robust_tilt_checker_node.py --tag-family tag36h11 ...

# 2. 确认AprilTag尺寸（单位：米）
python robust_tilt_checker_node.py --tag-size 0.05 ...

# 3. 确保AprilTag清晰可见
# - AprilTag应该平整，没有褶皱
# - 光照均匀
# - 相机对焦清晰
```

### Q3: 重投影误差过高

**原因**：
- 相机内参不准确
- 标定板尺寸设置错误
- 图像畸变严重

**解决方法**：
```bash
# 1. 重新提取相机内参
python src/camera_rectifier.py \
    --camera_info_topic /camera/color/camera_info \
    --output config/camera_info.yaml

# 2. 确认标定板圆点间距（单位：米）
python robust_tilt_checker_node.py --board-spacing 0.065 ...

# 3. 调整误差阈值（如果确实需要）
python robust_tilt_checker_node.py --max-error 2.0 ...
```

### Q4: ROS2相关错误

**错误信息**：`ModuleNotFoundError: No module named 'rclpy'`

**解决方法**：
```bash
# 1. 确保ROS2环境已激活
source /opt/ros/humble/setup.bash

# 2. 安装ROS2 Python包
sudo apt install ros-humble-rclpy ros-humble-sensor-msgs ros-humble-cv-bridge

# 3. 验证安装
python -c "import rclpy; print('✅ ROS2 Python已安装')"
```

### Q5: 虚拟环境中找不到ROS2包

**原因**：虚拟环境隔离了系统包

**解决方法**：
```bash
# 方法1：创建虚拟环境时允许访问系统包
python3 -m venv .venv --system-site-packages

# 方法2：不使用虚拟环境（直接在系统Python中安装）
pip install --user -r requirements.txt
```

### Q6: 如何查看实时检测结果？

```bash
# 方法1：查看终端输出
# 程序会实时打印每一帧的检测结果

# 方法2：查看保存的图像
ls -lh outputs/*/images/

# 方法3：使用ROS2工具查看发布的话题
ros2 topic echo /tilt_checker/camera_to_board_transform
```

---

## 📁 项目结构

```
tilt_checker/
├── README.md                        # 本文档
├── requirements.txt                 # Python依赖
├── robust_tilt_checker_node.py     # 主程序（鲁棒AprilTag系统）⭐
│
├── config/                          # 配置文件
│   └── camera_info.yaml            # 相机内参（需要生成）
│
├── src/                             # 源代码
│   ├── robust_apriltag_system.py   # 鲁棒AprilTag系统
│   ├── apriltag_coordinate_system.py # AprilTag坐标系
│   ├── pnp_ambiguity_resolver.py   # PnP歧义解决器
│   ├── pose_validation.py          # 位姿验证
│   ├── detect_grid_improved.py     # 网格检测（改进版）
│   ├── estimate_tilt.py            # 倾斜角度计算
│   ├── utils.py                    # 工具函数
│   ├── camera_rectifier.py         # 相机内参提取工具
│   ├── tilt_checker_node.py        # 标准检测节点
│   └── tilt_checker_with_apriltag.py # AprilTag检测节点
│
├── outputs/                         # 输出目录（自动创建）
│   └── robust_apriltag_results/    # 检测结果
│       ├── images/                 # 可视化图像
│       ├── robust_results.json     # JSON结果
│       ├── detailed_results.csv    # CSV结果
│       └── summary_report.txt      # 统计报告
│
├── data/                            # 测试数据
│   └── *.png                       # 测试图像
│
├── rosbags/                         # rosbag文件
│   └── *.bag                       # 录制的数据
│
├── docs/                            # 文档
│   ├── QUICKSTART.md               # 快速开始
│   ├── ROS2_NODE_USAGE.md          # ROS2节点使用
│   └── camera_intrinsics_usage.md  # 相机内参指南
│
└── launch/                          # ROS2 launch文件
    └── tilt_checker.launch.py      # 启动文件
```

### 核心文件说明

| 文件 | 说明 | 何时使用 |
|------|------|----------|
| `robust_tilt_checker_node.py` | **主程序**，最鲁棒的检测方法 | ⭐ 推荐使用 |
| `src/robust_apriltag_system.py` | 鲁棒AprilTag系统核心 | 被主程序调用 |
| `src/camera_rectifier.py` | 提取相机内参 | 首次使用时 |
| `config/camera_info.yaml` | 相机内参文件 | 必需 |

---

## 🔬 技术原理

### 1. 检测流程

```
输入图像
    ↓
畸变矫正（使用相机内参）
    ↓
圆点网格检测（15×15）
    ↓
AprilTag检测和坐标系建立
    ↓
角点重新排列（基于AprilTag方向）
    ↓
鲁棒PnP求解（多种方法交叉验证）
    ↓
重投影误差计算和验证
    ↓
相机倾斜角度计算
    ↓
输出结果（角度、误差、可视化）
```

### 2. 坐标系定义

#### 相机坐标系
- **X轴**：向右
- **Y轴**：向下
- **Z轴**：向前（光轴方向）

#### 标定板坐标系（基于AprilTag）
- **原点**：离AprilTag最近的圆点
- **X轴**：沿标定板行方向
- **Y轴**：沿标定板列方向
- **Z轴**：垂直于标定板平面

### 3. 角度定义

#### Roll（横滚角）
- **定义**：相机绕X轴的旋转
- **物理意义**：相机前后倾斜
- **正值**：相机向前倾
- **负值**：相机向后仰

#### Pitch（俯仰角）
- **定义**：相机绕Z轴的旋转
- **物理意义**：相机在平面内旋转
- **正值**：顺时针旋转
- **负值**：逆时针旋转

#### Yaw（偏航角）
- **定义**：相机绕Y轴的旋转
- **物理意义**：相机左右倾斜
- **正值**：向右倾斜
- **负值**：向左倾斜

### 4. 鲁棒AprilTag系统

#### 问题：传统方法的247像素重投影误差

传统的AprilTag检测方法在某些情况下会产生高达247像素的重投影误差，原因：
- PnP求解的歧义性
- AprilTag角点检测误差
- 坐标系建立不一致

#### 解决方案：多种PnP方法交叉验证

本系统使用以下方法：

1. **ITERATIVE（迭代法）**
   - OpenCV的默认方法
   - 适合大多数情况
   - 需要良好的初始估计

2. **SQPNP（平方PnP）**
   - 更鲁棒的方法
   - 不需要初始估计
   - 计算量稍大

3. **IPPE（无限平面位姿估计）**
   - 专门用于平面物体
   - 可以处理歧义
   - 返回两个可能的解

4. **AprilTag位姿约束**
   - 使用AprilTag的位姿作为参考
   - 验证PnP解的几何一致性
   - 自动选择最优解

#### 效果

- ✅ 平均重投影误差 < 1.0 像素
- ✅ 成功率 > 95%
- ✅ 完全解决247像素误差问题

### 5. 重投影误差

#### 定义
重投影误差是3D点投影回图像平面后，与实际检测到的2D点之间的距离。

#### 计算公式
```
误差 = ||投影点 - 检测点||
平均误差 = Σ(误差) / 点数
```

#### 误差阈值
- **< 1.0 像素**：优秀
- **1.0 - 2.0 像素**：良好
- **> 2.0 像素**：需要检查

#### 自动淘汰机制
程序会自动淘汰重投影误差超过阈值的帧，确保结果质量。

---

## 🎓 使用示例

### 示例1：基础使用

```bash
# 最简单的使用方式
python robust_tilt_checker_node.py \
    --rosbag rosbags/test.bag \
    --image-topic /camera/color/image_raw \
    --camera-yaml config/camera_info.yaml
```

### 示例2：保存详细结果

```bash
# 保存图像和详细结果
python robust_tilt_checker_node.py \
    --rosbag rosbags/test.bag \
    --image-topic /camera/color/image_raw \
    --camera-yaml config/camera_info.yaml \
    --save-images \
    --output-dir outputs/detailed_analysis
```

### 示例3：快速测试（跳帧）

```bash
# 每10帧处理1帧，最多处理50帧
python robust_tilt_checker_node.py \
    --rosbag rosbags/large.bag \
    --image-topic /camera/color/image_raw \
    --camera-yaml config/camera_info.yaml \
    --skip-frames 10 \
    --max-frames 50
```

### 示例4：实时监控

```bash
# 实时监控并发布结果
python robust_tilt_checker_node.py \
    --image-topic /camera/color/image_raw \
    --camera-yaml config/camera_info.yaml \
    --publish-results \
    --save-images
```

### 示例5：自定义标定板

```bash
# 使用10×10标定板，圆点间距50mm
python robust_tilt_checker_node.py \
    --rosbag rosbags/test.bag \
    --image-topic /camera/color/image_raw \
    --camera-yaml config/camera_info.yaml \
    --rows 10 \
    --cols 10 \
    --board-spacing 0.05
```

---

## 📚 相关文档

### 详细文档
- [快速开始指南](docs/QUICKSTART.md) - 从零开始的完整教程
- [ROS2节点使用](docs/ROS2_NODE_USAGE.md) - ROS2集成详解
- [相机内参指南](docs/camera_intrinsics_usage.md) - 相机标定和内参使用

### 技术文档
- [ERROR_REJECTION_CHANGES.md](ERROR_REJECTION_CHANGES.md) - 误差淘汰机制
- [MODIFICATION_SUMMARY.md](MODIFICATION_SUMMARY.md) - 系统改进总结
- [坐标系建立分析.md](坐标系建立分析.md) - 坐标系技术细节

### 测试文档
- [HOW_TO_TEST.md](HOW_TO_TEST.md) - 测试指南
- [TEST_GUIDE.md](TEST_GUIDE.md) - 完整测试文档
- [测试说明.md](测试说明.md) - 中文测试说明

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

## 📧 联系方式

- **邮箱**: lhrlhr0808@163.com
- **问题反馈**: 请在GitHub上提交Issue

---

## 📄 许可证

[根据项目实际情况填写]

---

## 🙏 致谢

感谢所有为本项目做出贡献的开发者！

---

## 📝 更新日志

### v1.0.0 (2024-11-26)
- ✅ 实现鲁棒AprilTag系统
- ✅ 解决247像素重投影误差问题
- ✅ 支持多种PnP方法交叉验证
- ✅ 完整的ROS2集成
- ✅ 详细的可视化和统计报告

---

**祝使用愉快！如有问题，请查看[常见问题](#常见问题)或联系开发者。** 

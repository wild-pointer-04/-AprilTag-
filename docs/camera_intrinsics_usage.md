# 相机内参使用指南

本文档说明如何从 ROS2 CameraInfo 话题提取相机内参，并在本项目中使用。

## 目录结构

```
tilt_checker/
├── config/
│   └── camera_info.yaml          # 相机内参配置文件（从此文件加载）
├── src/
│   ├── camera_rectifier.py       # 从 ROS2 CameraInfo 提取内参的工具
│   ├── utils.py                  # 包含 load_camera_intrinsics() 和 get_camera_intrinsics()
│   ├── estimate_tilt.py          # 使用真实内参进行位姿估计
│   └── run_tilt_check.py         # 主程序，支持 --camera-yaml 参数
└── docs/
    └── camera_intrinsics_usage.md # 本文档
```

## 步骤 1: 从 ROS2 CameraInfo 提取相机内参

### 方法 1: 直接运行 Python 脚本

```bash
# 确保 ROS2 环境已激活
# 首先找到你的 ROS2 发行版名称（常见的有: humble, foxy, iron, jazzy）
# 查看已安装的发行版: ls /opt/ros/
source /opt/ros/humble/setup.bash  # 替换 humble 为你的发行版名称

# 运行提取工具（需要相机正在发布 CameraInfo 话题）
python src/camera_rectifier.py \
    --camera_info_topic /camera/color/camera_info \
    --output config/camera_info.yaml
```

**注意**: 如果你的 ROS2 发行版不是 `humble`，请替换为正确的名称（如 `foxy`, `iron`, `jazzy` 等）。可以通过以下命令查看：
```bash
ls /opt/ros/  # 查看已安装的 ROS2 发行版
```

### 方法 2: 作为 ROS2 节点运行

```bash
# 使用 ROS2 参数
ros2 run tilt_checker camera_rectifier \
    --ros-args \
    -p camera_info_topic:=/camera/color/camera_info \
    -p output_path:=config/camera_info.yaml
```

### 检查 CameraInfo 话题

在运行提取工具之前，确认相机正在发布 CameraInfo：

```bash
# 查看可用的话题
ros2 topic list | grep camera_info

# 查看 CameraInfo 消息内容
ros2 topic echo /camera/color/camera_info
```

### 输出文件格式

提取工具会生成 `config/camera_info.yaml` 文件，格式如下：

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

## 步骤 2: 在项目中使用相机内参

### 自动加载（推荐）

项目会自动从 `config/camera_info.yaml` 加载相机内参。如果文件不存在，会使用默认的近似内参。

```bash
# 处理单张图片（自动使用 config/camera_info.yaml）
python src/run_tilt_check.py --image data/board.png

# 批量处理（自动使用 config/camera_info.yaml）
python src/run_tilt_check.py --dir data
```

### 指定自定义 YAML 路径

```bash
# 使用自定义路径的相机内参文件
python src/run_tilt_check.py \
    --image data/board.png \
    --camera-yaml /path/to/your/camera_info.yaml
```

### 代码中使用

在 Python 代码中直接使用：

```python
from src.utils import get_camera_intrinsics, load_camera_intrinsics

# 方法1: 自动加载（优先从 YAML，失败则使用默认值）
h, w = 480, 640
K, dist = get_camera_intrinsics(h, w, yaml_path='config/camera_info.yaml')

# 方法2: 仅从 YAML 加载（失败返回 None）
K, dist, image_size = load_camera_intrinsics('config/camera_info.yaml')
if K is not None:
    print("成功加载相机内参")
else:
    print("加载失败，使用默认内参")
```

## 步骤 3: 验证相机内参

### 检查 YAML 文件内容

```bash
# 查看 YAML 文件
cat config/camera_info.yaml
```

### 检查加载日志

运行程序时，如果成功加载相机内参，会看到类似输出：

```
[INFO] 成功加载相机内参: config/camera_info.yaml
  内参矩阵 K:
[[fx   0  cx]
 [ 0  fy  cy]
 [ 0   0   1]]
  畸变系数 D: [k1 k2 p1 p2 k3]
  图像尺寸: 640 x 480
```

如果 YAML 文件不存在，会看到：

```
[WARN] 相机内参文件不存在: config/camera_info.yaml
[INFO] 使用默认内参（近似值）
```

### 图像尺寸不匹配警告

如果 YAML 中记录的图像尺寸与实际图像尺寸不匹配，会显示警告：

```
[WARN] 图像尺寸不匹配: YAML中为(640, 480), 实际为(1280, 720)
      继续使用 YAML 中的内参，但可能不够准确
```

**注意**: 如果图像尺寸变化，应该重新提取相机内参，或者使用对应尺寸的标定结果。

## 内参对结果的影响

### 使用真实内参 vs 默认内参

- **真实内参**: 
  - ✅ 更准确的位姿估计
  - ✅ 考虑镜头畸变
  - ✅ 更准确的倾斜角计算
  - ✅ 适用于精确测量

- **默认内参**:
  - ⚠️ 近似值（假设无畸变，焦距近似）
  - ⚠️ 仅适用于粗略判断是否歪斜
  - ⚠️ 不适用于精确测量

### 何时需要重新提取内参

1. **更换相机**: 不同相机的内参不同
2. **改变分辨率**: 同一相机不同分辨率的内参可能不同
3. **调整焦距**: 变焦镜头在不同焦距下的内参不同
4. **重新标定**: 如果进行了重新标定，应更新内参文件

## 故障排除

### 问题 1: 找不到 CameraInfo 话题

**症状**: 
```
[ERROR] 无法订阅话题: /camera/color/camera_info
```

**解决方案**:
1. 检查相机节点是否运行: `ros2 node list`
2. 查看可用话题: `ros2 topic list | grep camera`
3. 确认话题名称是否正确
4. 检查话题是否发布数据: `ros2 topic echo /camera/color/camera_info`

### 问题 2: YAML 文件格式错误

**症状**:
```
[ERROR] 解析 YAML 文件失败: ...
```

**解决方案**:
1. 检查 YAML 文件语法: `python -c "import yaml; yaml.safe_load(open('config/camera_info.yaml'))"`
2. 确保文件编码为 UTF-8
3. 检查缩进是否正确（YAML 对缩进敏感）

### 问题 3: 内参矩阵数据长度错误

**症状**:
```
[ERROR] 内参矩阵数据长度错误: 期望9，实际X
```

**解决方案**:
1. 检查 YAML 文件中 `camera_matrix.data` 是否有 9 个元素
2. 确保数据格式正确: `[fx, 0, cx, 0, fy, cy, 0, 0, 1]`

### 问题 4: 图像尺寸不匹配

**症状**:
```
[WARN] 图像尺寸不匹配: YAML中为(640, 480), 实际为(1280, 720)
```

**解决方案**:
1. 重新提取对应尺寸的相机内参
2. 或者使用对应尺寸的标定结果
3. 如果只是分辨率缩放（相同相机），可以手动调整内参（缩放 fx, fy, cx, cy）

## 高级用法

### 多相机支持

如果项目中使用多个相机，可以为每个相机创建独立的 YAML 文件：

```bash
config/
├── camera_info_front.yaml   # 前置相机
├── camera_info_back.yaml    # 后置相机
└── camera_info_left.yaml    # 左侧相机
```

使用时指定对应的文件：

```bash
python src/run_tilt_check.py --image data/board.png --camera-yaml config/camera_info_front.yaml
```

### 从标定工具导入

如果使用 OpenCV 的 `camera_calibration` 工具进行标定，生成的 YAML 文件格式可能略有不同。可以手动转换，或修改 `load_camera_intrinsics()` 函数以支持多种格式。

## 相关文件

- `src/camera_rectifier.py`: 从 ROS2 CameraInfo 提取内参
- `src/utils.py`: 内参加载函数
- `src/estimate_tilt.py`: 使用内参进行位姿估计
- `src/run_tilt_check.py`: 主程序

## 参考

- [OpenCV 相机标定文档](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [ROS2 CameraInfo 消息定义](http://docs.ros.org/en/api/sensor_msgs/html/msg/CameraInfo.html)
- [OpenCV FileStorage YAML 格式](https://docs.opencv.org/4.x/da/d56/classcv_1_1FileStorage.html)


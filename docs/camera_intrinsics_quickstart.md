# 相机内参快速开始指南

## 概述

本项目现在支持从 ROS2 CameraInfo 话题提取相机内参，并在倾斜检测中使用真实内参进行更准确的位姿估计。

## 快速流程

### 1. 提取相机内参（一次性操作）

```bash
# 确保 ROS2 环境已激活
# 查看已安装的 ROS2 发行版: ls /opt/ros/
source /opt/ros/humble/setup.bash  # 替换 humble 为你的发行版名称

# 确保相机正在发布 CameraInfo 话题
ros2 topic list | grep camera_info

# 运行提取工具
python src/camera_rectifier.py \
    --camera_info_topic /camera/color/camera_info \
    --output config/camera_info.yaml
```

**输出**: `config/camera_info.yaml` 文件已创建

### 2. 使用真实内参进行倾斜检测

```bash
# 处理单张图片（自动使用 config/camera_info.yaml）
python src/run_tilt_check.py --image data/board.png

# 批量处理
python src/run_tilt_check.py --dir data
```

**输出**: 程序会自动加载 `config/camera_info.yaml` 中的内参，并在日志中显示：

```
[INFO] 成功加载相机内参: config/camera_info.yaml
  内参矩阵 K:
[[fx   0  cx]
 [ 0  fy  cy]
 [ 0   0   1]]
  畸变系数 D: [k1 k2 p1 p2 k3]
```

## 项目结构

```
tilt_checker/
├── config/
│   ├── camera_info.yaml          # 相机内参配置文件（自动生成）
│   └── README.md                 # 配置目录说明
├── src/
│   ├── camera_rectifier.py       # 从 ROS2 CameraInfo 提取内参
│   ├── utils.py                  # 内参加载函数
│   ├── estimate_tilt.py          # 使用真实内参进行位姿估计
│   └── run_tilt_check.py         # 主程序（支持 --camera-yaml）
└── docs/
    ├── camera_intrinsics_usage.md # 详细使用文档
    └── camera_intrinsics_quickstart.md # 本文档
```

## 主要函数

### `load_camera_intrinsics(yaml_path)`
从 YAML 文件加载相机内参和畸变系数。

**返回**: `(K, dist, image_size)` 或 `(None, None, None)`（如果失败）

### `get_camera_intrinsics(h, w, yaml_path, f_scale)`
获取相机内参：优先从 YAML 文件加载，如果失败则使用默认值。

**返回**: `(K, dist)`

## 配置选项

### 命令行参数

- `--camera-yaml`: 指定相机内参 YAML 文件路径（默认: `config/camera_info.yaml`）

### 代码中使用

```python
from src.utils import get_camera_intrinsics

# 自动加载（优先从 YAML，失败则使用默认值）
K, dist = get_camera_intrinsics(h, w, yaml_path='config/camera_info.yaml')
```

## 验证

### 检查内参是否加载成功

运行程序时查看日志输出：

- ✅ **成功**: `[INFO] 成功加载相机内参: config/camera_info.yaml`
- ⚠️ **失败**: `[WARN] 相机内参文件不存在: config/camera_info.yaml` → 使用默认内参

### 检查内参值

```bash
# 查看 YAML 文件内容
cat config/camera_info.yaml

# 或者查看程序日志中的内参输出
```

## 故障排除

### 问题: 找不到 CameraInfo 话题

**解决**:
```bash
# 查看可用话题
ros2 topic list

# 确认相机节点是否运行
ros2 node list
```

### 问题: YAML 文件格式错误

**解决**:
```bash
# 验证 YAML 语法
python -c "import yaml; yaml.safe_load(open('config/camera_info.yaml'))"
```

### 问题: 图像尺寸不匹配

**解决**: 重新提取对应尺寸的相机内参，或使用对应尺寸的标定结果。

## 更多信息

详细文档请参考: `docs/camera_intrinsics_usage.md`


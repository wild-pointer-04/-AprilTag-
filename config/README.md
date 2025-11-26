# 相机内参配置目录

此目录用于存放相机内参配置文件 `camera_info.yaml`。

## 文件说明

- `camera_info.yaml`: 相机内参和畸变系数配置文件（OpenCV YAML 格式）

## 生成配置文件

### 方法 1: 从 ROS2 CameraInfo 话题提取（推荐）

```bash
# 确保 ROS2 环境已激活
source /opt/ros/<your_ros_distro>/setup.bash

# 运行提取工具
python src/camera_rectifier.py \
    --camera_info_topic /camera/color/camera_info \
    --output config/camera_info.yaml
```

### 方法 2: 手动创建

如果已经有相机标定结果，可以手动创建 YAML 文件。参考格式：

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
```

## 使用

项目会自动从 `config/camera_info.yaml` 加载相机内参。如果文件不存在，会使用默认的近似内参。

详细使用说明请参考: `docs/camera_intrinsics_usage.md`


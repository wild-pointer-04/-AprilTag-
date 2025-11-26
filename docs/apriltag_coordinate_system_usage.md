# 基于AprilTag的坐标系建立使用指南

## 概述

本方案通过在标定板外侧放置AprilTag二维码，建立统一的坐标系来解决相机倾斜检测中坐标轴不确定和旋转的问题。

## 方法原理

### 问题分析
1. **坐标轴不确定性**: 传统方法中，每帧图像的坐标系可能不一致
2. **旋转歧义**: 对称网格存在90°旋转歧义，导致角度计算不准确
3. **原点不固定**: 没有统一的参考点，难以建立一致的坐标系

### 解决方案
1. **AprilTag作为参考**: 在标定板外侧放置AprilTag，提供明确的方向信息
2. **统一原点**: 找到离AprilTag最近的标定板角点作为原点
3. **固定坐标轴**: X轴方向为AprilTag的正方向，Y轴垂直于X轴
4. **角点重排**: 根据新坐标系重新排列所有检测到的角点

## 硬件设置

### AprilTag放置要求
1. **位置**: 放置在标定板的最上侧（外侧），不能遮挡标定板
2. **尺寸**: 建议使用20mm x 20mm的AprilTag
3. **家族**: 推荐使用tag36h11家族（检测稳定性好）
4. **方向**: AprilTag的正方向应该与期望的X轴方向一致

### 标定板要求
1. **网格**: 支持15x15或其他尺寸的圆点网格
2. **间距**: 圆点间距通常为10mm
3. **对比度**: 确保圆点与背景有足够的对比度

## 软件安装

### 依赖库安装
```bash
# 安装Python依赖
pip install -r requirements.txt

# 如果AprilTag安装失败，尝试：
pip install apriltag-python

# 或者从源码编译：
sudo apt install cmake build-essential
pip install apriltag
```

### ROS2依赖（如果使用ROS2）
```bash
sudo apt install ros-humble-rclpy ros-humble-sensor-msgs ros-humble-cv-bridge
# 或者
rosdep install --from-paths src --ignore-src -r -y
```

## 使用方法

### 1. 测试AprilTag检测
```bash
# 测试单张图像
python test_apriltag_detection.py --image data/board1.png

# 测试数据目录中的多张图像
python test_apriltag_detection.py --data-dir data --max-images 5

# 指定参数
python test_apriltag_detection.py \
    --image data/board1.png \
    --camera-yaml config/camera_info.yaml \
    --rows 15 --cols 15 \
    --spacing 10.0 --tag-size 20.0
```

### 2. 从rosbag处理
```bash
# 使用AprilTag坐标系处理rosbag
python src/tilt_checker_with_apriltag.py \
    --rosbag rosbags/your_bag \
    --image-topic /camera/image_raw \
    --camera-yaml config/camera_info.yaml \
    --save-images \
    --output-dir outputs/apriltag_results
```

### 3. 实时处理
```bash
# 从实时话题处理
python src/tilt_checker_with_apriltag.py \
    --image-topic /camera/image_raw \
    --camera-yaml config/camera_info.yaml \
    --save-images
```

## 参数说明

### AprilTag相关参数
- `--tag-family`: AprilTag家族 (tag36h11, tag25h9, tag16h5)
- `--tag-size`: AprilTag实际尺寸(mm)，默认20.0

### 网格参数
- `--rows`: 圆点网格行数，默认15
- `--cols`: 圆点网格列数，默认15
- `--spacing`: 圆点间距(mm)，默认10.0

### 输出参数
- `--output-dir`: 输出目录
- `--save-images`: 保存可视化图像
- `--no-save-results`: 不保存结果文件

## 输出结果

### 文件结构
```
outputs/apriltag_results/
├── images/                 # 可视化图像
│   ├── frame_000001_result.png
│   └── ...
├── results.json           # 详细结果（JSON格式）
├── results.csv           # 结果表格（CSV格式）
└── summary_report.txt    # 统计报告
```

### 结果字段说明
- `apriltag_success`: AprilTag检测是否成功
- `apriltag_id`: 检测到的AprilTag ID
- `origin_idx`: 作为原点的角点索引
- `camera_tilt_angles`: 基于统一坐标系的相机倾斜角
- `reprojection_error`: 重投影误差

## 方法优势

### 1. 坐标系一致性
- 所有帧使用相同的坐标系参考
- 消除了旋转歧义问题
- 提供稳定的角度测量

### 2. 检测鲁棒性
- AprilTag检测稳定，抗干扰能力强
- 即使部分标定板被遮挡也能工作
- 支持不同光照条件

### 3. 精度提升
- 统一的坐标系减少了测量误差
- 明确的方向参考提高了角度精度
- 重投影误差更加稳定

## 注意事项

### 1. AprilTag放置
- 确保AprilTag在所有图像中都能被检测到
- AprilTag不能遮挡标定板的圆点
- 保持AprilTag平整，避免弯曲变形

### 2. 相机标定
- 使用准确的相机内参矩阵
- 确保畸变矫正参数正确
- 如果图像分辨率改变，内参会自动缩放

### 3. 环境要求
- 保证足够的光照条件
- 避免强烈的阴影或反光
- 确保AprilTag和标定板都清晰可见

## 故障排除

### AprilTag检测失败
1. 检查AprilTag是否清晰可见
2. 确认AprilTag家族设置正确
3. 调整图像对比度和亮度
4. 检查AprilTag是否有损坏或污渍

### 坐标系建立失败
1. 确认标定板检测成功
2. 检查AprilTag与标定板的相对位置
3. 验证相机内参是否正确
4. 确保AprilTag尺寸参数设置正确

### 角度测量异常
1. 检查坐标系建立是否成功
2. 验证AprilTag方向是否正确
3. 确认标定板间距参数设置
4. 检查重投影误差是否过大

## 性能优化

### 1. 处理速度
- 使用跳帧处理减少计算量
- 限制最大处理帧数
- 优化图像分辨率

### 2. 检测精度
- 使用高质量的AprilTag打印
- 确保相机标定精度
- 选择合适的AprilTag尺寸

### 3. 稳定性
- 使用多个AprilTag（如果需要）
- 实现AprilTag检测失败的回退机制
- 添加结果平滑和滤波

## 扩展功能

### 1. 多AprilTag支持
可以扩展支持多个AprilTag，提高检测的鲁棒性：
```python
# 在AprilTagCoordinateSystem类中添加多标签支持
def establish_coordinate_system_multi_tags(self, image, board_corners, ...):
    # 检测所有AprilTag
    # 选择最佳的AprilTag作为参考
    # 建立坐标系
```

### 2. 动态标定
可以实现基于AprilTag的在线相机标定：
```python
# 使用AprilTag作为标定目标
# 结合标定板进行混合标定
```

### 3. 3D坐标系
可以扩展到3D坐标系建立：
```python
# 使用AprilTag的3D位姿信息
# 建立完整的3D坐标系
```
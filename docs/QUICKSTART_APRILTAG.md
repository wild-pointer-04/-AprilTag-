# AprilTag坐标系快速开始指南

## 问题背景

在相机倾斜检测中，传统方法存在以下问题：
- **坐标轴不确定性**：每帧图像的坐标系可能不一致
- **旋转歧义**：对称网格存在90°旋转歧义
- **原点不固定**：缺乏统一的参考点

## 解决方案

使用AprilTag作为参考标记，建立统一的坐标系：
1. 在标定板外侧放置AprilTag
2. 以AprilTag方向建立坐标轴
3. 找到最近的标定板角点作为原点
4. 重新排列所有角点，消除旋转歧义

## 快速开始

### 1. 安装依赖
```bash
# 安装Python依赖
pip install opencv-python numpy pyyaml apriltag

# 如果AprilTag安装失败，尝试：
pip install apriltag-python
```

### 2. 准备硬件
- 标定板：15x15圆点网格，间距10mm
- AprilTag：20mm x 20mm，tag36h11家族
- 放置：AprilTag在标定板上方外侧

### 3. 运行演示
```bash
# 查看概念演示
python demo_apriltag_system.py

# 测试检测功能（需要真实的AprilTag图像）
python test_apriltag_detection.py --image your_image.png
```

### 4. 处理数据
```bash
# 从rosbag处理
python src/tilt_checker_with_apriltag.py \
    --rosbag your_bag \
    --image-topic /camera/image_raw \
    --save-images

# 实时处理
python src/tilt_checker_with_apriltag.py \
    --image-topic /camera/image_raw
```

## 输出结果

处理完成后，在输出目录中会生成：
- `images/`: 可视化结果图像
- `results.json`: 详细检测结果
- `results.csv`: 表格格式结果
- `summary_report.txt`: 统计报告

关键字段：
- `apriltag_success`: AprilTag检测是否成功
- `camera_tilt_angles`: 基于统一坐标系的倾斜角
- `origin_idx`: 作为原点的角点索引

## 方法优势

✅ **坐标系一致性**：所有帧使用相同参考系
✅ **消除旋转歧义**：明确的方向参考
✅ **提高精度**：统一坐标系减少测量误差
✅ **检测鲁棒性**：AprilTag抗干扰能力强

## 注意事项

⚠️ **AprilTag放置**：确保在所有图像中可见，不遮挡标定板
⚠️ **相机标定**：使用准确的内参矩阵
⚠️ **环境条件**：保证足够光照，避免强烈阴影

## 故障排除

**AprilTag检测失败**：
- 检查AprilTag是否清晰可见
- 确认家族设置正确（tag36h11）
- 调整图像对比度

**坐标系建立失败**：
- 确认标定板检测成功
- 检查AprilTag与标定板相对位置
- 验证相机内参正确性

## 下一步

1. 阅读详细文档：`docs/apriltag_coordinate_system_usage.md`
2. 查看代码实现：`src/apriltag_coordinate_system.py`
3. 根据需要调整参数和配置

---

**提示**：这个方案特别适合需要高精度、一致性测量的应用场景。通过建立统一的坐标系，可以显著提高相机倾斜检测的准确性和稳定性。
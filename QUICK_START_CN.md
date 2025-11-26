# 快速开始指南 - 5分钟上手

这是一个超级简化的指南，帮助你在5分钟内开始使用相机倾斜检测系统。

---

## 第一步：检查你有什么

你需要：
- ✅ Ubuntu电脑（20.04/22.04/24.04）
- ✅ 相机（支持ROS2的任何相机）
- ✅ 15×15圆点标定板（带AprilTag）
- ✅ ROS2已安装（Humble推荐）

---

## 第二步：安装（3分钟）

```bash
# 1. 进入项目目录
cd /path/to/tilt_checker

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装ROS2包（如果使用ROS2功能）
source /opt/ros/humble/setup.bash
sudo apt install -y ros-humble-rclpy ros-humble-sensor-msgs ros-humble-cv-bridge
```

---

## 第三步：获取相机内参（2分钟）

```bash
# 终端1：启动相机
source /opt/ros/humble/setup.bash
ros2 launch orbbec_camera gemini_330_series.launch.py  # 替换为你的相机

# 终端2：提取内参
source /opt/ros/humble/setup.bash
source .venv/bin/activate
python src/camera_rectifier.py \
    --camera_info_topic /camera/color/camera_info \
    --output config/camera_info.yaml
```

看到 `✅ 相机内参已保存` 就成功了！

---

## 第四步：运行检测（1分钟）

### 方法A：从实时相机

```bash
# 终端1：启动相机（如果还没启动）
source /opt/ros/humble/setup.bash
ros2 launch orbbec_camera gemini_330_series.launch.py

# 终端2：运行检测
source /opt/ros/humble/setup.bash
source .venv/bin/activate
python robust_tilt_checker_node.py \
    --image-topic /camera/color/image_raw \
    --camera-yaml config/camera_info.yaml \
    --save-images
```

### 方法B：从rosbag

```bash
source /opt/ros/humble/setup.bash
source .venv/bin/activate
python robust_tilt_checker_node.py \
    --rosbag /path/to/your.bag \
    --image-topic /camera/color/image_raw \
    --camera-yaml config/camera_info.yaml \
    --save-images
```

---

## 第五步：查看结果

```bash
# 查看统计报告
cat outputs/robust_apriltag_results/summary_report.txt

# 查看可视化图像
ls outputs/robust_apriltag_results/images/

# 用图像查看器打开
eog outputs/robust_apriltag_results/images/frame_000001_robust_result.png
```

---

## 结果解读

### 终端输出示例

```
[frame_000001] ✅ 正常 | 均值中心(u,v)=(424.5, 240.3) | 中心(mid)(u,v)=(425.1, 239.8) | 平均重投影误差: 0.453px
   相机倾斜角（假设板子水平，相机相对于水平面）：
      Roll(前后仰,绕X轴): +0.15°
      Pitch(平面旋,绕Z轴): -0.08°
      Yaw(左右歪,绕Y轴): +0.03°
   AprilTag ID=0, 原点索引=0
[frame_000001] 🎯 结果: ✅ 正常 | ✅ AprilTag | ✅ 低误差
```

### 角度含义

- **Roll = +0.15°**：相机向前倾斜0.15度（很小，几乎水平）
- **Pitch = -0.08°**：相机逆时针旋转0.08度（很小）
- **Yaw = +0.03°**：相机向右倾斜0.03度（很小）

**结论**：相机安装非常好！所有角度都小于0.5度。

### 可视化图像

打开 `outputs/robust_apriltag_results/images/frame_000001_robust_result.png`，你会看到：

- **黄色点**：检测到的圆点
- **绿色框**：AprilTag
- **红/绿/蓝箭头**：坐标轴（X/Y/Z）
- **左上角信息**：
  - AprilTag状态（绿色=成功）
  - 重投影误差（绿色=低误差）
  - Roll/Pitch/Yaw角度（绿色=小角度）

---

## 常见问题速查

### ❌ 问题：无法检测到网格

```bash
# 解决：检查标定板是否清晰可见
# 1. 改善光照
# 2. 确保标定板平整
# 3. 调整相机距离
```

### ❌ 问题：AprilTag检测失败

```bash
# 解决：确认AprilTag参数
python robust_tilt_checker_node.py \
    --tag-family tag36h11 \
    --tag-size 0.05 \
    ...
```

### ❌ 问题：重投影误差过高

```bash
# 解决：重新提取相机内参
python src/camera_rectifier.py \
    --camera_info_topic /camera/color/camera_info \
    --output config/camera_info.yaml
```

### ❌ 问题：找不到ROS2包

```bash
# 解决：激活ROS2环境
source /opt/ros/humble/setup.bash
```

---

## 下一步

✅ **成功运行了？** 恭喜！现在你可以：

1. **查看详细文档**：`cat README.md`
2. **调整参数**：修改标定板尺寸、误差阈值等
3. **批量处理**：处理多个rosbag
4. **集成到系统**：发布结果到ROS话题

✅ **遇到问题？** 查看：
- 完整README：`README.md`
- 常见问题：README.md的"常见问题"部分
- 技术细节：`坐标系建立分析.md`

---

## 一键命令速查表

```bash
# 安装
pip install -r requirements.txt

# 提取相机内参
python src/camera_rectifier.py --camera_info_topic /camera/color/camera_info --output config/camera_info.yaml

# 从实时相机检测
python robust_tilt_checker_node.py --image-topic /camera/color/image_raw --camera-yaml config/camera_info.yaml --save-images

# 从rosbag检测
python robust_tilt_checker_node.py --rosbag /path/to/bag --image-topic /camera/color/image_raw --camera-yaml config/camera_info.yaml --save-images

# 查看结果
cat outputs/robust_apriltag_results/summary_report.txt
```

---

**就这么简单！祝使用愉快！** 🚀

如需更多帮助，请查看完整的 `README.md` 文档。

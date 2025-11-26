# 测试指南 - 相机拍照服务

## 问题说明

如果你看到 "等待拍照服务..." 的消息，说明**拍照服务节点还没有启动**。

测试需要两个步骤：
1. **先启动服务节点**（提供服务）
2. **再运行测试脚本**（调用服务）

## 方法一：使用快速测试脚本（推荐）

这个脚本会自动启动服务并测试：

```bash
./quick_test.sh
```

这个脚本会：
1. 检查相机是否运行
2. 自动启动拍照服务
3. 运行测试
4. 询问是否保持服务运行

## 方法二：手动测试（分步骤）

### 步骤 1：启动相机

**终端 1**：
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch orbbec_camera gemini_330_series.launch.py
```

等待看到相机启动成功的消息。

### 步骤 2：启动拍照服务

**终端 2**：
```bash
source ~/ros2_ws/install/setup.bash
python3 src/camera_capture_service_node.py
```

等待看到这些消息：
```
📷 相机拍照服务节点已启动
  图像话题: /camera/color/image_raw
  服务名称: /camera_capture
  输出目录: captured_images
等待图像消息...
✅ 已接收到图像，服务就绪
🎯 服务就绪，等待拍照请求...
```

### 步骤 3：测试服务

**终端 3**：
```bash
source ~/ros2_ws/install/setup.bash
python3 test_camera_capture_service.py --single
```

预期输出：
```
[INFO] 正在连接服务: /camera_capture
[INFO] 等待拍照服务...
[INFO] ✅ 服务已就绪
[INFO] 📸 正在拍照...
[INFO] ✅ 拍照成功: 已拍好第1张照片
```

## 方法三：使用 ROS2 命令行测试

如果服务已经启动，可以直接用命令行测试：

```bash
# 查看服务是否存在
ros2 service list | grep capture

# 查看服务类型
ros2 service type /camera_capture

# 调用服务
ros2 service call /camera_capture std_srvs/srv/Trigger
```

预期输出：
```
waiting for service to become available...
requester: making request: std_srvs.srv.Trigger_Request()

response:
std_srvs.srv.Trigger_Response(success=True, message='已拍好第1张照片')
```

## 常见问题排查

### 问题 1：一直显示"等待拍照服务..."

**原因**：拍照服务节点没有启动

**解决**：
```bash
# 检查服务是否运行
ros2 service list | grep capture

# 如果没有输出，说明服务未启动
# 在另一个终端启动服务：
python3 src/camera_capture_service_node.py
```

### 问题 2：服务启动后一直显示"等待图像消息..."

**原因**：相机没有启动或图像话题名称不对

**解决**：
```bash
# 检查图像话题
ros2 topic list | grep image

# 如果没有输出，启动相机：
ros2 launch orbbec_camera gemini_330_series.launch.py

# 如果话题名称不同，使用正确的话题：
python3 src/camera_capture_service_node.py --image-topic /正确的话题名称
```

### 问题 3：sequence size exceeds remaining buffer

**原因**：这是 ROS2 的一个警告信息，通常可以忽略

**解决**：不影响功能，可以继续使用

### 问题 4：服务调用返回 "未接收到图像数据"

**原因**：相机图像未正常发布

**解决**：
```bash
# 检查图像话题是否有数据
ros2 topic hz /camera/color/image_raw

# 应该看到类似输出：
# average rate: 30.000
#   min: 0.033s max: 0.033s std dev: 0.00000s window: 30
```

## 完整测试流程示例

### 测试 1：单次拍照

```bash
# 终端 1：相机
ros2 launch orbbec_camera gemini_330_series.launch.py

# 终端 2：服务
python3 src/camera_capture_service_node.py

# 终端 3：测试
python3 test_camera_capture_service.py --single
```

### 测试 2：多次拍照（模拟机械臂）

```bash
# 终端 3：多次测试
python3 test_camera_capture_service.py --count 5 --interval 2.0
```

预期输出：
```
--- 第 1/5 次拍照 ---
🤖 模拟机械臂移动到位置 1...
📸 正在拍照...
✅ 已拍好第1张照片
等待 2.0 秒...

--- 第 2/5 次拍照 ---
🤖 模拟机械臂移动到位置 2...
📸 正在拍照...
✅ 已拍好第2张照片
等待 2.0 秒...
...
```

### 测试 3：批量快速拍照

```bash
# 终端 3：批量测试
python3 test_camera_capture_service.py --batch 10
```

## 验证结果

### 1. 检查保存的图像

```bash
# 查看保存的图像
ls -lh captured_images/

# 应该看到：
# capture_0001_20231121_143025_123.png
# capture_0002_20231121_143030_456.png
# ...
```

### 2. 查看图像内容

```bash
# 使用图像查看器
eog captured_images/capture_0001_*.png

# 或使用 OpenCV
python3 -c "import cv2; img = cv2.imread('captured_images/capture_0001_*.png'); print(f'图像尺寸: {img.shape}')"
```

### 3. 检查服务日志

在服务节点的终端应该看到：
```
📸 已拍好第1张照片
   保存路径: captured_images/capture_0001_20231121_143025_123.png
   图像尺寸: 640x480
📸 已拍好第2张照片
   保存路径: captured_images/capture_0002_20231121_143030_456.png
   图像尺寸: 640x480
...
```

## 调试技巧

### 1. 查看详细日志

```bash
# 启动服务时查看详细日志
python3 src/camera_capture_service_node.py 2>&1 | tee service.log
```

### 2. 监控 ROS2 话题

```bash
# 监控图像话题频率
ros2 topic hz /camera/color/image_raw

# 查看图像信息
ros2 topic info /camera/color/image_raw

# 查看一帧图像数据
ros2 topic echo /camera/color/image_raw --once
```

### 3. 使用 rqt 工具

```bash
# 图像查看器
ros2 run rqt_image_view rqt_image_view

# 服务调用器
ros2 run rqt_service_caller rqt_service_caller

# 日志查看器
ros2 run rqt_console rqt_console
```

## 性能测试

### 测试拍照速度

```bash
# 批量拍照 100 张，测试性能
python3 test_camera_capture_service.py --batch 100
```

预期结果：
- 平均速度：5-10 张/秒（取决于图像大小和磁盘速度）
- 成功率：100%

### 测试稳定性

```bash
# 长时间运行测试
python3 test_camera_capture_service.py --count 100 --interval 1.0
```

## 下一步

测试通过后，可以：

1. **集成到机械臂程序**
   - 参考 `docs/camera_capture_service_usage.md` 中的集成示例
   - 在机械臂到达位置后调用服务

2. **调整参数**
   - 修改图像保存目录
   - 更改图像格式（PNG/JPG）
   - 自定义服务名称

3. **添加后处理**
   - 在拍照后进行 AprilTag 检测
   - 进行图像质量检查
   - 自动上传到服务器

## 快速参考

```bash
# 一键测试（推荐）
./quick_test.sh

# 手动测试
# 终端1: ros2 launch orbbec_camera gemini_330_series.launch.py
# 终端2: python3 src/camera_capture_service_node.py
# 终端3: python3 test_camera_capture_service.py --single

# 命令行测试
ros2 service call /camera_capture std_srvs/srv/Trigger

# 查看结果
ls -lh captured_images/
```

## 获取帮助

如果遇到问题：

1. 检查所有终端的输出日志
2. 确认 ROS2 环境已正确设置
3. 验证相机和图像话题正常
4. 查看 `CAMERA_CAPTURE_SETUP.md` 中的常见问题部分

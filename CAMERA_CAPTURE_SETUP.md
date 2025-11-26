# 相机拍照服务 - 安装与配置指南

## 已创建的文件

### 核心文件
1. **src/camera_capture_service_node.py** - 拍照服务节点（主程序）
2. **launch/camera_capture_service.launch.py** - ROS2 启动文件
3. **test_camera_capture_service.py** - 测试脚本
4. **start_camera_capture_service.sh** - 快速启动脚本

### 文档文件
5. **docs/camera_capture_service_usage.md** - 详细使用文档
6. **README_CAMERA_CAPTURE_SERVICE.md** - 快速指南
7. **CAMERA_CAPTURE_SETUP.md** - 本文件（安装配置指南）

## 安装步骤

### 1. 确认依赖已安装

```bash
# 检查 ROS2 环境
echo $ROS_DISTRO

# 如果未设置，运行：
source ~/ros2_ws/install/setup.bash

# 检查 Python 依赖
pip3 list | grep -E "opencv|cv-bridge|rclpy"
```

### 2. 确认相机正常工作

```bash
# 启动相机
ros2 launch orbbec_camera gemini_330_series.launch.py

# 在另一个终端检查相机
ros2 run orbbec_camera list_devices_node

# 查看图像话题
ros2 topic list | grep image

# 查看图像（可选）
ros2 run rqt_image_view rqt_image_view
```

### 3. 测试拍照服务

```bash
# 终端1：启动拍照服务
python3 src/camera_capture_service_node.py

# 终端2：测试服务
python3 test_camera_capture_service.py --single
```

## 使用方式

### 方式一：快速启动脚本（最简单）

```bash
./start_camera_capture_service.sh
```

这个脚本提供交互式菜单，可以：
- 启动拍照服务（默认或自定义参数）
- 测试拍照服务
- 查看服务状态

### 方式二：直接运行 Python 脚本

```bash
# 使用默认参数
python3 src/camera_capture_service_node.py

# 自定义参数
python3 src/camera_capture_service_node.py \
    --image-topic /camera/color/image_raw \
    --service-name /camera_capture \
    --output-dir my_captures \
    --save-format png
```

### 方式三：使用 ROS2 Launch

```bash
# 默认参数
ros2 launch launch/camera_capture_service.launch.py

# 自定义参数
ros2 launch launch/camera_capture_service.launch.py \
    image_topic:=/camera/color/image_raw \
    output_dir:=my_captures
```

## 与机械臂集成

### 机械臂端需要做的事情

1. **创建服务客户端**
   ```python
   from std_srvs.srv import Trigger
   client = node.create_client(Trigger, '/camera_capture')
   ```

2. **在到达位置后调用服务**
   ```python
   # 移动到位置
   move_to_position(x, y, z)
   
   # 调用拍照
   request = Trigger.Request()
   future = client.call_async(request)
   rclpy.spin_until_future_complete(node, future)
   
   # 获取结果
   response = future.result()
   if response.success:
       print(response.message)  # "已拍好第1张照片"
   ```

3. **完整工作流程示例**
   ```python
   positions = [
       (100, 200, 300),
       (150, 250, 300),
       (200, 300, 300),
       # ... 更多位置
   ]
   
   for i, pos in enumerate(positions):
       # 移动机械臂
       move_to_position(*pos)
       
       # 等待稳定
       time.sleep(0.5)
       
       # 拍照
       response = capture_photo()
       if response.success:
           print(f"位置 {i+1}: {response.message}")
   ```

## 测试流程

### 1. 单次拍照测试

```bash
python3 test_camera_capture_service.py --single
```

预期输出：
```
✅ 拍照成功: 已拍好第1张照片
```

### 2. 多次拍照测试（模拟机械臂）

```bash
python3 test_camera_capture_service.py --count 5 --interval 2.0
```

这会模拟机械臂工作流程：
- 移动到位置1 → 拍照 → 等待2秒
- 移动到位置2 → 拍照 → 等待2秒
- ...

### 3. 批量快速拍照测试

```bash
python3 test_camera_capture_service.py --batch 20
```

这会快速连续拍摄20张照片，测试服务性能。

### 4. 使用 ROS2 命令行测试

```bash
# 查看服务
ros2 service list | grep capture

# 查看服务类型
ros2 service type /camera_capture

# 调用服务
ros2 service call /camera_capture std_srvs/srv/Trigger
```

## 输出文件说明

拍摄的图像保存在 `captured_images/` 目录（可自定义）：

```
captured_images/
├── capture_0001_20231121_143025_123.png
├── capture_0002_20231121_143030_456.png
├── capture_0003_20231121_143035_789.png
└── ...
```

文件名格式：
- `capture_` - 固定前缀
- `0001` - 照片序号（4位数字，从1开始）
- `20231121_143025_123` - 时间戳（年月日_时分秒_毫秒）
- `.png` - 文件格式

## 参数配置

### 图像话题名称

根据你的相机型号，图像话题可能不同：

```bash
# 查看可用话题
ros2 topic list | grep image

# 常见的话题名称：
# - /camera/color/image_raw
# - /camera/image_raw
# - /camera/rgb/image_raw
```

### 输出目录

可以指定任意目录保存图像：

```bash
python3 src/camera_capture_service_node.py --output-dir /path/to/save
```

### 图像格式

支持 PNG 和 JPG 格式：

```bash
# PNG（无损，文件较大）
python3 src/camera_capture_service_node.py --save-format png

# JPG（有损压缩，文件较小）
python3 src/camera_capture_service_node.py --save-format jpg
```

## 常见问题排查

### 问题1：服务启动后一直显示"等待图像消息..."

**原因**：图像话题名称不正确或相机未启动

**解决**：
```bash
# 1. 确认相机已启动
ros2 topic list | grep image

# 2. 使用正确的话题名称
python3 src/camera_capture_service_node.py --image-topic /正确的话题名称
```

### 问题2：调用服务返回 "未接收到图像数据"

**原因**：相机图像未正常发布

**解决**：
```bash
# 检查图像话题是否有数据
ros2 topic hz /camera/color/image_raw

# 查看图像内容
ros2 topic echo /camera/color/image_raw --once
```

### 问题3：找不到服务

**原因**：服务节点未启动或服务名称不匹配

**解决**：
```bash
# 查看所有服务
ros2 service list

# 确认服务是否存在
ros2 service list | grep capture
```

### 问题4：权限错误

**原因**：脚本没有执行权限或输出目录无写入权限

**解决**：
```bash
# 添加执行权限
chmod +x src/camera_capture_service_node.py
chmod +x test_camera_capture_service.py
chmod +x start_camera_capture_service.sh

# 创建输出目录并设置权限
mkdir -p captured_images
chmod 755 captured_images
```

## 性能说明

- **响应时间**：通常 < 100ms（取决于图像大小和磁盘速度）
- **拍照频率**：可以连续快速拍照，建议间隔 > 0.5秒以确保机械臂稳定
- **图像质量**：保存原始相机图像，无压缩（PNG）或轻微压缩（JPG）
- **内存占用**：只缓存最新一帧图像，内存占用低

## 扩展功能

如果需要在拍照后立即进行处理（如 AprilTag 检测），可以修改 `camera_capture_service_node.py` 中的 `capture_callback` 方法：

```python
def capture_callback(self, request, response):
    # ... 拍照代码 ...
    
    # 添加 AprilTag 检测
    from src.apriltag_coordinate_system import AprilTagCoordinateSystem
    coord_sys = AprilTagCoordinateSystem()
    
    # 进行检测
    success, origin, x_axis, y_axis, info = coord_sys.establish_coordinate_system(
        image_to_save, board_corners, K, dist, rows, cols
    )
    
    if success:
        response.message += f' (检测到 AprilTag ID: {info["tag_id"]})'
    
    return response
```

## 下一步

1. **测试服务**：使用测试脚本确认服务正常工作
2. **集成机械臂**：在机械臂程序中添加服务调用代码
3. **调整参数**：根据实际需求调整图像话题、保存目录等参数
4. **监控运行**：使用 `rqt_console` 或日志文件监控服务运行状态

## 相关文档

- [快速指南](README_CAMERA_CAPTURE_SERVICE.md)
- [详细使用文档](docs/camera_capture_service_usage.md)
- [ROS2 服务教程](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Services/Understanding-ROS2-Services.html)

## 技术支持

如遇到问题，请提供以下信息：
1. ROS2 版本（`echo $ROS_DISTRO`）
2. 相机型号和驱动版本
3. 错误日志（完整的终端输出）
4. 使用的命令和参数

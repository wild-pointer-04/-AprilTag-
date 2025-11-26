# 如何测试拍照服务

## 问题：为什么测试一直等待？

你遇到的问题是：**测试脚本在等待服务，但服务还没有启动**。

测试需要两个程序同时运行：
1. **服务节点**（提供拍照服务）← 这个还没启动
2. **测试脚本**（调用服务进行测试）← 这个在等待

## 最简单的测试方法

### 方法 1：一键测试（推荐）

```bash
# 确保相机已启动
ros2 launch orbbec_camera gemini_330_series.launch.py

# 然后在另一个终端运行：
./run_test.sh
```

这个脚本会自动：
- 启动服务
- 运行测试
- 显示结果
- 清理资源

### 方法 2：分步测试（理解原理）

你需要**打开 3 个终端**：

#### 终端 1：启动相机
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch orbbec_camera gemini_330_series.launch.py
```

#### 终端 2：启动拍照服务
```bash
source ~/ros2_ws/install/setup.bash
python3 src/camera_capture_service_node.py
```

等待看到：
```
✅ 已接收到图像，服务就绪
🎯 服务就绪，等待拍照请求...
```

#### 终端 3：运行测试
```bash
source ~/ros2_ws/install/setup.bash
python3 test_camera_capture_service.py --single
```

现在应该能看到：
```
✅ 拍照成功: 已拍好第1张照片
```

## 快速测试命令

如果服务已经在运行，最简单的测试方法：

```bash
ros2 service call /camera_capture std_srvs/srv/Trigger
```

## 检查清单

在测试前确认：

```bash
# ✅ 1. ROS2 环境已设置
echo $ROS_DISTRO
# 应该输出：humble 或 foxy 等

# ✅ 2. 相机已启动
ros2 topic list | grep image
# 应该看到图像话题

# ✅ 3. 服务已启动
ros2 service list | grep capture
# 应该看到：/camera_capture

# ✅ 4. 服务类型正确
ros2 service type /camera_capture
# 应该输出：std_srvs/srv/Trigger
```

## 完整测试流程图

```
┌─────────────────────────────────────────────────────────┐
│ 步骤 1: 启动相机                                         │
│ 终端1: ros2 launch orbbec_camera gemini_330_series...  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 步骤 2: 启动拍照服务                                     │
│ 终端2: python3 src/camera_capture_service_node.py      │
│                                                          │
│ 等待看到: "服务就绪，等待拍照请求..."                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 步骤 3: 测试服务                                         │
│ 终端3: python3 test_camera_capture_service.py --single │
│                                                          │
│ 或者: ros2 service call /camera_capture ...            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 步骤 4: 查看结果                                         │
│ ls captured_images/                                     │
└─────────────────────────────────────────────────────────┘
```

## 常见错误

### ❌ 错误 1：直接运行测试（没有启动服务）

```bash
# 错误做法
python3 test_camera_capture_service.py --single
# 结果：一直等待服务...
```

**正确做法**：先启动服务，再运行测试

### ❌ 错误 2：没有启动相机

```bash
# 服务启动后一直显示：等待图像消息...
```

**正确做法**：先启动相机

### ❌ 错误 3：图像话题名称不对

```bash
# 服务一直等待图像
```

**正确做法**：
```bash
# 查看正确的话题名称
ros2 topic list | grep image

# 使用正确的话题启动服务
python3 src/camera_capture_service_node.py --image-topic /正确的话题
```

## 推荐的测试顺序

### 第一次测试（验证基本功能）

```bash
# 1. 启动相机（终端1）
ros2 launch orbbec_camera gemini_330_series.launch.py

# 2. 运行一键测试（终端2）
./run_test.sh
```

### 第二次测试（模拟机械臂工作流程）

```bash
# 1. 启动相机（终端1）
ros2 launch orbbec_camera gemini_330_series.launch.py

# 2. 启动服务（终端2）
python3 src/camera_capture_service_node.py

# 3. 多次拍照测试（终端3）
python3 test_camera_capture_service.py --count 10 --interval 2.0
```

### 第三次测试（性能测试）

```bash
# 批量快速拍照
python3 test_camera_capture_service.py --batch 50
```

## 验证成功

测试成功的标志：

1. **服务端输出**（终端2）：
```
📸 已拍好第1张照片
   保存路径: captured_images/capture_0001_20231121_143025_123.png
   图像尺寸: 640x480
```

2. **测试端输出**（终端3）：
```
✅ 拍照成功: 已拍好第1张照片
```

3. **文件系统**：
```bash
$ ls captured_images/
capture_0001_20231121_143025_123.png
capture_0002_20231121_143030_456.png
...
```

## 下一步

测试成功后：

1. **集成到机械臂**
   - 参考 `docs/camera_capture_service_usage.md`
   - 在机械臂程序中调用服务

2. **调整参数**
   - 修改保存目录
   - 更改图像格式

3. **添加功能**
   - AprilTag 检测
   - 图像质量检查

## 获取更多帮助

- 详细测试指南：`TEST_GUIDE.md`
- 安装配置：`CAMERA_CAPTURE_SETUP.md`
- 使用文档：`docs/camera_capture_service_usage.md`
- 快速指南：`README_CAMERA_CAPTURE_SERVICE.md`

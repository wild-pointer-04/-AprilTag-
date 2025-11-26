# 相机拍照服务 - 快速指南

## 概述

这是一个用于与机械臂协同工作的 ROS2 拍照服务。机械臂到达指定位置后，通过调用标准服务接口 `std_srvs/srv/Trigger` 触发相机拍照。

## 快速开始

### 1. 启动相机

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch orbbec_camera gemini_330_series.launch.py
```

### 2. 启动拍照服务

**方法一：使用快速启动脚本（推荐）**

```bash
./start_camera_capture_service.sh
```

**方法二：直接运行**

```bash
python3 src/camera_capture_service_node.py
```

**方法三：使用 launch 文件**

```bash
ros2 launch launch/camera_capture_service.launch.py
```

### 3. 测试服务

```bash
# 单次拍照测试
python3 test_camera_capture_service.py --single

# 多次拍照测试（模拟机械臂工作流程）
python3 test_camera_capture_service.py --count 5 --interval 2.0

# 或使用 ros2 命令行
ros2 service call /camera_capture std_srvs/srv/Trigger
```

## 服务接口

### 请求（Request）
- 空消息（无需发送任何数据）

### 响应（Response）
- `success` (bool): 拍照是否成功
- `message` (string): 响应消息，格式为 "已拍好第i张照片"

### 示例

```bash
$ ros2 service call /camera_capture std_srvs/srv/Trigger

# 响应：
success: True
message: '已拍好第1张照片'
```

## 与机械臂集成

### Python 示例

```python
import rclpy
from std_srvs.srv import Trigger

# 初始化 ROS2
rclpy.init()
node = rclpy.create_node('robot_arm_client')

# 创建服务客户端
client = node.create_client(Trigger, '/camera_capture')
client.wait_for_service()

# 调用拍照服务
request = Trigger.Request()
future = client.call_async(request)
rclpy.spin_until_future_complete(node, future)

response = future.result()
if response.success:
    print(f'拍照成功: {response.message}')
else:
    print(f'拍照失败: {response.message}')
```

### C++ 示例

```cpp
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>

auto node = rclcpp::Node::make_shared("robot_arm_client");
auto client = node->create_client<std_srvs::srv::Trigger>("/camera_capture");

auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
auto future = client->async_send_request(request);

if (rclcpp::spin_until_future_complete(node, future) == 
    rclcpp::FutureReturnCode::SUCCESS) {
    auto response = future.get();
    if (response->success) {
        RCLCPP_INFO(node->get_logger(), "拍照成功: %s", response->message.c_str());
    }
}
```

## 文件结构

```
.
├── src/
│   └── camera_capture_service_node.py    # 拍照服务节点
├── launch/
│   └── camera_capture_service.launch.py  # Launch 文件
├── docs/
│   └── camera_capture_service_usage.md   # 详细使用文档
├── test_camera_capture_service.py        # 测试脚本
├── start_camera_capture_service.sh       # 快速启动脚本
└── README_CAMERA_CAPTURE_SERVICE.md      # 本文件
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image-topic` | `/camera/color/image_raw` | 相机图像话题 |
| `--service-name` | `/camera_capture` | 服务名称 |
| `--output-dir` | `captured_images` | 图像保存目录 |
| `--save-format` | `png` | 图像格式（png/jpg） |

## 输出文件

拍摄的图像保存格式：

```
captured_images/
├── capture_0001_20231121_143025_123.png
├── capture_0002_20231121_143030_456.png
└── capture_0003_20231121_143035_789.png
```

## 工作流程

```
机械臂移动 → 到达位置 → 调用拍照服务 → 相机拍照 → 保存图像 → 返回成功
    ↓                                                              ↓
  位置1                                                      "已拍好第1张照片"
    ↓                                                              ↓
  位置2                                                      "已拍好第2张照片"
    ↓                                                              ↓
  位置3                                                      "已拍好第3张照片"
```

## 常见问题

### 1. 服务未找到

```bash
# 检查服务是否运行
ros2 service list | grep capture

# 如果没有输出，说明服务未启动
```

### 2. 图像话题错误

```bash
# 查看可用的图像话题
ros2 topic list | grep image

# 使用正确的话题启动服务
python3 src/camera_capture_service_node.py --image-topic /camera/color/image_raw
```

### 3. 相机未连接

```bash
# 检查相机设备
ros2 run orbbec_camera list_devices_node

# 查看相机话题
ros2 topic list | grep camera
```

## 详细文档

更多详细信息请参考：
- [完整使用文档](docs/camera_capture_service_usage.md)
- [AprilTag 坐标系使用](docs/apriltag_coordinate_system_usage.md)
- [ROS2 节点使用](docs/ROS2_NODE_USAGE.md)

## 技术支持

如有问题，请检查：
1. ROS2 环境是否正确设置
2. 相机是否正常连接
3. 图像话题名称是否正确
4. 服务是否正常启动

## 许可证

与主项目相同

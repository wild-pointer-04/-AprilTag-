# 相机拍照服务使用指南

## 功能说明

相机拍照服务节点用于与机械臂协同工作，当机械臂到达指定位置后，通过调用服务触发相机拍照。

### 特性
- 使用标准 ROS2 服务接口 `std_srvs/srv/Trigger`
- 自动保存拍摄的图像
- 返回拍照成功状态和照片编号
- 支持多种图像格式（PNG、JPG）
- 线程安全的图像缓存

## 快速开始

### 1. 启动相机

```bash
# 启动 Orbbec 相机
source ~/ros2_ws/install/setup.bash
ros2 launch orbbec_camera gemini_330_series.launch.py
```

### 2. 检查相机是否正常

```bash
# 检查相机设备
ros2 run orbbec_camera list_devices_node

# 查看图像话题
ros2 topic list | grep image

# 查看图像话题信息
ros2 topic info /camera/color/image_raw

# 查看图像（可选）
ros2 run rqt_image_view rqt_image_view
```

### 3. 启动拍照服务

#### 方法一：使用 launch 文件（推荐）

```bash
# 使用默认参数
ros2 launch launch/camera_capture_service.launch.py

# 自定义参数
ros2 launch launch/camera_capture_service.launch.py \
    image_topic:=/camera/color/image_raw \
    output_dir:=my_captures \
    save_format:=png
```

#### 方法二：直接运行 Python 脚本

```bash
# 使用默认参数
python src/camera_capture_service_node.py

# 自定义参数
python src/camera_capture_service_node.py \
    --image-topic /camera/color/image_raw \
    --service-name /camera_capture \
    --output-dir captured_images \
    --save-format png
```

### 4. 测试服务

```bash
# 查看可用服务
ros2 service list | grep capture

# 查看服务类型
ros2 service type /camera_capture

# 调用拍照服务
ros2 service call /camera_capture std_srvs/srv/Trigger

# 预期输出：
# success: True
# message: '已拍好第1张照片'
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image-topic` | `/camera/color/image_raw` | 相机图像话题名称 |
| `--service-name` | `/camera_capture` | 拍照服务名称 |
| `--output-dir` | `captured_images` | 图像保存目录 |
| `--save-format` | `png` | 图像保存格式（png/jpg/jpeg） |

## 与机械臂集成

### 机械臂端调用示例（Python）

```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

class RobotArmNode(Node):
    def __init__(self):
        super().__init__('robot_arm_node')
        self.capture_client = self.create_client(Trigger, '/camera_capture')
        
    def move_and_capture(self, position):
        """移动到指定位置并拍照"""
        # 1. 移动机械臂到指定位置
        self.move_to_position(position)
        
        # 2. 等待服务可用
        while not self.capture_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待拍照服务...')
        
        # 3. 调用拍照服务
        request = Trigger.Request()
        future = self.capture_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        # 4. 处理响应
        response = future.result()
        if response.success:
            self.get_logger().info(f'拍照成功: {response.message}')
        else:
            self.get_logger().error(f'拍照失败: {response.message}')
        
        return response.success
```

### 机械臂端调用示例（C++）

```cpp
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>

class RobotArmNode : public rclcpp::Node {
public:
    RobotArmNode() : Node("robot_arm_node") {
        capture_client_ = this->create_client<std_srvs::srv::Trigger>("/camera_capture");
    }
    
    bool moveAndCapture(const Position& position) {
        // 1. 移动机械臂到指定位置
        moveToPosition(position);
        
        // 2. 等待服务可用
        while (!capture_client_->wait_for_service(std::chrono::seconds(1))) {
            RCLCPP_INFO(this->get_logger(), "等待拍照服务...");
        }
        
        // 3. 调用拍照服务
        auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
        auto future = capture_client_->async_send_request(request);
        
        // 4. 等待响应
        if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future) ==
            rclcpp::FutureReturnCode::SUCCESS) {
            auto response = future.get();
            if (response->success) {
                RCLCPP_INFO(this->get_logger(), "拍照成功: %s", response->message.c_str());
                return true;
            } else {
                RCLCPP_ERROR(this->get_logger(), "拍照失败: %s", response->message.c_str());
                return false;
            }
        }
        return false;
    }
    
private:
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr capture_client_;
};
```

## 工作流程示例

### 多点拍照流程

```bash
# 终端1：启动相机
ros2 launch orbbec_camera gemini_330_series.launch.py

# 终端2：启动拍照服务
python src/camera_capture_service_node.py

# 终端3：机械臂控制（伪代码）
# 位置1
move_robot_to_position(x=100, y=200, z=300)
ros2 service call /camera_capture std_srvs/srv/Trigger
# 输出：已拍好第1张照片

# 位置2
move_robot_to_position(x=150, y=250, z=300)
ros2 service call /camera_capture std_srvs/srv/Trigger
# 输出：已拍好第2张照片

# 位置3
move_robot_to_position(x=200, y=300, z=300)
ros2 service call /camera_capture std_srvs/srv/Trigger
# 输出：已拍好第3张照片
```

## 输出文件

拍摄的图像保存在指定的输出目录中，文件命名格式：

```
capture_0001_20231121_143025_123.png
capture_0002_20231121_143030_456.png
capture_0003_20231121_143035_789.png
```

格式说明：
- `capture_` - 固定前缀
- `0001` - 照片序号（4位数字）
- `20231121_143025_123` - 时间戳（年月日_时分秒_毫秒）
- `.png` - 文件格式

## 常见问题

### 1. 服务调用失败

**问题**：调用服务时返回 `success: False`，消息为"未接收到图像数据"

**解决方案**：
- 检查相机是否正常启动
- 检查图像话题名称是否正确
- 使用 `ros2 topic echo /camera/color/image_raw` 确认有图像数据

### 2. 图像话题名称不匹配

**问题**：节点启动后一直显示"等待图像消息..."

**解决方案**：
```bash
# 查看可用的图像话题
ros2 topic list | grep image

# 使用正确的话题名称启动服务
python src/camera_capture_service_node.py --image-topic /camera/color/image_raw
```

### 3. 权限问题

**问题**：无法创建输出目录或保存图像

**解决方案**：
```bash
# 确保有写入权限
chmod +x src/camera_capture_service_node.py
mkdir -p captured_images
chmod 755 captured_images
```

## 高级用法

### 1. 与 AprilTag 检测结合

可以在拍照后立即进行 AprilTag 检测：

```python
# 修改 capture_callback 方法
def capture_callback(self, request, response):
    # ... 拍照代码 ...
    
    # 进行 AprilTag 检测
    from src.apriltag_coordinate_system import AprilTagCoordinateSystem
    coord_sys = AprilTagCoordinateSystem()
    success, origin, x_axis, y_axis, info = coord_sys.establish_coordinate_system(
        image_to_save, board_corners, K, dist, rows, cols
    )
    
    if success:
        response.message += f' (检测到 AprilTag ID: {info["tag_id"]})'
    
    return response
```

### 2. 批量拍照脚本

```bash
#!/bin/bash
# batch_capture.sh

for i in {1..20}; do
    echo "拍摄第 $i 张照片..."
    ros2 service call /camera_capture std_srvs/srv/Trigger
    sleep 2  # 等待2秒
done
```

### 3. 监控拍照状态

```bash
# 实时查看日志
ros2 run rqt_console rqt_console

# 或使用命令行
ros2 topic echo /rosout | grep camera_capture
```

## 性能优化

1. **图像格式选择**：PNG 质量高但文件大，JPG 文件小但有损压缩
2. **缓存策略**：节点只保存最新一帧图像，减少内存占用
3. **线程安全**：使用锁保护图像缓存，避免并发问题

## 相关文档

- [ROS2 服务教程](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Services/Understanding-ROS2-Services.html)
- [Orbbec 相机文档](https://github.com/orbbec/OrbbecSDK_ROS2)
- [AprilTag 坐标系使用](./apriltag_coordinate_system_usage.md)

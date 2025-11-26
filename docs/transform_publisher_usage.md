# 相机到标定板变换参数发布功能说明

## 功能概述

本功能实现了从相机坐标系到标定板坐标系的变换参数计算和ROS话题发布。

## 变换定义

### 变换顺序

相机坐标系通过以下变换转换到标定板坐标系：

1. **平移变换**：先进行平移 (δx, δy, δz)
2. **旋转变换**：然后按ZYX顺序旋转
   - 先绕Z轴旋转 γ 度（弧度）
   - 再绕Y轴旋转 α 度（弧度）
   - 最后绕X轴旋转 β 度（弧度）

### 数学表示

变换矩阵表示为：
```
T = Translate(δx, δy, δz) * RotateZ(γ) * RotateY(α) * RotateX(β)
```

其中：
- `Translate(δx, δy, δz)` 是平移变换
- `RotateZ(γ)` 是绕Z轴旋转γ弧度
- `RotateY(α)` 是绕Y轴旋转α弧度
- `RotateX(β)` 是绕X轴旋转β弧度

### 坐标系说明

- **相机坐标系**：X右、Y下、Z前（光轴方向）
- **标定板坐标系**：基于AprilTag建立的坐标系，X轴为AprilTag正方向，Y轴垂直X轴，Z轴垂直板面向上

## 输出格式

### ROS话题消息

**话题名称**：`/tilt_checker/camera_to_board_transform`

**消息类型**：`std_msgs/Float64MultiArray`

**数据格式**：数组 `[δx, δy, δz, γ, α, β]`

- `δx, δy, δz`：平移量（单位：米）
- `γ, α, β`：ZYX欧拉角（单位：弧度）

### 消息结构

```python
Float64MultiArray:
  header:
    stamp: 时间戳
    frame_id: 'camera_frame'
  data: [δx, δy, δz, γ, α, β]
```

## 使用方法

### 1. 启用发布功能

在运行节点时添加 `--publish-results` 参数：

```bash
python robust_tilt_checker_node.py \
    --rosbag rosbags/testbag \
    --image-topic /camera/color/image_raw \
    --camera-yaml config/camera_info.yaml \
    --rows 15 --cols 15 \
    --tag-family tagStandard41h12 \
    --tag-size 0.0071 \
    --board-spacing 0.065 \
    --publish-results \
    --save-images
```

### 2. 订阅话题

在另一个终端中订阅话题：

```bash
# 查看话题列表
ros2 topic list | grep transform

# 查看消息内容
ros2 topic echo /tilt_checker/camera_to_board_transform

# 查看消息频率
ros2 topic hz /tilt_checker/camera_to_board_transform
```

### 3. 在代码中订阅

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class TransformSubscriber(Node):
    def __init__(self):
        super().__init__('transform_subscriber')
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/tilt_checker/camera_to_board_transform',
            self.transform_callback,
            10
        )
    
    def transform_callback(self, msg):
        # 提取变换参数
        delta_x, delta_y, delta_z, gamma, alpha, beta = msg.data
        
        self.get_logger().info(
            f'变换参数: 平移=[{delta_x:.4f}, {delta_y:.4f}, {delta_z:.4f}]m, '
            f'旋转=[{gamma:.4f}, {alpha:.4f}, {beta:.4f}]rad'
        )

def main():
    rclpy.init()
    node = TransformSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 代码实现说明

### 1. 欧拉角转换函数

**位置**：`src/utils.py`

**函数**：`rvec_to_euler_zyx(rvec)`

将OpenCV的旋转向量转换为ZYX欧拉角（内旋顺序）。

```python
def rvec_to_euler_zyx(rvec):
    """
    将旋转向量转换为ZYX欧拉角（内旋顺序）
    返回: (gamma, alpha, beta) - 弧度
    """
    R, _ = cv2.Rodrigues(rvec)
    alpha = np.arcsin(-R[2, 0])
    if abs(np.cos(alpha)) > 1e-6:
        gamma = np.arctan2(R[1, 0], R[0, 0])
        beta = np.arctan2(R[2, 1], R[2, 2])
    else:
        # 万向锁情况
        gamma = np.arctan2(-R[0, 1], R[1, 1])
        beta = 0.0
    return gamma, alpha, beta
```

### 2. 变换计算函数

**位置**：`src/utils.py`

**函数**：`compute_camera_to_board_transform(rvec, tvec)`

计算从相机坐标系到标定板坐标系的变换参数。

**关键步骤**：

1. **逆变换计算**：
   - OpenCV的solvePnP返回的是从标定板到相机的变换
   - 需要计算逆变换（从相机到标定板）
   - `R_cam_to_board = R_board_to_cam^T`
   - `t_cam_to_board = -R_board_to_cam^T * t_board_to_cam`

2. **欧拉角提取**：
   - 将旋转矩阵转换为ZYX欧拉角

3. **单位转换**：
   - 平移量已经是米单位（因为board_spacing使用米）
   - 角度保持弧度单位

```python
def compute_camera_to_board_transform(rvec, tvec):
    # 计算逆变换
    R_board_to_cam, _ = cv2.Rodrigues(rvec)
    R_cam_to_board = R_board_to_cam.T
    t_cam_to_board = -R_cam_to_board @ tvec
    
    # 转换为欧拉角
    rvec_cam_to_board, _ = cv2.Rodrigues(R_cam_to_board)
    gamma, alpha, beta = rvec_to_euler_zyx(rvec_cam_to_board)
    
    return delta_x, delta_y, delta_z, gamma, alpha, beta
```

### 3. ROS话题发布

**位置**：`robust_tilt_checker_node.py`

**发布器创建**：

```python
if self.publish_results:
    self.transform_publisher = self.create_publisher(
        Float64MultiArray,
        '/tilt_checker/camera_to_board_transform',
        10
    )
```

**消息发布**：

```python
# 计算变换参数
delta_x, delta_y, delta_z, gamma, alpha, beta = compute_camera_to_board_transform(
    rvec_robust, tvec_robust
)

# 构建消息
transform_msg = Float64MultiArray()
transform_msg.data = [delta_x, delta_y, delta_z, gamma, alpha, beta]
transform_msg.header.stamp = self.get_clock().now().to_msg()
transform_msg.header.frame_id = 'camera_frame'

# 发布
self.transform_publisher.publish(transform_msg)
```

## 注意事项

### 1. 单位一致性

- **平移量**：米（m）
- **角度**：弧度（rad）
- 确保 `--tag-size` 和 `--board-spacing` 参数使用米单位

### 2. 坐标系约定

- 变换是从**相机坐标系**到**标定板坐标系**
- 标定板坐标系基于AprilTag建立，原点为离AprilTag最近的角点

### 3. 万向锁问题

当绕Y轴旋转接近±90°时，会出现万向锁（gimbal lock），此时γ和β不能唯一确定。代码中处理了这种情况，选择β=0。

### 4. 发布频率

发布频率取决于图像处理频率。如果处理rosbag，发布频率等于图像帧率。

## 调试建议

### 1. 验证变换参数

可以通过以下方式验证变换参数的正确性：

```python
# 使用变换参数重建旋转矩阵
from scipy.spatial.transform import Rotation as R

# 从ZYX欧拉角重建旋转矩阵
r = R.from_euler('ZYX', [gamma, alpha, beta], degrees=False)
R_reconstructed = r.as_matrix()

# 与原始旋转矩阵比较
# R_cam_to_board 应该与 R_reconstructed 接近
```

### 2. 可视化变换

可以使用RViz或其他可视化工具查看变换关系。

### 3. 日志输出

代码中会输出详细的变换参数信息：

```
[frame_000000] 📤 已发布变换参数: 
  平移=[0.1234, -0.0567, 0.8901]m, 
  旋转=[0.0123, -0.0456, 0.0789]rad 
  (ZYX欧拉角: γ=0.70°, α=-2.61°, β=4.52°)
```

## 相关文件

- `src/utils.py`: 变换计算函数
- `robust_tilt_checker_node.py`: ROS节点和发布器
- `src/apriltag_coordinate_system.py`: 坐标系建立

## 示例输出

```
[frame_000000] 📤 已发布变换参数: 
  平移=[0.1234, -0.0567, 0.8901]m, 
  旋转=[0.0123, -0.0456, 0.0789]rad 
  (ZYX欧拉角: γ=0.70°, α=-2.61°, β=4.52°)
```

话题消息内容：
```
header:
  stamp:
    sec: 1234567890
    nanosec: 123456789
  frame_id: 'camera_frame'
data:
- 0.1234    # δx (m)
- -0.0567   # δy (m)
- 0.8901    # δz (m)
- 0.0123    # γ (rad, 绕Z轴)
- -0.0456   # α (rad, 绕Y轴)
- 0.0789    # β (rad, 绕X轴)
```


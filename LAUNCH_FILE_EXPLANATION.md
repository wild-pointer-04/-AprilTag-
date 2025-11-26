# Launch 文件说明 - camera_capture_service.launch.py

## 什么是 Launch 文件？

Launch 文件是 ROS2 的**启动配置文件**，用于：
- 一次性启动多个节点
- 配置节点参数
- 管理节点的生命周期
- 提供统一的启动接口

## camera_capture_service.launch.py 的作用

这个 launch 文件的作用是**启动拍照服务节点**，它是 `python3 src/camera_capture_service_node.py` 的**另一种启动方式**。

### 两种启动方式对比

#### 方式1：直接运行 Python 脚本
```bash
python3 src/camera_capture_service_node.py \
    --image-topic /camera/color/image_raw \
    --service-name /camera_capture \
    --output-dir captured_images \
    --save-format png
```

#### 方式2：使用 Launch 文件
```bash
ros2 launch launch/camera_capture_service.launch.py \
    image_topic:=/camera/color/image_raw \
    service_name:=/camera_capture \
    output_dir:=captured_images \
    save_format:=png
```

**它们的效果完全相同！**

## Launch 文件的优势

### 1. 更符合 ROS2 规范
```bash
# ROS2 标准方式
ros2 launch package_name launch_file.launch.py

# 更容易被其他 ROS2 工具识别和管理
```

### 2. 可以同时启动多个节点
```python
# 可以扩展为同时启动相机和拍照服务
return LaunchDescription([
    # 启动相机
    IncludeLaunchDescription(...),
    # 启动拍照服务
    camera_capture_node,
    # 启动其他节点
    ...
])
```

### 3. 参数管理更方便
```bash
# 使用 := 语法传递参数
ros2 launch launch/camera_capture_service.launch.py output_dir:=my_photos

# 比命令行参数更清晰
python3 src/camera_capture_service_node.py --output-dir my_photos
```

### 4. 可以被其他 Launch 文件包含
```python
# 在其他 launch 文件中包含这个服务
from launch.actions import IncludeLaunchDescription

IncludeLaunchDescription(
    PythonLaunchDescriptionSource('launch/camera_capture_service.launch.py'),
    launch_arguments={'output_dir': 'robot_captures'}.items()
)
```

## 实际使用场景

### 场景1：测试阶段（现在）

**推荐使用直接运行 Python 脚本**：
```bash
python3 src/camera_capture_service_node.py
```

**原因**：
- ✅ 更简单直接
- ✅ 容易看到输出和调试
- ✅ 快速启动和停止

### 场景2：生产环境（与机械臂集成后）

**推荐使用 Launch 文件**：
```bash
ros2 launch launch/camera_capture_service.launch.py
```

**原因**：
- ✅ 更规范，符合 ROS2 最佳实践
- ✅ 可以集成到系统启动流程
- ✅ 便于参数管理和版本控制

### 场景3：完整系统启动

创建一个主 launch 文件，同时启动所有组件：

```python
# launch/robot_system.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription

def generate_launch_description():
    return LaunchDescription([
        # 1. 启动相机
        IncludeLaunchDescription(
            'orbbec_camera/gemini_330_series.launch.py'
        ),
        
        # 2. 启动拍照服务
        IncludeLaunchDescription(
            'launch/camera_capture_service.launch.py',
            launch_arguments={'output_dir': 'robot_captures'}.items()
        ),
        
        # 3. 启动机械臂控制节点
        Node(
            package='robot_arm',
            executable='arm_controller',
            ...
        ),
        
        # 4. 启动其他节点...
    ])
```

然后只需一条命令启动整个系统：
```bash
ros2 launch launch/robot_system.launch.py
```

## 何时使用哪种方式？

### 使用直接运行 Python 脚本的情况：

✅ **测试和调试**
```bash
python3 src/camera_capture_service_node.py
```

✅ **快速验证功能**
```bash
python3 src/camera_capture_service_node.py --output-dir test_images
```

✅ **开发阶段**
- 容易修改和重启
- 输出更清晰
- 调试更方便

### 使用 Launch 文件的情况：

✅ **生产部署**
```bash
ros2 launch launch/camera_capture_service.launch.py
```

✅ **系统集成**
- 与其他 ROS2 节点一起启动
- 作为完整系统的一部分

✅ **自动化启动**
```bash
# 可以配置为系统服务
systemctl start robot_camera_service
```

✅ **参数配置管理**
```bash
# 使用配置文件
ros2 launch launch/camera_capture_service.launch.py \
    --launch-arguments config_file:=production.yaml
```

## 实际例子

### 例子1：开发测试（使用 Python 脚本）

```bash
# 终端1：启动相机
ros2 launch orbbec_camera gemini_330_series.launch.py

# 终端2：启动拍照服务（直接运行）
python3 src/camera_capture_service_node.py

# 终端3：测试
ros2 service call /camera_capture std_srvs/srv/Trigger
```

### 例子2：生产部署（使用 Launch 文件）

```bash
# 终端1：启动相机
ros2 launch orbbec_camera gemini_330_series.launch.py

# 终端2：启动拍照服务（使用 launch）
ros2 launch launch/camera_capture_service.launch.py \
    output_dir:=/data/robot_captures \
    save_format:=jpg

# 机械臂程序自动调用服务
```

### 例子3：完整系统（创建主 launch 文件）

创建 `launch/complete_system.launch.py`：

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    # 参数
    output_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value='/data/captures',
        description='图像保存目录'
    )
    
    return LaunchDescription([
        output_dir_arg,
        
        # 启动相机
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('orbbec_camera'),
                    'launch',
                    'gemini_330_series.launch.py'
                )
            )
        ),
        
        # 启动拍照服务
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                'launch/camera_capture_service.launch.py'
            ),
            launch_arguments={
                'output_dir': LaunchConfiguration('output_dir'),
                'save_format': 'png'
            }.items()
        ),
    ])
```

然后一条命令启动所有：
```bash
ros2 launch launch/complete_system.launch.py output_dir:=/my/captures
```

## 总结

### Launch 文件的本质
**Launch 文件只是一个启动脚本**，它的作用是：
1. 启动节点（和直接运行 Python 脚本效果一样）
2. 配置参数
3. 管理多个节点

### 你现在应该用哪个？

**测试阶段（现在）**：
```bash
# 推荐直接运行，更简单
python3 src/camera_capture_service_node.py
```

**与机械臂集成后**：
```bash
# 推荐使用 launch 文件，更规范
ros2 launch launch/camera_capture_service.launch.py
```

### 关键点

1. **功能完全相同**：两种方式启动的是同一个程序
2. **选择标准**：测试用 Python 脚本，生产用 Launch 文件
3. **可以混用**：根据实际情况选择最方便的方式

## 快速参考

```bash
# 方式1：直接运行（测试推荐）
python3 src/camera_capture_service_node.py

# 方式2：使用 launch（生产推荐）
ros2 launch launch/camera_capture_service.launch.py

# 方式3：使用 launch + 参数
ros2 launch launch/camera_capture_service.launch.py \
    image_topic:=/camera/color/image_raw \
    output_dir:=my_captures \
    save_format:=jpg

# 效果完全相同！
```

## 建议

**现在（测试阶段）**：
- 使用 `python3 src/camera_capture_service_node.py`
- 或使用测试脚本 `./simple_test.sh`

**以后（生产部署）**：
- 使用 `ros2 launch launch/camera_capture_service.launch.py`
- 或创建完整的系统 launch 文件

两种方式都可以，选择你觉得方便的！

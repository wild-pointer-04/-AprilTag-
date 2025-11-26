# ROS2 环境设置指南

## 检查 ROS2 安装

### 1. 查看已安装的 ROS2 发行版

```bash
ls /opt/ros/
```

常见输出示例：
- `humble` - ROS2 Humble Hawksbill
- `foxy` - ROS2 Foxy Fitzroy
- `iron` - ROS2 Iron Irwini
- `jazzy` - ROS2 Jazzy Jalisco

### 2. 检查 ROS2 是否可用

```bash
which ros2
```

如果输出类似 `/opt/ros/humble/bin/ros2`，说明 ROS2 已安装。

## 激活 ROS2 环境

### 方法 1: 临时激活（当前终端会话）

```bash
# 替换 <distro> 为你的发行版名称（如 humble, foxy, iron 等）
source /opt/ros/<distro>/setup.bash

# 例如，如果是 Humble:
source /opt/ros/humble/setup.bash
```

### 方法 2: 永久激活（添加到 ~/.bashrc）

```bash
# 编辑 ~/.bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# 重新加载配置
source ~/.bashrc
```

### 方法 3: 检查当前 ROS2 环境

```bash
# 查看 ROS_DISTRO 环境变量
echo $ROS_DISTRO

# 查看 ROS2 版本
ros2 --version
```

## 验证 ROS2 环境

```bash
# 检查 ROS2 命令是否可用
ros2 --help

# 查看可用的话题（需要运行 ROS2 节点）
ros2 topic list

# 查看节点列表
ros2 node list
```

## 常见问题

### 问题 1: `source /opt/ros/xxx/setup.bash: 没有那个文件或目录`

**原因**: ROS2 未安装或发行版名称错误

**解决**:
1. 检查已安装的发行版: `ls /opt/ros/`
2. 使用正确的发行版名称
3. 如果未安装，参考 [ROS2 官方安装指南](https://docs.ros.org/en/humble/Installation.html)

### 问题 2: `ros2: command not found`

**原因**: ROS2 环境未激活

**解决**:
```bash
source /opt/ros/humble/setup.bash  # 替换为你的发行版名称
```

### 问题 3: 每次打开新终端都需要激活

**解决**: 将激活命令添加到 `~/.bashrc`:
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## 本项目中的 ROS2 使用

本项目中的 `camera_rectifier.py` 需要 ROS2 环境来：
- 订阅 CameraInfo 话题
- 使用 ROS2 消息类型（sensor_msgs）
- 使用 cv_bridge 进行图像转换

**注意**: 如果只是使用倾斜检测功能（`run_tilt_check.py`），**不需要** ROS2 环境。只有在提取相机内参时才需要。

## 相关资源

- [ROS2 官方文档](https://docs.ros.org/)
- [ROS2 Humble 安装指南](https://docs.ros.org/en/humble/Installation.html)
- [ROS2 环境设置](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Configuring-ROS2-Environment.html)


#!/bin/bash
# 快速测试脚本 - 自动启动服务并测试

echo "=========================================="
echo "相机拍照服务快速测试"
echo "=========================================="
echo ""

# 检查 ROS2 环境
if [ -z "$ROS_DISTRO" ]; then
    echo "⚠️  ROS2 环境未设置，正在尝试 source..."
    if [ -f ~/ros2_ws/install/setup.bash ]; then
        source ~/ros2_ws/install/setup.bash
        echo "✅ 已加载 ROS2 环境: $ROS_DISTRO"
    else
        echo "❌ 错误：找不到 ROS2 环境"
        exit 1
    fi
fi

echo ""
echo "步骤 1/3: 检查相机是否运行..."
if ros2 topic list 2>/dev/null | grep -q "image"; then
    echo "✅ 检测到图像话题"
    ros2 topic list | grep image
else
    echo "❌ 未检测到图像话题"
    echo ""
    echo "请先在另一个终端启动相机："
    echo "  source ~/ros2_ws/install/setup.bash"
    echo "  ros2 launch orbbec_camera gemini_330_series.launch.py"
    echo ""
    read -p "相机启动后按回车继续..."
fi

echo ""
echo "步骤 2/3: 启动拍照服务..."
echo "在后台启动服务节点..."

# 在后台启动服务
python3 src/camera_capture_service_node.py &
SERVICE_PID=$!

echo "服务进程 PID: $SERVICE_PID"
echo "等待服务初始化（5秒）..."
sleep 5

# 检查服务是否启动成功
if ros2 service list 2>/dev/null | grep -q "camera_capture"; then
    echo "✅ 拍照服务已启动"
else
    echo "❌ 拍照服务启动失败"
    kill $SERVICE_PID 2>/dev/null
    exit 1
fi

echo ""
echo "步骤 3/3: 测试拍照服务..."
echo ""

# 运行测试
python3 test_camera_capture_service.py --single

TEST_RESULT=$?

echo ""
echo "=========================================="
if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ 测试通过！"
else
    echo "❌ 测试失败"
fi
echo "=========================================="
echo ""

# 询问是否继续运行服务
read -p "是否保持服务运行？(y/n) [n]: " keep_running
keep_running=${keep_running:-n}

if [ "$keep_running" = "y" ] || [ "$keep_running" = "Y" ]; then
    echo ""
    echo "服务继续运行中..."
    echo "进程 PID: $SERVICE_PID"
    echo ""
    echo "你可以继续测试："
    echo "  ros2 service call /camera_capture std_srvs/srv/Trigger"
    echo ""
    echo "按 Ctrl+C 停止服务"
    wait $SERVICE_PID
else
    echo "正在停止服务..."
    kill $SERVICE_PID 2>/dev/null
    sleep 1
    echo "服务已停止"
fi

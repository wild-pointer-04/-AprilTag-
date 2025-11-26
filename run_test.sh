#!/bin/bash
# 一键测试脚本 - 最简单的测试方式

echo "=========================================="
echo "相机拍照服务 - 一键测试"
echo "=========================================="
echo ""

# 检查相机是否运行
echo "检查相机状态..."
if ! ros2 topic list 2>/dev/null | grep -q "image"; then
    echo ""
    echo "❌ 未检测到相机图像话题"
    echo ""
    echo "请先在另一个终端启动相机："
    echo ""
    echo "  终端 1："
    echo "  source ~/ros2_ws/install/setup.bash"
    echo "  ros2 launch orbbec_camera gemini_330_series.launch.py"
    echo ""
    echo "然后再运行此脚本"
    exit 1
fi

echo "✅ 检测到图像话题："
ros2 topic list | grep image | head -3
echo ""

# 启动服务
echo "启动拍照服务..."
python3 src/camera_capture_service_node.py &
SERVICE_PID=$!

echo "等待服务初始化..."
sleep 3

# 检查服务
if ! ros2 service list 2>/dev/null | grep -q "camera_capture"; then
    echo "❌ 服务启动失败"
    kill $SERVICE_PID 2>/dev/null
    exit 1
fi

echo "✅ 服务已启动"
echo ""

# 等待服务完全就绪
echo "等待服务就绪..."
sleep 2

# 运行测试
echo "=========================================="
echo "开始测试..."
echo "=========================================="
echo ""

# 测试 1：单次拍照
echo "【测试 1】单次拍照"
ros2 service call /camera_capture std_srvs/srv/Trigger
echo ""
sleep 1

# 测试 2：再拍一张
echo "【测试 2】再拍一张"
ros2 service call /camera_capture std_srvs/srv/Trigger
echo ""
sleep 1

# 测试 3：第三张
echo "【测试 3】第三张"
ros2 service call /camera_capture std_srvs/srv/Trigger
echo ""

# 显示结果
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "保存的图像："
ls -lh captured_images/ | tail -5
echo ""

# 询问是否继续
read -p "是否继续更多测试？(y/n) [n]: " continue_test
continue_test=${continue_test:-n}

if [ "$continue_test" = "y" ] || [ "$continue_test" = "Y" ]; then
    echo ""
    echo "运行多次拍照测试（5次，间隔2秒）..."
    python3 test_camera_capture_service.py --count 5 --interval 2.0
fi

# 清理
echo ""
echo "正在停止服务..."
kill $SERVICE_PID 2>/dev/null
sleep 1

echo ""
echo "=========================================="
echo "全部完成！"
echo "=========================================="
echo ""
echo "查看拍摄的图像："
echo "  ls captured_images/"
echo ""
echo "查看图像："
echo "  eog captured_images/capture_0001_*.png"
echo ""

#!/bin/bash
# 最简单的测试脚本 - 一步一步引导

clear
echo "╔════════════════════════════════════════════════════════════╗"
echo "║        相机拍照服务 - 简单测试向导                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 步骤 1
echo "【步骤 1/4】检查相机"
echo "────────────────────────────────────────────────────────────"
echo ""

if ros2 topic list 2>/dev/null | grep -q "image"; then
    echo "✅ 相机已启动"
    echo ""
    echo "检测到的图像话题："
    ros2 topic list | grep image | head -3
    echo ""
else
    echo "❌ 相机未启动"
    echo ""
    echo "请打开一个新终端，运行以下命令启动相机："
    echo ""
    echo "┌────────────────────────────────────────────────────────┐"
    echo "│ source ~/ros2_ws/install/setup.bash                   │"
    echo "│ ros2 launch orbbec_camera gemini_330_series.launch.py │"
    echo "└────────────────────────────────────────────────────────┘"
    echo ""
    read -p "相机启动后，按回车继续..."
    
    # 再次检查
    if ! ros2 topic list 2>/dev/null | grep -q "image"; then
        echo ""
        echo "❌ 仍未检测到相机，请检查相机是否正常启动"
        exit 1
    fi
    echo ""
    echo "✅ 相机已启动"
    echo ""
fi

# 步骤 2
echo "【步骤 2/4】启动拍照服务"
echo "────────────────────────────────────────────────────────────"
echo ""
echo "正在后台启动拍照服务..."
echo ""

python3 src/camera_capture_service_node.py > /tmp/camera_service.log 2>&1 &
SERVICE_PID=$!

echo "服务进程 PID: $SERVICE_PID"
echo "等待服务初始化（5秒）..."

for i in {5..1}; do
    echo -n "$i... "
    sleep 1
done
echo ""
echo ""

# 检查服务
if ros2 service list 2>/dev/null | grep -q "camera_capture"; then
    echo "✅ 拍照服务已启动"
    echo ""
    
    # 显示服务日志的最后几行
    echo "服务状态："
    tail -3 /tmp/camera_service.log | grep -E "(已启动|就绪|接收)" || echo "  服务运行中..."
    echo ""
else
    echo "❌ 服务启动失败"
    echo ""
    echo "错误日志："
    cat /tmp/camera_service.log
    kill $SERVICE_PID 2>/dev/null
    exit 1
fi

# 步骤 3
echo "【步骤 3/4】测试拍照"
echo "────────────────────────────────────────────────────────────"
echo ""

# 测试 1
echo "测试 1: 拍第一张照片"
echo "命令: ros2 service call /camera_capture std_srvs/srv/Trigger"
echo ""
ros2 service call /camera_capture std_srvs/srv/Trigger 2>&1 | grep -A 2 "response:"
echo ""
sleep 1

# 测试 2
echo "测试 2: 拍第二张照片"
echo ""
ros2 service call /camera_capture std_srvs/srv/Trigger 2>&1 | grep -A 2 "response:"
echo ""
sleep 1

# 测试 3
echo "测试 3: 拍第三张照片"
echo ""
ros2 service call /camera_capture std_srvs/srv/Trigger 2>&1 | grep -A 2 "response:"
echo ""

# 步骤 4
echo "【步骤 4/4】查看结果"
echo "────────────────────────────────────────────────────────────"
echo ""

if [ -d "captured_images" ] && [ "$(ls -A captured_images 2>/dev/null)" ]; then
    echo "✅ 拍照成功！保存的图像："
    echo ""
    ls -lh captured_images/ | tail -5
    echo ""
    
    IMAGE_COUNT=$(ls captured_images/ | wc -l)
    echo "总共拍摄了 $IMAGE_COUNT 张照片"
    echo ""
else
    echo "❌ 未找到保存的图像"
    echo ""
fi

# 完成
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    测试完成！                               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 询问是否继续
echo "接下来你可以："
echo "  1. 继续测试（多次拍照）"
echo "  2. 查看拍摄的图像"
echo "  3. 停止服务并退出"
echo ""
read -p "请选择 [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "运行多次拍照测试（5次，间隔2秒）..."
        echo ""
        python3 test_camera_capture_service.py --count 5 --interval 2.0
        ;;
    2)
        echo ""
        echo "图像保存在: captured_images/"
        echo ""
        if command -v eog &> /dev/null; then
            echo "使用图像查看器打开..."
            eog captured_images/ &
        else
            echo "请使用你喜欢的图像查看器打开 captured_images/ 目录"
        fi
        ;;
    3)
        echo ""
        echo "好的，准备退出"
        ;;
esac

# 清理
echo ""
echo "正在停止服务..."
kill $SERVICE_PID 2>/dev/null
sleep 1

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  服务已停止。如需再次测试，重新运行此脚本即可。            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "查看完整文档："
echo "  cat HOW_TO_TEST.md"
echo ""

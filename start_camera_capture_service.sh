#!/bin/bash
# 相机拍照服务快速启动脚本

echo "=========================================="
echo "相机拍照服务启动脚本"
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
        echo "请先运行: source ~/ros2_ws/install/setup.bash"
        exit 1
    fi
else
    echo "✅ ROS2 环境已设置: $ROS_DISTRO"
fi

echo ""
echo "=========================================="
echo "启动选项："
echo "1. 启动拍照服务（默认参数）"
echo "2. 启动拍照服务（自定义参数）"
echo "3. 测试拍照服务（单次）"
echo "4. 测试拍照服务（多次）"
echo "5. 查看服务状态"
echo "6. 退出"
echo "=========================================="
echo ""

read -p "请选择 [1-6]: " choice

case $choice in
    1)
        echo ""
        echo "启动拍照服务（默认参数）..."
        echo "  图像话题: /camera/color/image_raw"
        echo "  服务名称: /camera_capture"
        echo "  输出目录: captured_images"
        echo ""
        python3 src/camera_capture_service_node.py
        ;;
    2)
        echo ""
        read -p "图像话题 [/camera/color/image_raw]: " topic
        topic=${topic:-/camera/color/image_raw}
        
        read -p "服务名称 [/camera_capture]: " service
        service=${service:-/camera_capture}
        
        read -p "输出目录 [captured_images]: " output
        output=${output:-captured_images}
        
        read -p "保存格式 [png]: " format
        format=${format:-png}
        
        echo ""
        echo "启动拍照服务..."
        echo "  图像话题: $topic"
        echo "  服务名称: $service"
        echo "  输出目录: $output"
        echo "  保存格式: $format"
        echo ""
        python3 src/camera_capture_service_node.py \
            --image-topic "$topic" \
            --service-name "$service" \
            --output-dir "$output" \
            --save-format "$format"
        ;;
    3)
        echo ""
        echo "测试拍照服务（单次）..."
        echo ""
        python3 test_camera_capture_service.py --single
        ;;
    4)
        echo ""
        read -p "拍照次数 [5]: " count
        count=${count:-5}
        
        read -p "拍照间隔（秒） [2.0]: " interval
        interval=${interval:-2.0}
        
        echo ""
        echo "测试拍照服务（多次）..."
        echo "  拍照次数: $count"
        echo "  拍照间隔: $interval 秒"
        echo ""
        python3 test_camera_capture_service.py --count "$count" --interval "$interval"
        ;;
    5)
        echo ""
        echo "查看服务状态..."
        echo ""
        echo "可用服务列表："
        ros2 service list | grep capture || echo "  未找到拍照服务"
        echo ""
        echo "服务类型："
        ros2 service type /camera_capture 2>/dev/null || echo "  服务未运行"
        echo ""
        echo "图像话题列表："
        ros2 topic list | grep image || echo "  未找到图像话题"
        echo ""
        ;;
    6)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

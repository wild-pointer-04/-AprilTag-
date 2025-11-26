#!/usr/bin/env python3
"""
相机拍照服务节点

功能：
1. 订阅相机图像话题
2. 提供拍照服务（std_srvs/srv/Trigger）
3. 接收机械臂的拍照请求，保存图像并返回结果

使用方法:
    # 启动相机
    source ~/ros2_ws/install/setup.bash
    ros2 launch orbbec_camera gemini_330_series.launch.py
    
    # 启动拍照服务节点
    python src/camera_capture_service_node.py --image-topic /camera/color/image_raw --output-dir captured_images
    
    # 测试服务调用
    ros2 service call /camera_capture std_srvs/srv/Trigger
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2
import os
import sys
from datetime import datetime
from pathlib import Path
import argparse
import threading

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class CameraCaptureServiceNode(Node):
    """相机拍照服务节点"""
    
    def __init__(self, 
                 image_topic: str = '/camera/color/image_raw',
                 service_name: str = '/camera_capture',
                 output_dir: str = 'captured_images',
                 save_format: str = 'png'):
        super().__init__('camera_capture_service_node')
        
        self.bridge = CvBridge()
        self.image_topic = image_topic
        self.service_name = service_name
        self.output_dir = output_dir
        self.save_format = save_format
        
        # 当前图像缓存
        self.current_image = None
        self.image_lock = threading.Lock()
        self.image_received = False
        
        # 拍照计数器
        self.capture_count = 0
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 订阅图像话题
        self.image_subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )
        
        # 创建拍照服务
        self.capture_service = self.create_service(
            Trigger,
            self.service_name,
            self.capture_callback
        )
        
        self.get_logger().info('='*60)
        self.get_logger().info('📷 相机拍照服务节点已启动')
        self.get_logger().info(f'  图像话题: {self.image_topic}')
        self.get_logger().info(f'  服务名称: {self.service_name}')
        self.get_logger().info(f'  输出目录: {self.output_dir}')
        self.get_logger().info(f'  保存格式: {self.save_format}')
        self.get_logger().info('='*60)
        self.get_logger().info('等待图像消息...')
    
    def image_callback(self, msg: Image):
        """接收并缓存最新的图像"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            with self.image_lock:
                self.current_image = cv_image.copy()
                if not self.image_received:
                    self.image_received = True
                    self.get_logger().info('✅ 已接收到图像，服务就绪')
                    
        except Exception as e:
            self.get_logger().error(f'处理图像消息失败: {e}')
    
    def capture_callback(self, request, response):
        """处理拍照服务请求"""
        try:
            # 检查是否有可用图像
            with self.image_lock:
                if self.current_image is None:
                    response.success = False
                    response.message = '错误：未接收到图像数据'
                    self.get_logger().warn('拍照失败：未接收到图像数据')
                    return response
                
                # 复制当前图像
                image_to_save = self.current_image.copy()
            
            # 增加拍照计数
            self.capture_count += 1
            
            # 生成文件名（带时间戳）
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f'capture_{self.capture_count:04d}_{timestamp}.{self.save_format}'
            filepath = os.path.join(self.output_dir, filename)
            
            # 保存图像
            cv2.imwrite(filepath, image_to_save)
            
            # 构建响应
            response.success = True
            response.message = f'已拍好第{self.capture_count}张照片'
            
            # 日志输出
            self.get_logger().info(f'📸 {response.message}')
            self.get_logger().info(f'   保存路径: {filepath}')
            self.get_logger().info(f'   图像尺寸: {image_to_save.shape[1]}x{image_to_save.shape[0]}')
            
            return response
            
        except Exception as e:
            response.success = False
            response.message = f'拍照失败：{str(e)}'
            self.get_logger().error(f'拍照服务异常: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return response


def main(args=None):
    """主函数"""
    parser = argparse.ArgumentParser(description='相机拍照服务节点')
    parser.add_argument('--image-topic', type=str, default='/camera/color/image_raw',
                       help='图像话题名称（默认：/camera/color/image_raw）')
    parser.add_argument('--service-name', type=str, default='/camera_capture',
                       help='服务名称（默认：/camera_capture）')
    parser.add_argument('--output-dir', type=str, default='captured_images',
                       help='图像保存目录（默认：captured_images）')
    parser.add_argument('--save-format', type=str, default='png',
                       choices=['png', 'jpg', 'jpeg'],
                       help='图像保存格式（默认：png）')
    
    # 解析参数
    if args is None:
        cli_args, _ = parser.parse_known_args()
    else:
        cli_args, _ = parser.parse_known_args(args)
    
    # 初始化 ROS2
    rclpy.init(args=args)
    
    # 创建节点
    node = CameraCaptureServiceNode(
        image_topic=cli_args.image_topic,
        service_name=cli_args.service_name,
        output_dir=cli_args.output_dir,
        save_format=cli_args.save_format
    )
    
    try:
        node.get_logger().info('🎯 服务就绪，等待拍照请求...')
        node.get_logger().info('按 Ctrl+C 退出')
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('接收到中断信号，正在关闭...')
    finally:
        node.get_logger().info(f'总共拍摄了 {node.capture_count} 张照片')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

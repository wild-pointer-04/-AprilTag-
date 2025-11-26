#!/usr/bin/env python3
"""
从 ROS2 CameraInfo 话题读取相机内参并保存为 OpenCV 兼容的 YAML 文件。

使用方法:
    python src/camera_rectifier.py --camera_info_topic /camera/color/camera_info --output config/camera_info.yaml

或者作为 ROS2 节点运行:
    ros2 run tilt_checker camera_rectifier --ros-args -p camera_info_topic:=/camera/color/camera_info -p output_path:=config/camera_info.yaml
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
import cv2
import yaml
import numpy as np
import os
import argparse


class CameraInfoExtractor(Node):
    """从 ROS2 CameraInfo 话题提取相机内参并保存为 YAML 文件"""

    def __init__(self, camera_info_topic: str = None, 
                 output_path: str = None,
                 save_image: bool = None):
        super().__init__('camera_info_extractor')
        self.bridge = CvBridge()
        self.K = None
        self.D = None
        self.width = None
        self.height = None
        self.distortion_model = None
        self.saved_yaml = False
        self.image_pub = None
        self.image_sub = None
        
        # 声明 ROS2 参数（如果作为节点运行）
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('output_path', 'config/camera_info.yaml')
        self.declare_parameter('save_image', False)
        
        # 获取参数值（优先使用传入参数，否则使用 ROS2 参数或默认值）
        if camera_info_topic is None:
            camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        if output_path is None:
            output_path = self.get_parameter('output_path').get_parameter_value().string_value
        if save_image is None:
            save_image = self.get_parameter('save_image').get_parameter_value().bool_value
        
        self.output_path = output_path
        self.save_image = save_image
        self.camera_info_topic = camera_info_topic

        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            self.get_logger().info(f'创建输出目录: {output_dir}')

        # 订阅 CameraInfo 话题
        self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            10
        )
        self.get_logger().info(f'订阅 CameraInfo 话题: {camera_info_topic}')

        # 如果需要保存图像（用于验证）
        if save_image:
            image_topic = camera_info_topic.replace('/camera_info', '/image_raw')
            if image_topic == camera_info_topic:
                # 如果替换失败，尝试常见的话题名称
                image_topic = '/camera/color/image_raw'
            self.create_subscription(
                Image,
                image_topic,
                self.image_callback,
                10
            )
            rect_topic = camera_info_topic.replace('/camera_info', '/image_rect')
            if rect_topic == camera_info_topic:
                rect_topic = '/camera/color/image_rect'
            self.image_pub = self.create_publisher(
                Image,
                rect_topic,
                10
            )
            self.get_logger().info(f'订阅图像话题: {image_topic}')
            self.get_logger().info(f'发布矫正图像话题: {rect_topic}')

    def camera_info_callback(self, msg: CameraInfo):
        """处理 CameraInfo 消息，提取内参并保存为 YAML"""
        if self.K is None:
            # 内参矩阵 K (3x3)
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            # 畸变系数 D
            self.D = np.array(msg.d, dtype=np.float64)
            # 图像尺寸
            self.width = msg.width
            self.height = msg.height
            # 畸变模型
            self.distortion_model = msg.distortion_model

            self.get_logger().info('=' * 60)
            self.get_logger().info('接收到 CameraInfo:')
            self.get_logger().info(f'  图像尺寸: {self.width} x {self.height}')
            self.get_logger().info(f'  畸变模型: {self.distortion_model}')
            self.get_logger().info(f'  内参矩阵 K:\n{self.K}')
            self.get_logger().info(f'  畸变系数 D: {self.D}')
            self.get_logger().info('=' * 60)

            # 保存为 OpenCV 兼容的 YAML 格式
            if not self.saved_yaml:
                self.save_camera_info_yaml(msg)
                self.saved_yaml = True

    def save_camera_info_yaml(self, msg: CameraInfo):
        """保存相机内参为 OpenCV 兼容的 YAML 格式"""
        # OpenCV YAML 格式（兼容 cv::FileStorage）
        camera_info_dict = {
            # 基本信息
            'image_width': int(msg.width),
            'image_height': int(msg.height),
            'camera_name': 'camera',
            'distortion_model': msg.distortion_model,
            
            # 内参矩阵 (3x3)
            'camera_matrix': {
                'rows': 3,
                'cols': 3,
                'dt': 'd',  # 数据类型: double
                'data': [float(x) for x in msg.k]  # [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            },
            
            # 畸变系数
            'distortion_coefficients': {
                'rows': 1,
                'cols': len(msg.d),
                'dt': 'd',  # 数据类型: double
                'data': [float(x) for x in msg.d]
            },
            
            # 矫正矩阵 (可选)
            'rectification_matrix': {
                'rows': 3,
                'cols': 3,
                'dt': 'd',
                'data': [float(x) for x in msg.r]
            },
            
            # 投影矩阵 (可选)
            'projection_matrix': {
                'rows': 3,
                'cols': 4,
                'dt': 'd',
                'data': [float(x) for x in msg.p]
            }
        }

        # 保存 YAML 文件
        try:
            with open(self.output_path, 'w') as f:
                yaml.dump(camera_info_dict, f, default_flow_style=False, sort_keys=False)
            self.get_logger().info(f'✅ 相机内参已保存到: {self.output_path}')
            self.get_logger().info(f'   文件路径: {os.path.abspath(self.output_path)}')
        except Exception as e:
            self.get_logger().error(f'❌ 保存 YAML 文件失败: {e}')

    def image_callback(self, msg: Image):
        """处理图像消息（用于验证矫正效果）"""
        if self.K is None or self.D is None:
            return

        try:
            # 转换 ROS Image 到 OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # 畸变矫正
            undistorted = cv2.undistort(cv_image, self.K, self.D)
            # 发布矫正后的图像
            rect_msg = self.bridge.cv2_to_imgmsg(undistorted, encoding='bgr8')
            rect_msg.header = msg.header
            self.image_pub.publish(rect_msg)
        except Exception as e:
            self.get_logger().warn(f'图像处理失败: {e}')


def main(args=None):
    """主函数：支持命令行参数和 ROS2 参数"""
    import sys
    
    # 检查是否作为 ROS2 节点运行（有 --ros-args）
    is_ros_node = '--ros-args' in sys.argv
    
    # 解析命令行参数（如果不是 ROS2 节点模式）
    camera_info_topic = None
    output_path = None
    save_image = None
    
    if not is_ros_node:
        # 直接运行 Python 脚本，解析命令行参数
        parser = argparse.ArgumentParser(description='从 ROS2 CameraInfo 提取相机内参')
        parser.add_argument('--camera_info_topic', type=str, 
                           default='/camera/color/camera_info',
                           help='CameraInfo 话题名称')
        parser.add_argument('--output', type=str,
                           default='config/camera_info.yaml',
                           help='输出 YAML 文件路径')
        parser.add_argument('--save_image', action='store_true',
                           help='是否发布矫正后的图像（用于验证）')
        
        cli_args, _ = parser.parse_known_args()
        camera_info_topic = cli_args.camera_info_topic
        output_path = cli_args.output
        save_image = cli_args.save_image
    
    # 初始化 ROS2
    rclpy.init(args=args)
    
    # 创建节点（参数会在节点内部处理）
    node = CameraInfoExtractor(
        camera_info_topic=camera_info_topic,
        output_path=output_path,
        save_image=save_image
    )
    
    # 输出配置信息
    node.get_logger().info('=' * 60)
    node.get_logger().info('相机内参提取工具')
    node.get_logger().info(f'  CameraInfo 话题: {node.camera_info_topic}')
    node.get_logger().info(f'  输出文件路径: {node.output_path}')
    node.get_logger().info(f'  保存图像: {node.save_image}')
    node.get_logger().info('=' * 60)

    try:
        node.get_logger().info('等待 CameraInfo 消息...')
        node.get_logger().info('按 Ctrl+C 退出')
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('接收到中断信号，退出...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

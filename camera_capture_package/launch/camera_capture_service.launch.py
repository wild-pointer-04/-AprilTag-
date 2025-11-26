#!/usr/bin/env python3
"""
相机拍照服务启动文件

使用方法:
    ros2 launch launch/camera_capture_service.launch.py
    
    # 自定义参数
    ros2 launch launch/camera_capture_service.launch.py image_topic:=/camera/color/image_raw output_dir:=my_captures
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    # 获取项目根目录
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 声明启动参数
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/color/image_raw',
        description='相机图像话题名称'
    )
    
    service_name_arg = DeclareLaunchArgument(
        'service_name',
        default_value='/camera_capture',
        description='拍照服务名称'
    )
    
    output_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value='captured_images',
        description='图像保存目录'
    )
    
    save_format_arg = DeclareLaunchArgument(
        'save_format',
        default_value='png',
        description='图像保存格式 (png/jpg/jpeg)'
    )
    
    # 相机拍照服务节点
    camera_capture_node = ExecuteProcess(
        cmd=[
            'python3',
            os.path.join(package_dir, 'src', 'camera_capture_service_node.py'),
            '--image-topic', LaunchConfiguration('image_topic'),
            '--service-name', LaunchConfiguration('service_name'),
            '--output-dir', LaunchConfiguration('output_dir'),
            '--save-format', LaunchConfiguration('save_format')
        ],
        output='screen'
    )
    
    return LaunchDescription([
        image_topic_arg,
        service_name_arg,
        output_dir_arg,
        save_format_arg,
        camera_capture_node
    ])

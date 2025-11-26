#!/usr/bin/env python3
"""
完整系统启动文件 - 同时启动相机和拍照服务

这个 launch 文件演示了如何一次性启动整个系统。
在实际使用中，你可以根据需要添加更多节点（如机械臂控制节点）。

使用方法:
    # 基本用法
    ros2 launch launch/complete_system.launch.py
    
    # 自定义参数
    ros2 launch launch/complete_system.launch.py \
        output_dir:=/data/robot_captures \
        save_format:=jpg
    
    # 使用不同的图像话题
    ros2 launch launch/complete_system.launch.py \
        image_topic:=/camera/rgb/image_raw

优势：
    - 一条命令启动所有组件
    - 统一的参数管理
    - 便于系统部署和维护
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import LaunchConfiguration
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
    
    # 启动信息
    start_info = LogInfo(
        msg=[
            '\n',
            '='*60, '\n',
            '完整系统启动中...\n',
            '='*60, '\n',
            '组件:\n',
            '  1. 相机驱动 (需要手动启动)\n',
            '  2. 拍照服务\n',
            '\n',
            '参数:\n',
            '  图像话题: ', LaunchConfiguration('image_topic'), '\n',
            '  服务名称: ', LaunchConfiguration('service_name'), '\n',
            '  输出目录: ', LaunchConfiguration('output_dir'), '\n',
            '  保存格式: ', LaunchConfiguration('save_format'), '\n',
            '='*60, '\n'
        ]
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
        output='screen',
        name='camera_capture_service'
    )
    
    # 注意：相机驱动需要单独启动
    # 如果你的相机驱动也有 launch 文件，可以在这里添加：
    #
    # camera_driver = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         '/path/to/orbbec_camera/launch/gemini_330_series.launch.py'
    #     )
    # )
    #
    # 然后在 LaunchDescription 中添加 camera_driver
    
    return LaunchDescription([
        # 参数声明
        image_topic_arg,
        service_name_arg,
        output_dir_arg,
        save_format_arg,
        
        # 启动信息
        start_info,
        
        # 节点
        camera_capture_node,
        
        # 如果需要启动相机，在这里添加
        # camera_driver,
        
        # 如果需要启动机械臂控制节点，在这里添加
        # robot_arm_node,
    ])

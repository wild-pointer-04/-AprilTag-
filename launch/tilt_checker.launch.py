#!/usr/bin/env python3
"""
相机倾斜检测 Launch 文件

使用方法:
    ros2 launch tilt_checker tilt_checker.launch.py \
        image_topic:=/camera/image_raw \
        camera_yaml:=config/camera_info.yaml \
        rows:=15 \
        cols:=15
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    # 获取包路径
    pkg_share = FindPackageShare('tilt_checker').find('tilt_checker')
    
    # 声明启动参数
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/image_raw',
        description='图像话题名称'
    )
    
    camera_yaml_arg = DeclareLaunchArgument(
        'camera_yaml',
        default_value=os.path.join(pkg_share, 'config', 'camera_info.yaml'),
        description='相机内参 YAML 文件路径'
    )
    
    rows_arg = DeclareLaunchArgument(
        'rows',
        default_value='15',
        description='圆点行数'
    )
    
    cols_arg = DeclareLaunchArgument(
        'cols',
        default_value='15',
        description='圆点列数'
    )
    
    spacing_arg = DeclareLaunchArgument(
        'spacing',
        default_value='10.0',
        description='圆点间距（mm）'
    )
    
    output_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value='outputs/rosbag_results',
        description='输出目录'
    )
    
    save_images_arg = DeclareLaunchArgument(
        'save_images',
        default_value='false',
        description='是否保存检测结果图像'
    )
    
    # 创建节点
    tilt_checker_node = Node(
        package='tilt_checker',
        executable='tilt_checker_node',
        name='tilt_checker_node',
        parameters=[{
            'image_topic': LaunchConfiguration('image_topic'),
            'camera_yaml': LaunchConfiguration('camera_yaml'),
            'rows': LaunchConfiguration('rows'),
            'cols': LaunchConfiguration('cols'),
            'spacing': LaunchConfiguration('spacing'),
            'output_dir': LaunchConfiguration('output_dir'),
            'save_images': LaunchConfiguration('save_images'),
        }],
        output='screen'
    )
    
    return LaunchDescription([
        image_topic_arg,
        camera_yaml_arg,
        rows_arg,
        cols_arg,
        spacing_arg,
        output_dir_arg,
        save_images_arg,
        LogInfo(msg=['启动相机倾斜检测节点...']),
        tilt_checker_node,
    ])


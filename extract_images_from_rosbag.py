#!/usr/bin/env python3
"""
从rosbag中提取图像，用于检查AprilTag是否可见
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from rclpy.serialization import deserialize_message
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    ROSBAG_AVAILABLE = True
except ImportError:
    ROSBAG_AVAILABLE = False
    print("⚠️ rosbag2_py 不可用，请安装: sudo apt install ros-humble-rosbag2-py")


def extract_images_from_rosbag(bag_path: str, 
                              image_topic: str = '/camera/color/image_raw',
                              output_dir: str = 'extracted_images',
                              max_images: int = 10,
                              skip_frames: int = 10):
    """
    从rosbag中提取图像
    
    Args:
        bag_path: rosbag路径
        image_topic: 图像话题
        output_dir: 输出目录
        max_images: 最大提取图像数
        skip_frames: 跳帧数（每隔多少帧提取一张）
    """
    if not ROSBAG_AVAILABLE:
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    bridge = CvBridge()
    
    try:
        storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        
        reader = SequentialReader()
        reader.open(storage_options, converter_options)
        
        topic_types = reader.get_all_topics_and_types()
        
        # 检查话题是否存在
        image_topic_found = False
        for topic_metadata in topic_types:
            if topic_metadata.name == image_topic:
                image_topic_found = True
                break
        
        if not image_topic_found:
            print(f"❌ 在rosbag中未找到话题: {image_topic}")
            print(f"可用话题: {[t.name for t in topic_types]}")
            return False
        
        print(f"开始从rosbag提取图像...")
        print(f"  rosbag: {bag_path}")
        print(f"  话题: {image_topic}")
        print(f"  输出目录: {output_dir}")
        print(f"  最大图像数: {max_images}")
        print(f"  跳帧数: {skip_frames}")
        
        frame_idx = 0
        extracted_count = 0
        
        while reader.has_next() and extracted_count < max_images:
            (topic, data, timestamp) = reader.read_next()
            
            if topic == image_topic:
                # 跳帧逻辑
                if frame_idx % skip_frames != 0:
                    frame_idx += 1
                    continue
                
                try:
                    # 反序列化图像消息
                    msg_type = None
                    for topic_metadata in topic_types:
                        if topic_metadata.name == image_topic:
                            msg_type = topic_metadata.type
                            break
                    
                    if msg_type == 'sensor_msgs/msg/Image':
                        msg = deserialize_message(data, Image)
                        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    else:
                        if isinstance(data, Image):
                            cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
                        else:
                            print(f"⚠️ 未知的消息类型: {msg_type}")
                            frame_idx += 1
                            continue
                    
                    # 保存图像
                    image_filename = f'frame_{frame_idx:06d}.png'
                    image_path = os.path.join(output_dir, image_filename)
                    cv2.imwrite(image_path, cv_image)
                    
                    print(f"✅ 已提取: {image_filename} ({cv_image.shape[1]}x{cv_image.shape[0]})")
                    
                    extracted_count += 1
                    frame_idx += 1
                    
                except Exception as e:
                    print(f"⚠️ 处理帧失败: {e}")
                    frame_idx += 1
                    continue
            
        reader = None
        print(f"✅ 提取完成，共提取 {extracted_count} 张图像")
        return True
        
    except Exception as e:
        print(f"❌ 提取失败: {e}")
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='从rosbag中提取图像')
    parser.add_argument('--rosbag', type=str, required=True, help='rosbag路径')
    parser.add_argument('--image-topic', type=str, default='/camera/color/image_raw',
                       help='图像话题')
    parser.add_argument('--output-dir', type=str, default='extracted_images',
                       help='输出目录')
    parser.add_argument('--max-images', type=int, default=10,
                       help='最大提取图像数')
    parser.add_argument('--skip-frames', type=int, default=10,
                       help='跳帧数')
    
    args = parser.parse_args()
    
    success = extract_images_from_rosbag(
        args.rosbag,
        args.image_topic,
        args.output_dir,
        args.max_images,
        args.skip_frames
    )
    
    if success:
        print(f"\n请检查 {args.output_dir} 目录中的图像，确认AprilTag是否清晰可见")
        print("如果AprilTag可见，可以使用以下命令测试检测:")
        print(f"python test_apriltag_detection.py --image {args.output_dir}/frame_000000.png")


if __name__ == '__main__':
    main()
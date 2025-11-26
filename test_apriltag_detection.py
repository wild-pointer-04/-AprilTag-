#!/usr/bin/env python3
"""
AprilTag检测测试脚本

测试AprilTag坐标系建立功能，可以用于验证方法的正确性
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from src.apriltag_coordinate_system import AprilTagCoordinateSystem
from src.detect_grid_improved import try_find_adaptive, refine
from src.utils import load_camera_intrinsics, get_camera_intrinsics


def test_single_image(image_path: str, 
                     camera_yaml: str = 'config/camera_info.yaml',
                     rows: int = 15,
                     cols: int = 15,
                     spacing: float = 10.0,
                     tag_size: float = 20.0,
                     output_path: str = None):
    """
    测试单张图像的AprilTag坐标系建立
    
    Args:
        image_path: 图像文件路径
        camera_yaml: 相机内参文件路径
        rows: 网格行数
        cols: 网格列数
        spacing: 圆点间距(mm)
        tag_size: AprilTag尺寸(mm)
        output_path: 输出图像路径（可选）
    """
    print(f"测试图像: {image_path}")
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return False
    
    h, w = image.shape[:2]
    print(f"图像尺寸: {w} x {h}")
    
    # 加载相机内参
    try:
        K, dist, image_size = load_camera_intrinsics(camera_yaml)
        if K is None:
            print("⚠️ 使用默认相机内参")
            K, dist = get_camera_intrinsics(h, w)
        else:
            print(f"✅ 已加载相机内参")
    except Exception as e:
        print(f"⚠️ 加载内参失败，使用默认值: {e}")
        K, dist = get_camera_intrinsics(h, w)
    
    # 畸变矫正
    undistorted = cv2.undistort(image, K, dist)
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    
    # 检测圆点网格
    print("检测圆点网格...")
    try:
        ok, corners, blob_keypoints = try_find_adaptive(gray, rows, cols, symmetric=True)
        
        if not ok or corners is None:
            print("❌ 未检测到圆点网格")
            return False
        
        # 精化角点
        corners_refined = refine(gray, corners)
        board_corners = corners_refined.reshape(-1, 2)
        print(f"✅ 检测到 {len(board_corners)} 个圆点")
        
    except Exception as e:
        print(f"❌ 圆点检测失败: {e}")
        return False
    
    # 初始化AprilTag坐标系建立器
    print("初始化AprilTag坐标系建立器...")
    coord_system = AprilTagCoordinateSystem(
        tag_family='tag36h11',
        tag_size=tag_size,
        board_spacing=spacing
    )
    
    # 建立坐标系
    print("建立AprilTag坐标系...")
    try:
        success, origin, x_direction, y_direction, info = coord_system.establish_coordinate_system(
            undistorted, board_corners, K, dist, rows, cols
        )
        
        if not success:
            print("❌ AprilTag坐标系建立失败")
            return False
        
        print(f"✅ AprilTag坐标系建立成功!")
        print(f"  AprilTag ID: {info['tag_id']}")
        print(f"  原点坐标: ({origin[0]:.1f}, {origin[1]:.1f})")
        print(f"  原点角点索引: {info['origin_idx']}")
        print(f"  X轴方向: ({x_direction[0]:.3f}, {x_direction[1]:.3f})")
        print(f"  Y轴方向: ({y_direction[0]:.3f}, {y_direction[1]:.3f})")
        
        # 可视化结果
        vis_image = coord_system.visualize_coordinate_system(undistorted, info)
        
        # 添加额外的信息文本
        info_text = [
            f"AprilTag ID: {info['tag_id']}",
            f"Origin Index: {info['origin_idx']}",
            f"Origin: ({origin[0]:.1f}, {origin[1]:.1f})",
            f"Grid Points: {len(board_corners)}",
            f"Reordered: {len(info['reordered_corners'])}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(vis_image, text, (10, h - 150 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_image, text, (10, h - 150 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # 保存或显示结果
        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"✅ 结果已保存到: {output_path}")
        else:
            # 显示结果
            cv2.imshow('AprilTag Coordinate System', vis_image)
            print("按任意键继续...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return True
        
    except Exception as e:
        print(f"❌ AprilTag处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_directory(data_dir: str = 'data',
                       camera_yaml: str = 'config/camera_info.yaml',
                       output_dir: str = 'outputs/apriltag_test',
                       max_images: int = 5):
    """
    测试数据目录中的多张图像
    
    Args:
        data_dir: 数据目录路径
        camera_yaml: 相机内参文件路径
        output_dir: 输出目录
        max_images: 最大测试图像数量
    """
    print(f"测试数据目录: {data_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找图像文件
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(data_dir).glob(f'*{ext}'))
        image_files.extend(Path(data_dir).glob(f'*{ext.upper()}'))
    
    image_files = sorted(image_files)[:max_images]
    
    if not image_files:
        print(f"❌ 在 {data_dir} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件，开始测试...")
    
    success_count = 0
    total_count = len(image_files)
    
    for i, image_file in enumerate(image_files):
        print(f"\n--- 测试 {i+1}/{total_count}: {image_file.name} ---")
        
        output_path = os.path.join(output_dir, f'result_{image_file.stem}.png')
        
        try:
            success = test_single_image(
                str(image_file),
                camera_yaml=camera_yaml,
                output_path=output_path
            )
            
            if success:
                success_count += 1
                print(f"✅ 测试成功")
            else:
                print(f"❌ 测试失败")
                
        except Exception as e:
            print(f"❌ 测试异常: {e}")
    
    print(f"\n=== 测试总结 ===")
    print(f"总测试数: {total_count}")
    print(f"成功数: {success_count}")
    print(f"失败数: {total_count - success_count}")
    print(f"成功率: {success_count / total_count * 100:.1f}%")
    print(f"结果保存在: {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AprilTag坐标系建立测试')
    parser.add_argument('--image', type=str, help='单张图像文件路径')
    parser.add_argument('--data-dir', type=str, default='data', help='数据目录路径')
    parser.add_argument('--camera-yaml', type=str, default='config/camera_info.yaml',
                       help='相机内参文件路径')
    parser.add_argument('--output-dir', type=str, default='outputs/apriltag_test',
                       help='输出目录')
    parser.add_argument('--rows', type=int, default=15, help='网格行数')
    parser.add_argument('--cols', type=int, default=15, help='网格列数')
    parser.add_argument('--spacing', type=float, default=10.0, help='圆点间距(mm)')
    parser.add_argument('--tag-size', type=float, default=20.0, help='AprilTag尺寸(mm)')
    parser.add_argument('--max-images', type=int, default=5, help='最大测试图像数')
    
    args = parser.parse_args()
    
    print("AprilTag坐标系建立测试")
    print("=" * 50)
    
    if args.image:
        # 测试单张图像
        output_path = os.path.join(args.output_dir, 'single_test_result.png')
        os.makedirs(args.output_dir, exist_ok=True)
        
        success = test_single_image(
            args.image,
            camera_yaml=args.camera_yaml,
            rows=args.rows,
            cols=args.cols,
            spacing=args.spacing,
            tag_size=args.tag_size,
            output_path=output_path
        )
        
        if success:
            print("\n✅ 单图像测试成功!")
        else:
            print("\n❌ 单图像测试失败!")
    else:
        # 测试数据目录
        test_data_directory(
            data_dir=args.data_dir,
            camera_yaml=args.camera_yaml,
            output_dir=args.output_dir,
            max_images=args.max_images
        )


if __name__ == '__main__':
    main()
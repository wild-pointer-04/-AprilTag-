#!/usr/bin/env python3
"""
AprilTag检测调试工具

用于调试AprilTag检测问题，提供详细的检测信息和可视化
"""

import cv2
import numpy as np
import os
import sys
import argparse

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    import apriltag
    APRILTAG_AVAILABLE = True
except ImportError:
    APRILTAG_AVAILABLE = False
    print("❌ AprilTag库不可用，请安装: pip install apriltag")

from src.utils import load_camera_intrinsics, get_camera_intrinsics


def debug_apriltag_detection(image_path: str,
                            camera_yaml: str = None,
                            output_path: str = None,
                            show_debug: bool = True):
    """
    调试AprilTag检测
    
    Args:
        image_path: 图像路径
        camera_yaml: 相机内参文件
        output_path: 输出调试图像路径
        show_debug: 是否显示调试信息
    """
    if not APRILTAG_AVAILABLE:
        return False
    
    print(f"调试AprilTag检测: {image_path}")
    print("=" * 60)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return False
    
    h, w = image.shape[:2]
    print(f"图像尺寸: {w} x {h}")
    
    # 加载相机内参
    if camera_yaml:
        try:
            K, dist, image_size = load_camera_intrinsics(camera_yaml)
            if K is not None:
                print(f"✅ 已加载相机内参")
                # 畸变矫正
                undistorted = cv2.undistort(image, K, dist)
                print(f"✅ 已进行畸变矫正")
            else:
                print("⚠️ 使用默认相机内参")
                undistorted = image.copy()
                K, dist = get_camera_intrinsics(h, w)
        except Exception as e:
            print(f"⚠️ 加载内参失败: {e}")
            undistorted = image.copy()
            K, dist = get_camera_intrinsics(h, w)
    else:
        print("⚠️ 未提供相机内参，使用原始图像")
        undistorted = image.copy()
        K, dist = get_camera_intrinsics(h, w)
    
    # 转换为灰度图
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    
    # 测试不同的AprilTag家族
    tag_families = ['tag36h11', 'tag25h9', 'tag16h5']
    
    all_detections = []
    
    for family in tag_families:
        print(f"\n测试AprilTag家族: {family}")
        print("-" * 40)
        
        try:
            # 创建检测器
            options = apriltag.DetectorOptions(families=family)
            detector = apriltag.Detector(options)
            
            # 检测AprilTag
            detections = detector.detect(gray)
            
            print(f"检测到 {len(detections)} 个 {family} 标签")
            
            for i, detection in enumerate(detections):
                print(f"  标签 {i+1}:")
                print(f"    ID: {detection.tag_id}")
                print(f"    家族: {detection.tag_family}")
                print(f"    中心: ({detection.center[0]:.1f}, {detection.center[1]:.1f})")
                print(f"    决策边界: {detection.decision_margin:.3f}")
                print(f"    汉明距离: {detection.hamming}")
                print(f"    角点: {detection.corners}")
                
                all_detections.append((family, detection))
        
        except Exception as e:
            print(f"❌ {family} 检测失败: {e}")
    
    # 创建调试可视化图像
    debug_image = undistorted.copy()
    
    if all_detections:
        print(f"\n✅ 总共检测到 {len(all_detections)} 个AprilTag")
        
        # 绘制所有检测到的AprilTag
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, (family, detection) in enumerate(all_detections):
            color = colors[i % len(colors)]
            
            # 绘制边框
            corners = detection.corners.astype(int)
            cv2.polylines(debug_image, [corners], True, color, 3)
            
            # 绘制中心点
            center = detection.center.astype(int)
            cv2.circle(debug_image, tuple(center), 8, color, -1)
            
            # 绘制ID和家族信息
            text = f"{family}:{detection.tag_id}"
            cv2.putText(debug_image, text, 
                       (center[0] - 40, center[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 绘制决策边界信息
            margin_text = f"M:{detection.decision_margin:.2f}"
            cv2.putText(debug_image, margin_text,
                       (center[0] - 30, center[1] + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    else:
        print("\n❌ 未检测到任何AprilTag")
        
        # 添加调试信息到图像
        cv2.putText(debug_image, "No AprilTag Detected", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 提供调试建议
        print("\n调试建议:")
        print("1. 检查AprilTag是否在图像中清晰可见")
        print("2. 确认AprilTag没有被遮挡或模糊")
        print("3. 检查AprilTag的打印质量和尺寸")
        print("4. 确认使用的AprilTag家族正确")
        print("5. 调整图像对比度和亮度")
        print("6. 检查AprilTag是否有损坏或污渍")
    
    # 保存或显示调试图像
    if output_path:
        cv2.imwrite(output_path, debug_image)
        print(f"\n✅ 调试图像已保存: {output_path}")
    
    if show_debug:
        # 创建对比图像（原图 vs 调试图）
        comparison = np.hstack([undistorted, debug_image])
        
        # 调整显示尺寸
        display_width = 1200
        scale = display_width / comparison.shape[1]
        if scale < 1.0:
            new_height = int(comparison.shape[0] * scale)
            comparison = cv2.resize(comparison, (display_width, new_height))
        
        cv2.imshow('AprilTag Detection Debug (Original | Debug)', comparison)
        print("\n按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return len(all_detections) > 0


def batch_debug_images(image_dir: str,
                      camera_yaml: str = None,
                      output_dir: str = 'debug_output'):
    """
    批量调试多张图像
    """
    if not os.path.exists(image_dir):
        print(f"❌ 图像目录不存在: {image_dir}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找图像文件
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(image_dir) if f.lower().endswith(ext)])
    
    image_files.sort()
    
    if not image_files:
        print(f"❌ 在 {image_dir} 中未找到图像文件")
        return
    
    print(f"批量调试 {len(image_files)} 张图像...")
    
    detection_results = []
    
    for i, image_file in enumerate(image_files):
        print(f"\n{'='*60}")
        print(f"调试图像 {i+1}/{len(image_files)}: {image_file}")
        print(f"{'='*60}")
        
        image_path = os.path.join(image_dir, image_file)
        output_path = os.path.join(output_dir, f'debug_{image_file}')
        
        success = debug_apriltag_detection(
            image_path,
            camera_yaml=camera_yaml,
            output_path=output_path,
            show_debug=False
        )
        
        detection_results.append((image_file, success))
    
    # 总结
    print(f"\n{'='*60}")
    print("批量调试总结")
    print(f"{'='*60}")
    
    success_count = sum(1 for _, success in detection_results if success)
    total_count = len(detection_results)
    
    print(f"总图像数: {total_count}")
    print(f"检测成功: {success_count}")
    print(f"检测失败: {total_count - success_count}")
    print(f"成功率: {success_count / total_count * 100:.1f}%")
    
    print(f"\n详细结果:")
    for image_file, success in detection_results:
        status = "✅" if success else "❌"
        print(f"  {status} {image_file}")
    
    print(f"\n调试图像已保存到: {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AprilTag检测调试工具')
    parser.add_argument('--image', type=str, help='单张图像路径')
    parser.add_argument('--image-dir', type=str, help='图像目录路径（批量处理）')
    parser.add_argument('--camera-yaml', type=str, default='config/camera_info.yaml',
                       help='相机内参文件')
    parser.add_argument('--output', type=str, help='输出调试图像路径')
    parser.add_argument('--output-dir', type=str, default='debug_output',
                       help='批量处理输出目录')
    parser.add_argument('--no-display', action='store_true',
                       help='不显示调试图像')
    
    args = parser.parse_args()
    
    if not APRILTAG_AVAILABLE:
        print("请先安装AprilTag库: pip install apriltag")
        return
    
    if args.image:
        # 单张图像调试
        success = debug_apriltag_detection(
            args.image,
            camera_yaml=args.camera_yaml,
            output_path=args.output,
            show_debug=not args.no_display
        )
        
        if success:
            print("\n✅ AprilTag检测成功!")
        else:
            print("\n❌ AprilTag检测失败!")
    
    elif args.image_dir:
        # 批量图像调试
        batch_debug_images(
            args.image_dir,
            camera_yaml=args.camera_yaml,
            output_dir=args.output_dir
        )
    
    else:
        print("请指定 --image 或 --image-dir 参数")


if __name__ == '__main__':
    main()
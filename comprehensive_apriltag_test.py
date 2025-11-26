#!/usr/bin/env python3
"""
全面的AprilTag检测测试工具

测试所有可能的AprilTag家族，找到正确的家族
"""

import cv2
import numpy as np
import os
import sys

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from pupil_apriltags import Detector
    APRILTAG_AVAILABLE = True
    USING_PUPIL_APRILTAGS = True
except ImportError:
    try:
        import apriltag
        APRILTAG_AVAILABLE = True
        USING_PUPIL_APRILTAGS = False
    except ImportError:
        APRILTAG_AVAILABLE = False
        print("❌ AprilTag库不可用，请安装: pip install pupil-apriltags 或 pip install apriltag")

from src.utils import load_camera_intrinsics, get_camera_intrinsics


def test_all_apriltag_families(image_path: str, camera_yaml: str = None):
    """
    测试所有AprilTag家族
    """
    if not APRILTAG_AVAILABLE:
        return
    
    print(f"全面测试AprilTag检测: {image_path}")
    print("=" * 80)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return
    
    h, w = image.shape[:2]
    print(f"图像尺寸: {w} x {h}")
    
    # 加载相机内参并进行畸变矫正
    if camera_yaml:
        try:
            K, dist, image_size = load_camera_intrinsics(camera_yaml)
            if K is not None:
                undistorted = cv2.undistort(image, K, dist)
                print(f"✅ 已进行畸变矫正")
            else:
                undistorted = image.copy()
        except Exception as e:
            print(f"⚠️ 加载内参失败: {e}")
            undistorted = image.copy()
    else:
        undistorted = image.copy()
    
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    
    # 测试所有可能的AprilTag家族
    all_families = [
        'tag36h11', 'tag25h9', 'tag16h5',
        'tagStandard41h12', 'tagStandard52h13',
        'tagCircle21h7', 'tagCircle49h12',
        'tagCustom48h12'
    ]
    
    print(f"\n测试 {len(all_families)} 个AprilTag家族:")
    print("-" * 80)
    
    all_detections = []
    
    for family in all_families:
        print(f"\n🔍 测试家族: {family}")
        
        try:
            # 创建检测器
            if USING_PUPIL_APRILTAGS:
                detector = Detector(
                    families=family,
                    nthreads=4,
                    quad_decimate=1.0,
                    quad_sigma=0.0,
                    refine_edges=True
                )
            else:
                options = apriltag.DetectorOptions(families=family)
                detector = apriltag.Detector(options)
            
            # 检测AprilTag
            detections = detector.detect(gray)
            
            if len(detections) > 0:
                print(f"  ✅ 检测到 {len(detections)} 个 {family} 标签")
                
                for i, detection in enumerate(detections):
                    print(f"    标签 {i+1}:")
                    print(f"      ID: {detection.tag_id}")
                    print(f"      中心: ({detection.center[0]:.1f}, {detection.center[1]:.1f})")
                    print(f"      决策边界: {detection.decision_margin:.3f}")
                    print(f"      汉明距离: {detection.hamming}")
                    
                    all_detections.append((family, detection))
            else:
                print(f"  ❌ 未检测到 {family} 标签")
        
        except Exception as e:
            print(f"  ❌ {family} 检测失败: {e}")
    
    # 总结结果
    print("\n" + "=" * 80)
    print("检测结果总结:")
    print("=" * 80)
    
    if all_detections:
        print(f"✅ 总共检测到 {len(all_detections)} 个AprilTag")
        
        # 按家族分组
        family_groups = {}
        for family, detection in all_detections:
            if family not in family_groups:
                family_groups[family] = []
            family_groups[family].append(detection)
        
        for family, detections in family_groups.items():
            print(f"\n📋 {family} 家族:")
            for detection in detections:
                print(f"  - ID: {detection.tag_id}, 中心: ({detection.center[0]:.1f}, {detection.center[1]:.1f})")
        
        # 推荐使用的家族
        print(f"\n💡 推荐配置:")
        best_family = max(family_groups.keys(), key=lambda f: len(family_groups[f]))
        best_detections = family_groups[best_family]
        print(f"  使用家族: {best_family}")
        print(f"  检测到的标签ID: {[d.tag_id for d in best_detections]}")
        
        # 生成命令行参数
        print(f"\n🚀 使用以下参数运行程序:")
        print(f"  --tag-family {best_family}")
        
        # 可视化最佳结果
        vis_image = undistorted.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, detection in enumerate(best_detections):
            color = colors[i % len(colors)]
            
            # 绘制边框
            corners = detection.corners.astype(int)
            cv2.polylines(vis_image, [corners], True, color, 3)
            
            # 绘制中心点
            center = detection.center.astype(int)
            cv2.circle(vis_image, tuple(center), 8, color, -1)
            
            # 绘制ID
            text = f"{best_family}:{detection.tag_id}"
            cv2.putText(vis_image, text, 
                       (center[0] - 40, center[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 保存可视化结果
        output_path = f'comprehensive_test_result_{best_family}.png'
        cv2.imwrite(output_path, vis_image)
        print(f"  可视化结果已保存: {output_path}")
        
    else:
        print("❌ 未检测到任何AprilTag")
        print("\n🔧 调试建议:")
        print("  1. 检查AprilTag是否在图像中清晰可见")
        print("  2. 确认AprilTag没有被遮挡或模糊")
        print("  3. 检查AprilTag的打印质量")
        print("  4. 尝试调整图像对比度和亮度")
        print("  5. 确认AprilTag的实际家族类型")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='全面AprilTag家族测试')
    parser.add_argument('--image', type=str, required=True, help='图像路径')
    parser.add_argument('--camera-yaml', type=str, default='config/camera_info.yaml',
                       help='相机内参文件')
    
    args = parser.parse_args()
    
    if not APRILTAG_AVAILABLE:
        print("请先安装AprilTag库: pip install apriltag")
        return
    
    test_all_apriltag_families(args.image, args.camera_yaml)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
专门计算Roll和Pitch角度的脚本
使用修复后的鲁棒AprilTag系统，避免247像素重投影误差
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

from src.robust_apriltag_system import RobustAprilTagSystem
from src.detect_grid_improved import try_find_adaptive
from src.utils import load_camera_intrinsics


def calculate_roll_pitch_angles(image_path, tag_family='tagStandard41h12'):
    """
    计算图像中的Roll和Pitch角度
    
    Args:
        image_path: 图像路径
        tag_family: AprilTag家族
    
    Returns:
        (success, roll, pitch, yaw, error)
    """
    
    print(f"\n{'='*60}")
    print(f"计算Roll和Pitch角度: {image_path}")
    print(f"{'='*60}")
    
    # 加载图像
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return False, None, None, None, None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法加载图像: {image_path}")
        return False, None, None, None, None
    
    print(f"✅ 图像加载成功: {image.shape[1]}x{image.shape[0]}")
    
    # 加载相机参数
    result = load_camera_intrinsics('config/camera_info.yaml')
    if len(result) == 3:
        camera_matrix, dist_coeffs, image_size = result
    else:
        camera_matrix, dist_coeffs = result
    
    if camera_matrix is None:
        print("❌ 无法加载相机参数")
        return False, None, None, None, None
    
    print("✅ 相机参数加载成功")
    
    # 初始化鲁棒AprilTag系统
    robust_system = RobustAprilTagSystem(
        tag_family=tag_family,
        tag_size=20.0,
        board_spacing=10.0,
        max_reprojection_error=10.0
    )
    
    print(f"✅ 鲁棒AprilTag系统初始化完成 (家族: {tag_family})")
    
    # 检测标定板角点
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 尝试不同的网格配置
    grid_configs = [
        (4, 11), (11, 4), (5, 9), (9, 5), (6, 8), (8, 6),
        (7, 10), (10, 7), (3, 12), (12, 3)
    ]
    
    board_corners = None
    grid_rows, grid_cols = None, None
    
    print("🔍 检测标定板角点...")
    for rows, cols in grid_configs:
        ret, corners, keypoints = try_find_adaptive(gray, rows, cols)
        if ret and corners is not None:
            board_corners = corners.reshape(-1, 2)
            grid_rows, grid_cols = rows, cols
            print(f"✅ 检测到标定板: {rows}×{cols} = {len(board_corners)} 个角点")
            break
    
    if board_corners is None:
        print("❌ 未检测到标定板角点")
        return False, None, None, None, None
    
    # 使用鲁棒系统进行位姿估计
    print("🔧 执行鲁棒位姿估计...")
    try:
        success, rvec, tvec, error, info = robust_system.robust_pose_estimation(
            image, board_corners, camera_matrix, dist_coeffs, grid_rows, grid_cols
        )
        
        if not success:
            print(f"❌ 位姿估计失败，重投影误差: {error:.3f}px")
            return False, None, None, None, error
        
        print(f"✅ 位姿估计成功")
        print(f"   使用方法: {info['pnp_info'].get('method', 'Unknown')}")
        print(f"   重投影误差: {error:.3f}px")
        
        # 检查是否解决了247像素问题
        if error > 50:
            print(f"⚠️ 重投影误差仍然较高: {error:.3f}px")
        elif error > 10:
            print(f"⚠️ 重投影误差中等: {error:.3f}px")
        else:
            print(f"✅ 重投影误差正常: {error:.3f}px")
        
        # 计算Roll、Pitch、Yaw角度
        print(f"\n{'='*40}")
        print("角度计算结果")
        print(f"{'='*40}")
        
        # 从旋转向量转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvec)
        
        # 计算欧拉角 (ZYX顺序)
        # Roll (绕X轴旋转)
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        
        # Pitch (绕Y轴旋转)
        pitch = np.degrees(np.arcsin(-R[2, 0]))
        
        # Yaw (绕Z轴旋转)
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        
        print(f"Roll (横滚角):  {roll:8.2f}°")
        print(f"Pitch (俯仰角): {pitch:8.2f}°")
        print(f"Yaw (偏航角):   {yaw:8.2f}°")
        
        # 角度合理性检查
        print(f"\n角度合理性检查:")
        if abs(roll) > 90:
            print(f"⚠️ Roll角度可能异常: {roll:.1f}°")
        else:
            print(f"✅ Roll角度正常: {roll:.1f}°")
        
        if abs(pitch) > 90:
            print(f"⚠️ Pitch角度可能异常: {pitch:.1f}°")
        else:
            print(f"✅ Pitch角度正常: {pitch:.1f}°")
        
        # 显示AprilTag一致性
        consistency = info['pnp_info'].get('apriltag_consistency', {})
        if 'is_consistent' in consistency:
            if consistency['is_consistent']:
                print(f"✅ AprilTag一致性检查通过")
            else:
                print(f"⚠️ AprilTag一致性检查未通过")
                if 'angle_difference_deg' in consistency:
                    print(f"   角度差异: {consistency['angle_difference_deg']:.1f}°")
        
        return True, roll, pitch, yaw, error
        
    except Exception as e:
        print(f"❌ 位姿估计异常: {e}")
        return False, None, None, None, None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='计算图像中的Roll和Pitch角度')
    parser.add_argument('--image', required=True, help='输入图像路径')
    parser.add_argument('--tag-family', default='tagStandard41h12', 
                       help='AprilTag家族 (默认: tagStandard41h12)')
    parser.add_argument('--batch', action='store_true', 
                       help='批量处理目录下的所有图像')
    
    args = parser.parse_args()
    
    if args.batch:
        # 批量处理
        if os.path.isdir(args.image):
            image_dir = args.image
            results = []
            
            for filename in sorted(os.listdir(image_dir)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(image_dir, filename)
                    success, roll, pitch, yaw, error = calculate_roll_pitch_angles(
                        image_path, args.tag_family
                    )
                    
                    results.append({
                        'filename': filename,
                        'success': success,
                        'roll': roll,
                        'pitch': pitch,
                        'yaw': yaw,
                        'error': error
                    })
            
            # 显示批量处理结果
            print(f"\n{'='*80}")
            print("批量处理结果汇总")
            print(f"{'='*80}")
            print(f"{'文件名':<25} {'成功':<6} {'Roll':<8} {'Pitch':<8} {'Yaw':<8} {'误差':<8}")
            print("-" * 80)
            
            for result in results:
                if result['success']:
                    print(f"{result['filename']:<25} {'✅':<6} "
                          f"{result['roll']:7.1f}° {result['pitch']:7.1f}° "
                          f"{result['yaw']:7.1f}° {result['error']:7.3f}px")
                else:
                    print(f"{result['filename']:<25} {'❌':<6} {'---':<8} {'---':<8} {'---':<8} {'---':<8}")
        else:
            print(f"❌ 目录不存在: {args.image}")
    else:
        # 单张图像处理
        success, roll, pitch, yaw, error = calculate_roll_pitch_angles(
            args.image, args.tag_family
        )
        
        if success:
            print(f"\n🎯 最终结果:")
            print(f"   Roll:  {roll:7.2f}°")
            print(f"   Pitch: {pitch:7.2f}°")
            print(f"   Yaw:   {yaw:7.2f}°")
            print(f"   误差:  {error:7.3f}px")
        else:
            print(f"\n❌ 角度计算失败")


if __name__ == '__main__':
    main()
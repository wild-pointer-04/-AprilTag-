#!/usr/bin/env python3
"""
测试重投影误差问题
专门分析247像素重投影误差的原因和解决方案
"""

import cv2
import numpy as np
import os
import sys
import logging

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from src.apriltag_coordinate_system import AprilTagCoordinateSystem
from src.robust_apriltag_system import RobustAprilTagSystem
from src.detect_grid_improved import try_find_adaptive
from src.utils import load_camera_intrinsics

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_single_image_reprojection_error(image_path):
    """测试单张图像的重投影误差"""
    
    print(f"\n{'='*60}")
    print(f"测试图像重投影误差: {image_path}")
    print(f"{'='*60}")
    
    # 加载图像
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法加载图像: {image_path}")
        return
    
    print(f"图像尺寸: {image.shape[1]} x {image.shape[0]}")
    
    # 加载相机参数
    result = load_camera_intrinsics('config/camera_info.yaml')
    if len(result) == 3:
        camera_matrix, dist_coeffs, image_size = result
    else:
        camera_matrix, dist_coeffs = result
    
    if camera_matrix is None:
        print("❌ 无法加载相机参数")
        return
    
    # 检测标定板角点
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 尝试不同的网格尺寸
    grid_configs = [
        (4, 11), (11, 4), (5, 9), (9, 5), (6, 8), (8, 6),
        (7, 10), (10, 7), (3, 12), (12, 3)
    ]
    
    board_corners = None
    grid_rows, grid_cols = None, None
    
    for rows, cols in grid_configs:
        ret, corners, keypoints = try_find_adaptive(gray, rows, cols)
        if ret and corners is not None:
            board_corners = corners.reshape(-1, 2)
            grid_rows, grid_cols = rows, cols
            print(f"✅ 检测到标定板: {rows}×{cols} = {len(board_corners)} 个角点")
            break
    
    if board_corners is None:
        print("❌ 未检测到标定板角点")
        return
    
    # 初始化系统
    original_system = AprilTagCoordinateSystem(
        tag_family='tagStandard41h12',  # 使用正确的家族名称
        tag_size=20.0,
        board_spacing=10.0
    )
    
    robust_system = RobustAprilTagSystem(
        tag_family='tagStandard41h12',  # 使用正确的家族名称
        tag_size=20.0,
        board_spacing=10.0,
        max_reprojection_error=10.0
    )
    
    print(f"\n{'='*40}")
    print("测试原始系统")
    print(f"{'='*40}")
    
    # 测试原始系统
    try:
        success_orig, origin, x_axis, y_axis, info_orig = original_system.establish_coordinate_system(
            image, board_corners, camera_matrix, dist_coeffs, grid_rows, grid_cols
        )
        
        if success_orig and info_orig:
            print("✅ 原始系统成功建立坐标系")
            print(f"  AprilTag ID: {info_orig['tag_id']}")
            print(f"  AprilTag中心: ({info_orig['tag_center'][0]:.1f}, {info_orig['tag_center'][1]:.1f})")
            
            # 计算简单PnP的重投影误差
            objpoints_simple = np.zeros((len(board_corners), 3), dtype=np.float32)
            for i, corner in enumerate(board_corners):
                row = i // grid_cols
                col = i % grid_cols
                objpoints_simple[i] = [col * 10.0, row * 10.0, 0.0]  # 10mm间距
            
            # 标准PnP求解
            success_pnp, rvec_orig, tvec_orig = cv2.solvePnP(
                objpoints_simple, board_corners, camera_matrix, dist_coeffs
            )
            
            if success_pnp:
                # 计算重投影误差
                projected_points, _ = cv2.projectPoints(
                    objpoints_simple, rvec_orig, tvec_orig, camera_matrix, dist_coeffs
                )
                errors = np.linalg.norm(
                    projected_points.reshape(-1, 2) - board_corners.reshape(-1, 2),
                    axis=1
                )
                
                mean_error = np.mean(errors)
                max_error = np.max(errors)
                
                print(f"  PnP求解: 成功")
                print(f"  平均重投影误差: {mean_error:.3f} 像素")
                print(f"  最大重投影误差: {max_error:.3f} 像素")
                
                if max_error > 50:
                    print(f"  ⚠️ 检测到高重投影误差！")
                    print(f"  这可能是PnP多解歧义问题")
                
                # 分析误差分布
                high_error_count = np.sum(errors > 50)
                very_high_error_count = np.sum(errors > 100)
                
                if high_error_count > 0:
                    print(f"  误差>50px的点数: {high_error_count}")
                if very_high_error_count > 0:
                    print(f"  误差>100px的点数: {very_high_error_count}")
                
                # 显示前5个最大误差的点
                max_error_indices = np.argsort(errors)[-5:]
                print(f"  前5个最大误差:")
                for i, idx in enumerate(max_error_indices):
                    print(f"    点{idx}: {errors[idx]:.3f}px")
            else:
                print("  ❌ PnP求解失败")
        else:
            print("❌ 原始系统建立坐标系失败")
    
    except Exception as e:
        print(f"❌ 原始系统测试失败: {e}")
    
    print(f"\n{'='*40}")
    print("测试鲁棒系统")
    print(f"{'='*40}")
    
    # 测试鲁棒系统
    try:
        success_robust, rvec_robust, tvec_robust, robust_error, robust_info = robust_system.robust_pose_estimation(
            image, board_corners, camera_matrix, dist_coeffs, grid_rows, grid_cols
        )
        
        if success_robust:
            print("✅ 鲁棒系统成功估计位姿")
            print(f"  使用方法: {robust_info['pnp_info'].get('method', 'Unknown')}")
            print(f"  重投影误差: {robust_error:.3f} 像素")
            
            # 检查一致性
            consistency = robust_info['pnp_info'].get('apriltag_consistency', {})
            if 'is_consistent' in consistency:
                print(f"  AprilTag一致性: {'✅' if consistency['is_consistent'] else '❌'}")
                if 'angle_difference_deg' in consistency:
                    print(f"  角度差异: {consistency['angle_difference_deg']:.1f}°")
                if 'translation_difference_mm' in consistency:
                    print(f"  平移差异: {consistency['translation_difference_mm']:.1f}mm")
            
            # 验证信息
            validation = robust_info.get('validation', {})
            if 'mean_error' in validation:
                print(f"  验证平均误差: {validation['mean_error']:.3f}px")
                print(f"  验证最大误差: {validation['max_error']:.3f}px")
        else:
            print(f"❌ 鲁棒系统位姿估计失败，误差: {robust_error:.3f}px")
    
    except Exception as e:
        print(f"❌ 鲁棒系统测试失败: {e}")
    
    print(f"\n{'='*60}")
    print("测试总结")
    print(f"{'='*60}")


def main():
    """主函数"""
    
    # 测试图像列表
    test_images = []
    data_dir = 'data'
    
    # 查找所有测试图像
    for i in range(1, 13):
        img_path = os.path.join(data_dir, f'board{i}.png')
        if os.path.exists(img_path):
            test_images.append(img_path)
    
    if not test_images:
        print("❌ 未找到测试图像")
        print("请确保 data/ 目录下有 board1.png 到 board12.png 文件")
        return
    
    print(f"找到 {len(test_images)} 张测试图像")
    
    # 测试每张图像
    for img_path in test_images[:3]:  # 先测试前3张
        test_single_image_reprojection_error(img_path)
    
    print(f"\n{'='*60}")
    print("重投影误差问题分析完成")
    print(f"{'='*60}")
    print("如果看到247像素级别的误差，这通常是由于：")
    print("1. PnP多解歧义 - 对称网格导致多个可能的位姿解")
    print("2. 初始猜测错误 - PnP算法收敛到错误的局部最优解")
    print("3. 坐标系不一致 - 3D物体点定义与实际不符")
    print("4. AprilTag约束未充分利用")
    print("\n解决方案：")
    print("1. 使用鲁棒AprilTag系统 (RobustAprilTagSystem)")
    print("2. 多种PnP方法交叉验证")
    print("3. AprilTag位姿作为强约束")
    print("4. 几何一致性检查")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
测试PnP多解歧义修复效果

专门测试247像素重投影误差问题的解决方案
"""

import cv2
import numpy as np
import os
import logging
from src.robust_apriltag_system import RobustAprilTagSystem
from src.apriltag_coordinate_system import AprilTagCoordinateSystem
from src.utils import load_camera_intrinsics

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compare_systems():
    """对比原系统和鲁棒系统的效果"""
    
    # 加载相机参数
    result = load_camera_intrinsics('config/camera_info.yaml')
    if len(result) == 3:
        camera_matrix, dist_coeffs, image_size = result
    else:
        camera_matrix, dist_coeffs = result
    
    if camera_matrix is None:
        logger.error("无法加载相机参数")
        return
    
    # 初始化两个系统
    original_system = AprilTagCoordinateSystem(
        tag_family='tagStandard41h12',
        tag_size=20.0,
        board_spacing=10.0
    )
    
    robust_system = RobustAprilTagSystem(
        tag_family='tagStandard41h12',
        tag_size=20.0,
        board_spacing=10.0,
        max_reprojection_error=10.0  # 严格的误差阈值
    )
    
    # 测试图像
    test_images = []
    data_dir = 'data'
    for i in range(1, 13):  # board1.png 到 board12.png
        img_path = os.path.join(data_dir, f'board{i}.png')
        if os.path.exists(img_path):
            test_images.append(img_path)
    
    if not test_images:
        logger.error("未找到测试图像")
        return
    
    logger.info(f"找到 {len(test_images)} 张测试图像")
    
    results = {
        'original': [],
        'robust': []
    }
    
    for img_path in test_images:
        logger.info(f"\n处理图像: {img_path}")
        
        # 加载图像
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # 使用现有的检测方法
        from src.detect_grid_improved import try_find_adaptive
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 尝试检测网格
        grid_rows, grid_cols = 4, 11  # 根据你的标定板调整
        ret, corners, keypoints = try_find_adaptive(gray, grid_rows, grid_cols)
        
        if not ret or corners is None:
            logger.warning(f"  未检测到标定板角点")
            continue
        
        corners = corners.reshape(-1, 2)
        
        # 测试原系统
        try:
            success_orig, origin, x_axis, y_axis, info_orig = original_system.establish_coordinate_system(
                image, corners, camera_matrix, dist_coeffs, grid_rows, grid_cols
            )
            
            if success_orig and info_orig:
                # 计算原系统的重投影误差
                tag_rvec = info_orig['tag_rvec']
                tag_tvec = info_orig['tag_tvec']
                
                # 简单的PnP求解
                objpoints_simple = np.zeros((len(corners), 3), dtype=np.float32)
                for i, corner in enumerate(corners):
                    row = i // grid_cols
                    col = i % grid_cols
                    objpoints_simple[i] = [col * 10.0, row * 10.0, 0.0]  # 10mm间距
                
                success_pnp, rvec_orig, tvec_orig = cv2.solvePnP(
                    objpoints_simple, corners, camera_matrix, dist_coeffs
                )
                
                if success_pnp:
                    projected_points, _ = cv2.projectPoints(
                        objpoints_simple, rvec_orig, tvec_orig, camera_matrix, dist_coeffs
                    )
                    errors = np.linalg.norm(
                        projected_points.reshape(-1, 2) - corners.reshape(-1, 2),
                        axis=1
                    )
                    orig_error = np.mean(errors)
                else:
                    orig_error = float('inf')
            else:
                orig_error = float('inf')
                
        except Exception as e:
            logger.error(f"  原系统失败: {e}")
            orig_error = float('inf')
        
        # 测试鲁棒系统
        try:
            success_robust, rvec_robust, tvec_robust, robust_error, robust_info = robust_system.robust_pose_estimation(
                image, corners, camera_matrix, dist_coeffs, grid_rows, grid_cols
            )
            
            if not success_robust:
                robust_error = float('inf')
                
        except Exception as e:
            logger.error(f"  鲁棒系统失败: {e}")
            robust_error = float('inf')
        
        # 记录结果
        results['original'].append(orig_error)
        results['robust'].append(robust_error)
        
        logger.info(f"  原系统误差: {orig_error:.3f}px")
        logger.info(f"  鲁棒系统误差: {robust_error:.3f}px")
        logger.info(f"  改进效果: {orig_error - robust_error:.3f}px")
    
    # 统计结果
    print("\n" + "="*60)
    print("PnP多解歧义修复效果统计")
    print("="*60)
    
    orig_errors = [e for e in results['original'] if e != float('inf')]
    robust_errors = [e for e in results['robust'] if e != float('inf')]
    
    if orig_errors:
        print(f"原系统:")
        print(f"  平均误差: {np.mean(orig_errors):.3f}px")
        print(f"  最大误差: {np.max(orig_errors):.3f}px")
        print(f"  超过50px的帧数: {sum(1 for e in orig_errors if e > 50)}")
        print(f"  超过100px的帧数: {sum(1 for e in orig_errors if e > 100)}")
    
    if robust_errors:
        print(f"\n鲁棒系统:")
        print(f"  平均误差: {np.mean(robust_errors):.3f}px")
        print(f"  最大误差: {np.max(robust_errors):.3f}px")
        print(f"  超过50px的帧数: {sum(1 for e in robust_errors if e > 50)}")
        print(f"  超过100px的帧数: {sum(1 for e in robust_errors if e > 100)}")
    
    if orig_errors and robust_errors:
        improvement = np.mean(orig_errors) - np.mean(robust_errors)
        print(f"\n改进效果:")
        print(f"  平均误差减少: {improvement:.3f}px")
        print(f"  改进百分比: {improvement/np.mean(orig_errors)*100:.1f}%")


if __name__ == '__main__':
    compare_systems()
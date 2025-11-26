#!/usr/bin/env python3
"""
位姿验证和选择模块

用于解决PnP求解的多解性问题，选择最佳的位姿解
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def validate_pose_solution(rvec: np.ndarray, 
                          tvec: np.ndarray,
                          objpoints: np.ndarray,
                          imgpoints: np.ndarray,
                          camera_matrix: np.ndarray,
                          dist_coeffs: np.ndarray,
                          max_error_threshold: float = 5.0) -> Tuple[bool, float]:
    """
    验证位姿解的质量
    
    Args:
        rvec: 旋转向量
        tvec: 平移向量
        objpoints: 3D物体点
        imgpoints: 2D图像点
        camera_matrix: 相机内参
        dist_coeffs: 畸变系数
        max_error_threshold: 最大允许误差阈值
        
    Returns:
        (是否有效, 平均重投影误差)
    """
    try:
        # 计算重投影
        projected_points, _ = cv2.projectPoints(
            objpoints, rvec, tvec, camera_matrix, dist_coeffs
        )
        
        # 计算误差
        errors = np.linalg.norm(
            projected_points.reshape(-1, 2) - imgpoints.reshape(-1, 2), 
            axis=1
        )
        mean_error = np.mean(errors)
        
        # 验证条件
        is_valid = mean_error < max_error_threshold
        
        return is_valid, mean_error
        
    except Exception as e:
        logger.error(f"位姿验证失败: {e}")
        return False, float('inf')


def solve_robust_pnp(objpoints: np.ndarray,
                     imgpoints: np.ndarray,
                     camera_matrix: np.ndarray,
                     dist_coeffs: np.ndarray,
                     apriltag_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    鲁棒的PnP求解，尝试多种方法并选择最佳解
    
    Args:
        objpoints: 3D物体点
        imgpoints: 2D图像点  
        camera_matrix: 相机内参
        dist_coeffs: 畸变系数
        apriltag_pose: AprilTag位姿作为初始猜测 (rvec, tvec)
        
    Returns:
        (最佳rvec, 最佳tvec, 最佳误差)
    """
    solutions = []
    
    # 方法1: 标准ITERATIVE
    try:
        success, rvec, tvec = cv2.solvePnP(
            objpoints, imgpoints, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success:
            is_valid, error = validate_pose_solution(
                rvec, tvec, objpoints, imgpoints, camera_matrix, dist_coeffs
            )
            solutions.append(('ITERATIVE', rvec, tvec, error, is_valid))
    except:
        pass
    
    # 方法2: P3P
    if len(objpoints) >= 4:
        try:
            success, rvecs, tvecs = cv2.solvePnP(
                objpoints, imgpoints, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_P3P
            )
            if success:
                is_valid, error = validate_pose_solution(
                    rvecs, tvecs, objpoints, imgpoints, camera_matrix, dist_coeffs
                )
                solutions.append(('P3P', rvecs, tvecs, error, is_valid))
        except:
            pass
    
    # 方法3: 使用AprilTag位姿作为初始猜测
    if apriltag_pose is not None:
        try:
            tag_rvec, tag_tvec = apriltag_pose
            success, rvec, tvec = cv2.solvePnP(
                objpoints, imgpoints, camera_matrix, dist_coeffs,
                rvec=tag_rvec, tvec=tag_tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if success:
                is_valid, error = validate_pose_solution(
                    rvec, tvec, objpoints, imgpoints, camera_matrix, dist_coeffs
                )
                solutions.append(('APRILTAG_GUIDED', rvec, tvec, error, is_valid))
        except:
            pass
    
    # 选择最佳解
    if not solutions:
        return None, None, float('inf')
    
    # 优先选择有效解中误差最小的
    valid_solutions = [s for s in solutions if s[4]]  # is_valid=True
    if valid_solutions:
        best_solution = min(valid_solutions, key=lambda x: x[3])  # 按误差排序
    else:
        # 如果没有有效解，选择误差最小的
        best_solution = min(solutions, key=lambda x: x[3])
    
    method, rvec, tvec, error, is_valid = best_solution
    logger.debug(f"选择位姿解: {method}, 误差: {error:.3f}px, 有效: {is_valid}")
    
    return rvec, tvec, error
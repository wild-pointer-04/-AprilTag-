#!/usr/bin/env python3
"""
鲁棒的AprilTag系统 - 专门解决PnP多解歧义问题

这个模块专门解决你遇到的247像素重投影误差问题
核心改进：
1. 使用AprilTag位姿作为强约束
2. 多种PnP方法交叉验证
3. 几何一致性检查
4. 智能解选择策略
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
import logging
from .pnp_ambiguity_resolver import PnPAmbiguityResolver
from .apriltag_coordinate_system import AprilTagCoordinateSystem

logger = logging.getLogger(__name__)


class RobustAprilTagSystem:
    """鲁棒的AprilTag系统，解决PnP多解歧义"""
    
    def __init__(self, 
                 tag_family: str = 'tagStandard41h12',
                 tag_size: float = 0.0071,
                 board_spacing: float = 0.065,
                 max_reprojection_error: float = 10.0):
        """
        初始化鲁棒AprilTag系统
        
        Args:
            tag_family: AprilTag家族
            tag_size: AprilTag实际尺寸(mm)
            board_spacing: 标定板圆点间距(mm)
            max_reprojection_error: 最大允许重投影误差(像素)
        """
        self.apriltag_system = AprilTagCoordinateSystem(
            tag_family=tag_family,
            tag_size=tag_size,
            board_spacing=board_spacing,
            max_reprojection_error=max_reprojection_error
        )
        
        self.pnp_resolver = PnPAmbiguityResolver(
            max_reprojection_error=max_reprojection_error
        )
        
        logger.info("鲁棒AprilTag系统初始化完成")
        logger.info(f"最大重投影误差阈值: {max_reprojection_error}px")
    
    def robust_pose_estimation(self,
                               image: np.ndarray,
                               board_corners: np.ndarray,
                               camera_matrix: np.ndarray,
                               dist_coeffs: np.ndarray,
                               grid_rows: int,
                               grid_cols: int) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], float, Dict]:
        """
        鲁棒的位姿估计，解决PnP多解歧义
        
        Returns:
            (成功标志, rvec, tvec, 重投影误差, 详细信息)
        """ 
       
        # 步骤1: 建立AprilTag坐标系
        success, origin, x_axis, y_axis, apriltag_info = self.apriltag_system.establish_coordinate_system(
            image, board_corners, camera_matrix, dist_coeffs, grid_rows, grid_cols
        )
        
        if not success or apriltag_info is None:
            return False, None, None, float('inf'), {'error': 'AprilTag坐标系建立失败'}
        
        # 步骤2: 构建3D-2D对应关系
        objpoints_3d = self._build_3d_object_points(
            board_corners, origin, x_axis, y_axis, grid_rows, grid_cols
        )
        
        # 步骤3: 使用鲁棒PnP求解
        rvec, tvec, error, pnp_info = self.pnp_resolver.solve_robust_pnp_with_apriltag_constraint(
            objpoints_3d,
            board_corners,
            camera_matrix,
            dist_coeffs,
            apriltag_info['tag_rvec'],
            apriltag_info['tag_tvec']
        )
        
        if rvec is None:
            return False, None, None, float('inf'), {'error': 'PnP求解失败'}
        
        # 步骤4: 最终验证
        final_validation = self._final_validation(
            rvec, tvec, objpoints_3d, board_corners, 
            camera_matrix, dist_coeffs, apriltag_info
        )
        
        # 构建详细信息
        detailed_info = {
            'apriltag_info': apriltag_info,
            'pnp_info': pnp_info,
            'validation': final_validation,
            'objpoints_3d': objpoints_3d,
            'reprojection_error': error
        }
        
        success_flag = error < self.pnp_resolver.max_reprojection_error
        
        if success_flag:
            logger.info(f"✅ 鲁棒位姿估计成功")
            logger.info(f"  方法: {pnp_info.get('method', 'Unknown')}")
            logger.info(f"  重投影误差: {error:.3f}px")
            logger.info(f"  AprilTag一致性: {pnp_info.get('apriltag_consistency', {}).get('is_consistent', 'Unknown')}")
        else:
            logger.warning(f"⚠️ 位姿估计误差较大: {error:.3f}px")
        
        return success_flag, rvec, tvec, error, detailed_info
    
    def _build_3d_object_points(self,
                                board_corners: np.ndarray,
                                origin: np.ndarray,
                                x_axis: np.ndarray,
                                y_axis: np.ndarray,
                                grid_rows: int,
                                grid_cols: int) -> np.ndarray:
        """
        构建3D物体点坐标
        
        关键改进：使用AprilTag建立的坐标系来定义3D点
        这确保了3D-2D对应关系的一致性
        """
        objpoints_3d = []
        
        # 计算每个角点在新坐标系中的3D坐标
        for corner in board_corners:
            # 计算相对于原点的向量
            relative_vector = corner - origin
            
            # 投影到x轴和y轴上
            x_coord = np.dot(relative_vector, x_axis) * self.apriltag_system.board_spacing / np.linalg.norm(x_axis)
            y_coord = np.dot(relative_vector, y_axis) * self.apriltag_system.board_spacing / np.linalg.norm(y_axis)
            
            # 3D坐标（假设z=0，即所有点在同一平面上）
            objpoints_3d.append([x_coord, y_coord, 0.0])
        
        return np.array(objpoints_3d, dtype=np.float32)
    
    def _final_validation(self,
                          rvec: np.ndarray,
                          tvec: np.ndarray,
                          objpoints_3d: np.ndarray,
                          imgpoints: np.ndarray,
                          camera_matrix: np.ndarray,
                          dist_coeffs: np.ndarray,
                          apriltag_info: Dict) -> Dict:
        """最终验证位姿解的质量"""
        
        validation_results = {}
        
        try:
            # 1. 重投影误差验证
            projected_points, _ = cv2.projectPoints(
                objpoints_3d, rvec, tvec, camera_matrix, dist_coeffs
            )
            
            errors = np.linalg.norm(
                projected_points.reshape(-1, 2) - imgpoints.reshape(-1, 2),
                axis=1
            )
            
            validation_results['reprojection_errors'] = errors
            validation_results['mean_error'] = np.mean(errors)
            validation_results['max_error'] = np.max(errors)
            validation_results['error_std'] = np.std(errors)
            
            # 2. 与AprilTag位姿的一致性
            tag_rvec = apriltag_info['tag_rvec']
            tag_tvec = apriltag_info['tag_tvec']
            
            # 旋转差异
            R1 = cv2.Rodrigues(rvec)[0]
            R2 = cv2.Rodrigues(tag_rvec)[0]
            R_diff = R1.T @ R2
            angle_diff = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1)))
            
            # 平移差异
            trans_diff = np.linalg.norm(tvec - tag_tvec)
            
            validation_results['pose_consistency'] = {
                'rotation_diff_deg': angle_diff,
                'translation_diff_mm': trans_diff,
                'is_consistent': angle_diff < 30.0  # 30度阈值
            }
            
            # 3. 几何合理性检查
            validation_results['geometry_check'] = self._check_geometry_reasonableness(
                rvec, tvec, objpoints_3d
            )
            
        except Exception as e:
            validation_results['error'] = str(e)
        
        return validation_results
    
    def _check_geometry_reasonableness(self,
                                       rvec: np.ndarray,
                                       tvec: np.ndarray,
                                       objpoints_3d: np.ndarray) -> Dict:
        """检查几何合理性"""
        
        try:
            # 检查相机到物体的距离是否合理
            distance = np.linalg.norm(tvec)
            
            # 检查旋转角度是否合理
            rotation_angle = np.linalg.norm(rvec)
            rotation_angle_deg = np.degrees(rotation_angle)
            
            # 检查物体点的分布是否合理
            objpoints_range = {
                'x_range': np.max(objpoints_3d[:, 0]) - np.min(objpoints_3d[:, 0]),
                'y_range': np.max(objpoints_3d[:, 1]) - np.min(objpoints_3d[:, 1]),
                'z_range': np.max(objpoints_3d[:, 2]) - np.min(objpoints_3d[:, 2])
            }
            
            return {
                'distance_mm': distance,
                'rotation_angle_deg': rotation_angle_deg,
                'object_range': objpoints_range,
                'is_reasonable': 100 < distance < 2000 and rotation_angle_deg < 180
            }
            
        except Exception as e:
            return {'error': str(e)}


def test_robust_apriltag_system():
    """测试鲁棒AprilTag系统"""
    system = RobustAprilTagSystem()
    print("鲁棒AprilTag系统初始化完成")
    print("专门解决PnP多解歧义问题，避免247像素级别的重投影误差")


if __name__ == '__main__':
    test_robust_apriltag_system()
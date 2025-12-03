#!/usr/bin/env python3
"""
基于鲁棒AprilTag系统的相机倾斜检测节点

功能：
1. 从 ROS2 话题或 rosbag 读取图像
2. 检测AprilTag建立统一坐标系
3. 检测圆点网格并重新排列
4. 计算相机倾斜角度（基于统一坐标系）
5. 计算重投影误差
6. 发布检测结果

特点：
- 使用修复后的鲁棒AprilTag系统，避免247像素重投影误差
- 支持 tagStandard41h12 标签家族
- 多种PnP方法交叉验证
- AprilTag位姿约束
- 几何一致性检查

使用方法:
    # 从实时话题
    python robust_tilt_checker_node.py --image-topic /camera/image_raw --camera-yaml config/camera_info.yaml
    
    # 从 rosbag
    python robust_tilt_checker_node.py --rosbag /path/to/bag --image-topic /camera/image_raw --camera-yaml config/camera_info.yaml
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import sys
import argparse
import json
import csv
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) if 'src' in script_dir else script_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import (
    load_camera_intrinsics, scale_camera_intrinsics, get_camera_intrinsics,
    compute_camera_to_board_transform
)
from src.detect_grid_improved import try_find_adaptive, refine, auto_search
from src.robust_apriltag_system import RobustAprilTagSystem
from src.apriltag_coordinate_system import AprilTagCoordinateSystem


class RobustTiltCheckerNode(Node):
    """基于鲁棒AprilTag系统的相机倾斜检测节点"""
    
    def __init__(self, 
                 image_topic: str = '/camera/color/image_raw',
                 camera_yaml_path: str = 'config/camera_info.yaml',
                 rows: int = 15,
                 cols: int = 15,
                 tag_family: str = 'tagStandard41h12',
                 tag_size: float = 0.0071,  # AprilTag的实际尺寸(m)
                 board_spacing: float = 0.065,  # 标定板圆点间距(m）
                 max_reprojection_error: float = 1.0,  # 最大允许重投影误差
                 output_dir: str = 'outputs/robust_apriltag_results',
                 save_images: bool = True,
                 save_results: bool = True,
                 publish_results: bool = False,
                 rosbag_path: str = None
                 ):
        super().__init__('robust_tilt_checker_node')
        
        self.bridge = CvBridge()
        self.image_topic = image_topic
        self.camera_yaml_path = camera_yaml_path
        self.rows = rows
        self.cols = cols
        self.tag_family = tag_family
        self.tag_size = tag_size
        self.board_spacing = board_spacing
        self.max_reprojection_error = max_reprojection_error
        self.output_dir = output_dir
        self.save_images = save_images
        self.save_results = save_results
        self.publish_results = publish_results
        self.rosbag_path = rosbag_path
        # 创建ROS话题发布器（发布变换参数）
        if self.publish_results:
            self.transform_publisher = self.create_publisher(
                Float64MultiArray,
                '/tilt_checker/camera_to_board_transform',
                10
            )
            self.get_logger().info('✅ 已创建变换参数发布器: /tilt_checker/camera_to_board_transform')
        else:
            self.transform_publisher = None
        
        # 初始化鲁棒AprilTag系统
        self.robust_system = RobustAprilTagSystem(
            tag_family=tag_family,
            tag_size=tag_size,
            board_spacing=board_spacing,
            max_reprojection_error=max_reprojection_error
        )
        
        # 初始化标准AprilTag系统（用于对比）
        self.standard_system = AprilTagCoordinateSystem(
            tag_family=tag_family,
            tag_size=tag_size,
            board_spacing=board_spacing,
            max_reprojection_error=max_reprojection_error
        )
        
        # 加载相机内参
        self.K = None
        self.dist = None
        self.image_size = None
        self._load_camera_intrinsics()
        
        # 统计信息
        self.frame_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.apriltag_success_count = 0
        self.apriltag_failure_count = 0
        self.high_error_count = 0  # 高重投影误差计数
        self.fixed_error_count = 0  # 修复的高误差计数
        self.rejected_by_error_count = 0  # 因重投影误差超过阈值被淘汰的帧数
        self.all_results = []
        
        # 创建输出目录
        if self.save_results or self.save_images:
            os.makedirs(self.output_dir, exist_ok=True)
            if self.save_images:
                os.makedirs(os.path.join(self.output_dir, 'images'), exist_ok=True)
        
        self.get_logger().info('='*80)
        self.get_logger().info('🚀 基于鲁棒AprilTag系统的相机倾斜检测节点已启动')
        self.get_logger().info('='*80)
        self.get_logger().info(f'  图像话题: {self.image_topic}')
        self.get_logger().info(f'  相机内参: {self.camera_yaml_path}')
        self.get_logger().info(f'  网格尺寸: {self.rows} x {self.cols}')
        self.get_logger().info(f'  AprilTag家族: {tag_family}')
        self.get_logger().info(f'  AprilTag尺寸: {tag_size}mm')
        self.get_logger().info(f'  圆点间距: {board_spacing}mm')
        self.get_logger().info(f'  最大重投影误差: {max_reprojection_error}px')
        self.get_logger().info(f'  输出目录: {self.output_dir}')
        self.get_logger().info('='*80)
        self.get_logger().info('✅ 使用鲁棒AprilTag系统，避免247像素重投影误差问题')
        self.get_logger().info('✅ 支持多种PnP方法交叉验证')
        self.get_logger().info('✅ AprilTag位姿约束和几何一致性检查')
        self.get_logger().info('='*80)
    
    def _load_camera_intrinsics(self):
        """加载相机内参"""
        try:
            result = load_camera_intrinsics(self.camera_yaml_path)
            if len(result) == 3:
                K, dist, image_size = result
            else:
                K, dist = result
                image_size = None
            
            if K is None or dist is None:
                self.get_logger().warn(f'无法从 YAML 加载内参，将使用默认值')
                self.K = None
                self.dist = None
                self.image_size = None
            else:
                self.K = K
                self.dist = dist
                self.image_size = image_size
                if image_size:
                    self.get_logger().info(f'✅ 已加载相机内参 (YAML中图像尺寸: {image_size[0]} x {image_size[1]})')
                else:
                    self.get_logger().info(f'✅ 已加载相机内参 (YAML中未记录图像尺寸)')
        except Exception as e:
            self.get_logger().error(f'加载相机内参失败: {e}')
            self.K = None
            self.dist = None
            self.image_size = None
    
    def process_frame(self, cv_image, frame_id: str = None, timestamp: float = None):
        """
        处理单帧图像（使用鲁棒AprilTag系统）
        
        参数:
            cv_image: OpenCV 图像 (BGR)
            frame_id: 帧 ID（可选）
            timestamp: 时间戳（可选）
        
        返回:
            result: 检测结果字典，如果失败返回 None
        """
        self.frame_count += 1
        
        if frame_id is None:
            frame_id = f'frame_{self.frame_count:06d}'
        if timestamp is None:
            timestamp = self.frame_count * 0.1
        
        h, w = cv_image.shape[:2]
        actual_size = (w, h)
        
        self.get_logger().info(f'\n{"="*60}')
        self.get_logger().info(f'处理帧: {frame_id} ({w}x{h})')
        self.get_logger().info(f'{"="*60}')
        
        # 1. 获取并自动缩放相机内参
        if self.K is not None and self.dist is not None:
            if self.image_size is not None:
                yaml_size = self.image_size
                if yaml_size[0] != w or yaml_size[1] != h:
                    K_used, dist_used = scale_camera_intrinsics(
                        self.K, self.dist, yaml_size, actual_size
                    )
                    self.get_logger().info(
                        f'[{frame_id}] 已自动缩放内参矩阵 '
                        f'(缩放比例: {w/yaml_size[0]:.3f} x {h/yaml_size[1]:.3f})'
                    )
                else:
                    K_used = self.K.copy()
                    dist_used = self.dist.copy()
            else:
                K_used = self.K.copy()
                dist_used = self.dist.copy()
            
            undistorted = cv2.undistort(cv_image, K_used, dist_used)
            self.get_logger().info(f'[{frame_id}] ✅ 已进行畸变矫正')
        else:
            undistorted = cv_image.copy()
            K_used, dist_used = get_camera_intrinsics(h, w, yaml_path=None, f_scale=1.0)
            self.get_logger().info(f'[{frame_id}] 使用默认内参')
        
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        
        # 2. 检测圆点网格 - 恢复原始的成功检测方法
        grid_rows = self.rows
        grid_cols = self.cols
        grid_symmetric = True
        detection_source = 'direct'
        
        self.get_logger().info(f'[{frame_id}] 🔍 检测标定板角点 ({grid_rows}×{grid_cols})...')
        
        # 调试：保存预处理后的图像
        if self.save_images:
            debug_gray_path = os.path.join(self.output_dir, 'images', f'{frame_id}_debug_gray.png')
            cv2.imwrite(debug_gray_path, gray)
            self.get_logger().debug(f'[{frame_id}] 保存调试灰度图: {debug_gray_path}')
        
        try:
            # 首先尝试直接检测（与原始tilt_checker_node.py相同）
            ok, corners, blob_keypoints = try_find_adaptive(gray, grid_rows, grid_cols, symmetric=grid_symmetric)
            
            if (not ok) or (corners is None):
                self.get_logger().info(f'[{frame_id}] 首次检测失败，启用预处理增强后再试...')
                ok, corners, blob_keypoints = try_find_adaptive(
                    gray, grid_rows, grid_cols, symmetric=grid_symmetric, use_preprocessing=True
                )
            
            if (not ok) or (corners is None):
                self.get_logger().warn(f'[{frame_id}] 未检测到完整 {self.rows*self.cols} 网格，尝试降级搜索局部子网格...')
                # 使用与原始节点相同的降级搜索策略
                rows_range = (max(4, self.rows - 6), self.rows)
                cols_range = (max(4, self.cols - 6), self.cols)
                auto_ok, auto_corners, meta, blob_keypoints = auto_search(
                    gray, rows_range=rows_range, cols_range=cols_range
                )
                if auto_ok and auto_corners is not None and meta is not None:
                    ok = True
                    corners = auto_corners.reshape(-1, 1, 2)
                    grid_rows, grid_cols, grid_symmetric = meta
                    detection_source = 'fallback'
                    self.get_logger().info(
                        f'[{frame_id}] ✅ 降级搜索成功，使用 {grid_rows}×{grid_cols} 网格 (symmetric={grid_symmetric}) '
                        f'({len(corners)} 个点)'
                    )
                else:
                    ok = False
            
            if not ok or corners is None:
                self.get_logger().warn(f'[{frame_id}] ❌ 未检测到网格')
                
                # 保存失败帧的图像用于调试
                if self.save_images:
                    fail_img_path = os.path.join(self.output_dir, 'images', f'{frame_id}_FAILED.png')
                    fail_vis = undistorted.copy()
                    
                    # 如果有blob_keypoints，绘制出来
                    if blob_keypoints is not None and len(blob_keypoints) > 0:
                        for kp in blob_keypoints:
                            x, y = int(kp.pt[0]), int(kp.pt[1])
                            cv2.circle(fail_vis, (x, y), 5, (0, 255, 0), 2)
                        cv2.putText(fail_vis, f'Blob detected: {len(blob_keypoints)}', (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(fail_vis, 'NO BLOBS DETECTED!', (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    cv2.putText(fail_vis, f'Frame: {frame_id}', (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.imwrite(fail_img_path, fail_vis)
                    self.get_logger().info(f'[{frame_id}] 💾 已保存失败帧图像: {fail_img_path}')
                
                self.failure_count += 1
                return None
            
            # 检查点数是否匹配（与原始节点相同的验证）
            expected_pts = grid_rows * grid_cols
            if len(corners) != expected_pts:
                self.get_logger().warn(
                    f'[{frame_id}] 检测到 {len(corners)} 个点，但当前网格设置为 {grid_rows}×{grid_cols}={expected_pts} 个，'
                    ' 无法建立稳定坐标系。'
                )
                self.failure_count += 1
                return None
            
            # 精化角点
            corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
            corners_refined = refine(gray, corners)
            board_corners_2d = corners_refined.reshape(-1, 2)
            
            self.get_logger().info(f'[{frame_id}] ✅ 检测到 {len(board_corners_2d)} 个角点')
            
        except Exception as e:
            self.get_logger().error(f'[{frame_id}] 网格检测失败: {e}')
            self.failure_count += 1
            return None
        
        # 3. 基于AprilTag建立坐标系（与tilt_checker_with_apriltag.py相同的方法）
        self.get_logger().info(f'[{frame_id}] 🔧 建立AprilTag坐标系...')
        
        try:
            coord_success, origin_2d, x_direction, y_direction, coord_info = self.standard_system.establish_coordinate_system(
                undistorted, board_corners_2d, K_used, dist_used, grid_rows, grid_cols
            )
            
            if not coord_success:
                self.get_logger().warn(f'[{frame_id}] AprilTag坐标系建立失败，使用原始检测结果')
                self.apriltag_failure_count += 1
                # 回退到原始方法
                ordered_corners = corners_refined
                coord_info = None
            else:
                self.get_logger().info(f'[{frame_id}] ✅ AprilTag坐标系建立成功 (ID: {coord_info["tag_id"]})')
                self.apriltag_success_count += 1
                # 使用重新排列的角点
                reordered = np.asarray(coord_info['reordered_corners'], dtype=np.float32)
                ordered_corners = reordered.reshape(-1, 1, 2)
                
        except Exception as e:
            self.get_logger().warn(f'[{frame_id}] AprilTag处理失败: {e}，使用原始检测结果')
            self.apriltag_failure_count += 1
            ordered_corners = corners_refined
            coord_info = None
        
        # 4. 使用鲁棒PnP求解（基于AprilTag坐标系）
        self.get_logger().info(f'[{frame_id}] 🔧 执行鲁棒PnP求解...')
        
        try:
            # 构建3D物体点（基于AprilTag坐标系）
            objpoints_3d = self._build_apriltag_based_obj_points(
                grid_rows, grid_cols, self.board_spacing, coord_info, grid_symmetric
            )
            
            pts2d = ordered_corners.reshape(-1, 2)
            
            # 获取AprilTag位姿（如果可用）
            if coord_info is not None:
                apriltag_rvec = coord_info['tag_rvec']
                apriltag_tvec = coord_info['tag_tvec']
                
                # 使用鲁棒PnP求解器
                # ✅ 图像已去畸变，这里不再传入畸变系数
                rvec_robust, tvec_robust, robust_error_tmp, pnp_info = \
                    self.robust_system.pnp_resolver.solve_robust_pnp_with_apriltag_constraint(
                        objpoints_3d,
                        pts2d,
                        K_used,
                        None,  # ✅ 改为 None
                        apriltag_rvec,
                        apriltag_tvec
                    )
                
                if rvec_robust is None:
                    self.get_logger().warn(f'[{frame_id}] 鲁棒PnP求解失败，回退到标准方法')
                    # 回退到标准PnP（同样使用零畸变）
                    success_pnp, rvec_robust, tvec_robust = cv2.solvePnP(
                        objpoints_3d, pts2d, K_used, None
                    )
                    if not success_pnp:
                        self.failure_count += 1
                        return None
                    
                    # 计算重投影误差（零畸变）
                    projected_points, _ = cv2.projectPoints(
                        objpoints_3d, rvec_robust, tvec_robust, K_used, None
                    )
                    errors = np.linalg.norm(
                        projected_points.reshape(-1, 2) - pts2d,
                        axis=1
                    )
                    robust_error_mean = float(np.mean(errors))
                    robust_error_max = float(np.max(errors))
                    pnp_info = {'method': 'STANDARD_FALLBACK'}
                else:
                    # 鲁棒PnP成功，同样用零畸变重新精确计算误差
                    projected_points, _ = cv2.projectPoints(
                        objpoints_3d, rvec_robust, tvec_robust, K_used, None
                    )
                    errors = np.linalg.norm(
                        projected_points.reshape(-1, 2) - pts2d,
                        axis=1
                    )
                    robust_error_mean = float(np.mean(errors))
                    robust_error_max = float(np.max(errors))
                
                pnp_method = pnp_info.get('method', 'Unknown')
            
            else:
                # 没有AprilTag约束，使用标准PnP（零畸变）
                success_pnp, rvec_robust, tvec_robust = cv2.solvePnP(
                    objpoints_3d, pts2d, K_used, None
                )
                if not success_pnp:
                    self.failure_count += 1
                    return None
                
                # 计算重投影误差（零畸变）
                projected_points, _ = cv2.projectPoints(
                    objpoints_3d, rvec_robust, tvec_robust, K_used, None
                )
                   
                errors = np.linalg.norm(
                    projected_points.reshape(-1, 2) - pts2d,
                    axis=1
                )
                robust_error_mean = float(np.mean(errors))
                robust_error_max = float(np.max(errors))
                pnp_method = 'STANDARD_NO_APRILTAG'
                pnp_info = {'method': pnp_method}
            
            # 检查重投影误差并淘汰超过阈值的帧（使用 mean 和 max 进行日志输出）
            if robust_error_mean > self.max_reprojection_error:
                self.rejected_by_error_count += 1
                self.failure_count += 1
                self.get_logger().error(
                    f'[{frame_id}] ❌ 重投影误差 {robust_error_mean:.3f}px '
                    f'(最大: {robust_error_max:.3f}px) 超过阈值 {self.max_reprojection_error}px，淘汰该帧'
                )
                self.get_logger().info(
                    f'[{frame_id}] 📊 统计: 成功={self.success_count}, 失败={self.failure_count}, '
                    f'因误差淘汰={self.rejected_by_error_count}'
                )
                return None
            
            # 记录高误差但未超过阈值的情况
            if robust_error_mean > 50:
                self.high_error_count += 1
                self.get_logger().warn(
                    f'[{frame_id}] ⚠️ 重投影误差较高: 平均={robust_error_mean:.3f}px, '
                    f'最大={robust_error_max:.3f}px (但未超过阈值)'
                )
            else:
                self.get_logger().info(
                    f'[{frame_id}] ✅ 重投影误差正常: 平均={robust_error_mean:.3f}px, '
                    f'最大={robust_error_max:.3f}px'
                )
            
            self.get_logger().info(f'[{frame_id}] 使用方法: {pnp_method}')
            
            if coord_info:
                self.get_logger().info(f'[{frame_id}] ✅ AprilTag检测成功 (ID: {coord_info["tag_id"]})')
            else:
                self.get_logger().warn(f'[{frame_id}] ❌ AprilTag检测失败')
            
        except Exception as e:
            self.get_logger().error(f'[{frame_id}] 鲁棒PnP求解异常: {e}')
            self.failure_count += 1
            return None
        
        # 5. 计算角度（使用与tilt_checker_with_apriltag.py相同的方法）
        self.get_logger().info(f'[{frame_id}] 📐 计算相机倾斜角度...')
        
        try:
            from src.estimate_tilt import rvec_to_euler_xyz, rvec_to_camera_tilt, normalize_angles
            
            # 方法1: 标准欧拉角（板子相对于相机的旋转）
            roll_euler, pitch_euler, yaw_euler = rvec_to_euler_xyz(rvec_robust)
            roll_euler, pitch_euler, yaw_euler = normalize_angles(roll_euler, pitch_euler, yaw_euler)
            
            # 方法2: 相机倾斜角（假设板子水平，计算相机相对于水平面的倾斜）
            roll_tilt, pitch_tilt, yaw_tilt = rvec_to_camera_tilt(rvec_robust)
            roll_tilt, pitch_tilt, yaw_tilt = normalize_angles(roll_tilt, pitch_tilt, yaw_tilt)
            
            # 使用相机倾斜角作为主要结果
            roll = roll_tilt
            pitch = pitch_tilt
            yaw = yaw_tilt
            
            self.get_logger().info(f'[{frame_id}] 角度结果:')
            self.get_logger().info(f'[{frame_id}]   Roll (横滚角):  {roll:+8.2f}°')
            self.get_logger().info(f'[{frame_id}]   Pitch (俯仰角): {pitch:+8.2f}°')
            self.get_logger().info(f'[{frame_id}]   Yaw (偏航角):   {yaw:+8.2f}°')
            
        except Exception as e:
            self.get_logger().error(f'[{frame_id}] 角度计算失败: {e}')
            roll = pitch = yaw = 0.0
            roll_euler = pitch_euler = yaw_euler = 0.0
            roll_tilt = pitch_tilt = yaw_tilt = 0.0
        
        # 6. 计算板子中心
        # 6. 计算板子中心
        # 注意：pts2d 在PnP阶段已经构建，这里复用
        center_mean = pts2d.mean(axis=0)
        center_idx = (grid_rows // 2) * grid_cols + (grid_cols // 2)
        center_mid = pts2d[min(center_idx, pts2d.shape[0]-1)]
        
        # 7. 歪斜判断
        tol = 0.5  # 角度容差
        has_tilt = (abs(roll) > tol) or (abs(pitch) > tol) or (abs(yaw) > tol)
        
        # 8. 构建结果
        result = {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'success': True,
            'apriltag_success': coord_info is not None,
            'method_used': pnp_method,
            'grid': {
                'rows_requested': self.rows,
                'cols_requested': self.cols,
                'rows_used': grid_rows,
                'cols_used': grid_cols,
                'symmetric': bool(grid_symmetric),
                'detection_source': detection_source,
                'points_detected': len(pts2d)
            },
            'apriltag_info': coord_info,
            'board_center_px': {
                'mean': {'u': float(center_mean[0]), 'v': float(center_mean[1])},
                'mid': {'u': float(center_mid[0]), 'v': float(center_mid[1])}
            },
            'euler_angles': {  # 板子相对于相机
                'roll': float(roll_euler) if 'roll_euler' in locals() else float(roll),
                'pitch': float(pitch_euler) if 'pitch_euler' in locals() else float(pitch),
                'yaw': float(yaw_euler) if 'yaw_euler' in locals() else float(yaw)
            },
            'camera_tilt_angles': {  # 相机相对于水平面
                'roll': float(roll),
                'pitch': float(pitch),
                'yaw': float(yaw)
            },
            'reprojection_error': {
                'mean': float(robust_error_mean),
                'max': float(robust_error_max),
                'method': pnp_method,
                'point_count': len(pts2d)
            },
            'tilt_detection': {
                'has_tilt': bool(has_tilt),
                'roll_offset': float(roll),
                'pitch_offset': float(pitch),
                'yaw_offset': float(yaw),
                'threshold': float(tol)
            },
            'robust_info': {
                'total_solutions_tried': pnp_info.get('total_solutions', 1),
                'all_errors': pnp_info.get('all_errors', [robust_error_mean]),
                'consistency_check': pnp_info.get('apriltag_consistency', {})
            }
        }
        
        self.success_count += 1
        self.all_results.append(result)
        
        # 9. 计算并发布变换参数（如果需要）
        if self.publish_results and self.transform_publisher is not None:
            try:
                # 计算从相机坐标系到标定板坐标系的变换参数
                delta_x, delta_y, delta_z, gamma, alpha, beta = compute_camera_to_board_transform(
                    rvec_robust, tvec_robust
                )
                
                # 构建消息：数组格式 [δx, δy, δz, γ, α, β]
                transform_msg = Float64MultiArray()
                transform_msg.data = [delta_x, delta_y, delta_z, gamma, alpha, beta]
                
                # 注意：Float64MultiArray没有header字段，如果需要时间戳信息，
                # 可以考虑使用自定义消息类型或使用data数组的前几个元素存储元数据
                
                # 发布消息
                self.transform_publisher.publish(transform_msg)
                
                self.get_logger().info(
                    f'[{frame_id}] 📤 已发布变换参数: '
                    f'平移=[{delta_x:.4f}, {delta_y:.4f}, {delta_z:.4f}]m, '
                    f'旋转=[{gamma:.4f}, {alpha:.4f}, {beta:.4f}]rad '
                    f'(ZYX欧拉角: γ={np.degrees(gamma):.2f}°, α={np.degrees(alpha):.2f}°, β={np.degrees(beta):.2f}°)'
                )
                
                # 将变换参数添加到结果中
                result['camera_to_board_transform'] = {
                    'translation': {
                        'delta_x': delta_x,
                        'delta_y': delta_y,
                        'delta_z': delta_z
                    },
                    'rotation_zyx': {
                        'gamma': gamma,  # 绕Z轴旋转（弧度）
                        'alpha': alpha,  # 绕Y轴旋转（弧度）
                        'beta': beta     # 绕X轴旋转（弧度）
                    },
                    'rotation_zyx_deg': {
                        'gamma': np.degrees(gamma),
                        'alpha': np.degrees(alpha),
                        'beta': np.degrees(beta)
                    }
                }
                
            except Exception as e:
                self.get_logger().error(f'[{frame_id}] 发布变换参数失败: {e}')
        
        # 10. 保存图像（如果需要）
        if self.save_images:
            try:
                self.get_logger().info(f'[{frame_id}] DEBUG: output_dir={self.output_dir}, frame_id={frame_id}')
                img_save_path = os.path.join(self.output_dir, 'images', f'{frame_id}_robust_result.png')
                
                # 使用自定义的可视化方法，添加详细信息
                self._visualize_and_save_with_info(
                    undistorted, ordered_corners, K_used, dist_used, 
                    rvec_robust, tvec_robust, img_save_path,
                    center_px=center_mid,
                    center_mean_px=center_mean,
                    blob_keypoints=blob_keypoints if 'blob_keypoints' in locals() else None,
                    frame_id=frame_id,
                    apriltag_success=coord_info is not None,
                    apriltag_id=coord_info.get('tag_id', 'N/A') if coord_info else 'N/A',
                    reprojection_error=robust_error_mean,
                    roll=roll,
                    pitch=pitch,
                    yaw=yaw,
                    coord_info=coord_info
                )
                
                self.get_logger().info(f'[{frame_id}] 💾 已保存可视化图像: {img_save_path}')
                
            except Exception as e:
                self.get_logger().warn(f'[{frame_id}] 保存图像失败: {e}')
        
        # 11. 日志输出总结（与tilt_checker_with_apriltag.py相同的格式）
        status = "✅ 正常" if not has_tilt else "⚠️ 存在歪斜"
        apriltag_status = "✅ AprilTag" if coord_info else "❌ AprilTag"
        error_status = "✅ 低误差" if robust_error_mean <= self.max_reprojection_error else "⚠️ 高误差"
        
        # 中心点说明：
        # - 均值中心：所有检测到的角点的平均值（算术平均）
        # - 中心(mid)：网格中心位置的实际角点（中间行、中间列的那个点）
        center_mean_str = f'均值中心(所有角点平均)(u,v)=({center_mean[0]:.1f}, {center_mean[1]:.1f})'
        center_mid_str = f'中心(mid)(网格中心角点)(u,v)=({center_mid[0]:.1f}, {center_mid[1]:.1f})'
        
        self.get_logger().info(
            f'[{frame_id}] {status} | {center_mean_str} | {center_mid_str} | '
            f'平均重投影误差: {robust_error_mean:.3f}px'
        )
        self.get_logger().info('   相机倾斜角（假设板子水平，相机相对于水平面）：')
        self.get_logger().info(f'      Roll(前后仰,绕X轴): {roll:+.2f}°')
        self.get_logger().info(f'      Pitch(平面旋,绕Z轴): {pitch:+.2f}°')
        self.get_logger().info(f'      Yaw(左右歪,绕Y轴): {yaw:+.2f}°')
        
        if coord_info is not None:
            self.get_logger().info(
                f'   AprilTag ID={coord_info["tag_id"]}, 原点索引={coord_info["origin_idx"]}'
            )
        
        self.get_logger().info(f'[{frame_id}] 🎯 结果: {status} | {apriltag_status} | {error_status}')
        self.get_logger().info(
            f'[{frame_id}] 📊 统计: 成功={self.success_count}, 失败={self.failure_count}, '
            f'AprilTag成功={self.apriltag_success_count}, 因误差淘汰={self.rejected_by_error_count}'
        )
        
        return result
    
    def _visualize_and_save_with_info(self, img_bgr, corners, K, dist, rvec, tvec, save_path, 
                                     center_px=None, center_mean_px=None, blob_keypoints=None,
                                     frame_id="", apriltag_success=False, apriltag_id="N/A",
                                     reprojection_error=0.0, roll=0.0, pitch=0.0, yaw=0.0, coord_info=None):
        """
        可视化并保存图像，在左上角添加详细信息
        
        参数:
            img_bgr: 输入图像
            corners: 检测到的角点
            K, dist: 相机内参
            rvec, tvec: 位姿
            save_path: 保存路径
            center_px: 板子中心点
            center_mean_px: 平均中心点
            blob_keypoints: blob检测点
            frame_id: 帧ID
            apriltag_success: AprilTag检测是否成功
            apriltag_id: AprilTag ID
            reprojection_error: 重投影误差
            roll, pitch, yaw: 角度信息
        """
        vis = img_bgr.copy()
        
        # 先绘制所有 blob 检测到的点（绿色，较大）
        if blob_keypoints is not None:
            for kp in blob_keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                size = float(kp.size) if hasattr(kp, 'size') else 5.0
                radius = max(2, int(round(size / 2.0)))
                cv2.circle(vis, (x, y), radius, (0, 255, 0), 2)
        
        # 然后绘制网格匹配成功的点（黄色，较小，会覆盖在绿色点上）
        for p in corners.reshape(-1,2):
            cv2.circle(vis, tuple(np.round(p).astype(int)), 3, (0,255,255), -1)
        
        # 可选：绘制板子中心
        if center_px is not None:
            c = tuple(np.round(center_px).astype(int))
            cv2.drawMarker(vis, c, (255, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=24, thickness=2)
        if center_mean_px is not None:
            cm = tuple(np.round(center_mean_px).astype(int))
            cv2.drawMarker(vis, cm, (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2)
        
        # 如果有AprilTag信息，只绘制AprilTag检测框（不绘制AprilTag坐标系）
        if coord_info is not None:
            # 只绘制AprilTag检测框，不绘制坐标系
            self._draw_apriltag_detection_only(vis, coord_info)
        
        # 绘制统一的坐标轴（基于AprilTag方向）
        vis = self._draw_axes(vis, K, dist, rvec, tvec, axis_len=150, coord_info=coord_info)
        
        # 在左上角添加详细信息
        self._add_info_overlay(vis, frame_id, apriltag_success, apriltag_id, 
                              reprojection_error, roll, pitch, yaw)
        
        cv2.imwrite(save_path, vis)
        return save_path
    
    def _draw_axes(self, img, K, dist, rvec, tvec, axis_len=100, coord_info=None):
        """
        绘制坐标轴
        如果有AprilTag信息，使用AprilTag的固定方向
        否则使用PnP求解的结果
        """
        if coord_info is not None:
            # 使用AprilTag的固定方向绘制坐标轴
            return self._draw_apriltag_based_axes(img, coord_info, axis_len)
        else:
            # 回退到PnP求解的坐标轴
            return self._draw_pnp_based_axes(img, K, dist, rvec, tvec, axis_len)
    
    def _draw_apriltag_based_axes(self, img, coord_info, axis_len=100):
        """基于AprilTag方向绘制坐标轴"""
        # 获取AprilTag坐标系信息
        x_dir = coord_info.get('x_direction_2d')
        y_dir = coord_info.get('y_direction_2d')
        reordered_corners = coord_info.get('reordered_corners')
        origin_2d = coord_info.get('origin_2d')
        
        if x_dir is None or y_dir is None or reordered_corners is None:
            self.get_logger().warn("AprilTag坐标系信息不完整，无法绘制坐标轴")
            return img
        
        if origin_2d is None:
            self.get_logger().warn("AprilTag坐标系缺少原点信息，无法绘制")
            return img
        
        # 转换为numpy数组并展平
        origin_2d = np.asarray(origin_2d, dtype=np.float64).flatten()
        x_dir = np.asarray(x_dir, dtype=np.float64).flatten()
        y_dir = np.asarray(y_dir, dtype=np.float64).flatten()
        
        # 归一化方向向量
        x_dir = x_dir / np.linalg.norm(x_dir)
        y_dir = y_dir / np.linalg.norm(y_dir)
        
        # 计算坐标轴端点
        x_end_arr = origin_2d + x_dir * axis_len
        y_end_arr = origin_2d + y_dir * axis_len
        
        # Z轴：垂直于X轴
        z_dir = np.array([-x_dir[1], x_dir[0]], dtype=np.float64)
        z_dir = z_dir / np.linalg.norm(z_dir)
        z_end_arr = origin_2d + z_dir * axis_len
        
        # 简化为直接 astype(int)
        origin = tuple(origin_2d.astype(int))
        x_end = tuple(x_end_arr.astype(int))
        y_end = tuple(y_end_arr.astype(int))
        z_end = tuple(z_end_arr.astype(int))
        
        # 绘制坐标轴
        cv2.arrowedLine(img, origin, x_end, (0, 0, 255), 3)    # X轴 - 红色
        cv2.arrowedLine(img, origin, y_end, (0, 255, 0), 3)    # Y轴 - 绿色
        cv2.arrowedLine(img, origin, z_end, (255, 0, 0), 3)    # Z轴 - 蓝色
        
        # 添加轴标签
        cv2.putText(img, 'X', x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img, 'Y', y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, 'Z', z_end, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        return img
    
    def _draw_pnp_based_axes(self, img, K, dist, rvec, tvec, axis_len=100):
        """基于PnP求解结果绘制坐标轴（回退方案）"""
        # 定义坐标轴端点
        axis = np.float32([
            [axis_len, 0, 0],      # X轴 (红色)
            [0, axis_len, 0],      # Y轴 (绿色)  
            [0, 0, -axis_len],     # Z轴 (蓝色)
            [0, 0, 0]              # 原点
        ]).reshape(-1, 3)
        
        # 投影到图像平面
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
        imgpts = imgpts.reshape(-1, 2).astype(np.int32)
        
        # 提取坐标点（直接使用numpy数组的tolist()方法转换为Python列表）
        origin = tuple(imgpts[3].tolist())
        x_end = tuple(imgpts[0].tolist())
        y_end = tuple(imgpts[1].tolist())
        z_end = tuple(imgpts[2].tolist())
        
        # 绘制坐标轴
        cv2.arrowedLine(img, origin, x_end, (0, 0, 255), 3)    # X轴 - 红色
        cv2.arrowedLine(img, origin, y_end, (0, 255, 0), 3)    # Y轴 - 绿色
        cv2.arrowedLine(img, origin, z_end, (255, 0, 0), 3)    # Z轴 - 蓝色
        
        # 添加轴标签
        cv2.putText(img, 'X', x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img, 'Y', y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, 'Z', z_end, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        return img
    
    def _build_apriltag_based_obj_points(self, rows, cols, spacing, coord_info, symmetric=True):
        """
        构建基于AprilTag坐标系的3D物体点
        如果有AprilTag信息，以离AprilTag最近的四角点为原点
        否则使用标准的网格坐标系
        """
        from src.estimate_tilt import build_obj_points
        
        # 先构建标准的3D物体点
        standard_objpoints = build_obj_points(rows, cols, spacing, symmetric)
        
        if coord_info is None:
            # 没有AprilTag信息，使用标准坐标系
            return standard_objpoints
        
        permutation = coord_info.get('corner_permutation')
        if permutation is not None and len(permutation) == len(standard_objpoints):
            ordered_objpoints = standard_objpoints[permutation]
        else:
            ordered_objpoints = standard_objpoints
        
        origin_pos = coord_info.get('origin_position', 0)
        origin_pos = int(np.clip(origin_pos, 0, len(ordered_objpoints) - 1))
        origin_3d = ordered_objpoints[origin_pos].copy()
        apriltag_based_objpoints = ordered_objpoints - origin_3d
        
        origin_idx = coord_info.get('origin_idx', 0)
        self.get_logger().info(f'使用AprilTag坐标系，原点索引: {origin_idx}, 原点3D: {origin_3d}')
        
        return apriltag_based_objpoints.astype(np.float32)
    
    def _draw_apriltag_detection_only(self, img, coord_info):
        """只绘制AprilTag检测框，不绘制坐标系"""
        if coord_info is None:
            return img
        
        # 获取AprilTag的角点
        tag_corners = coord_info.get('tag_corners')
        tag_id = coord_info.get('tag_id', 'N/A')
        
        if tag_corners is not None:
            # 绘制AprilTag边框
            tag_corners_int = np.round(tag_corners).astype(int)
            cv2.polylines(img, [tag_corners_int], True, (0, 255, 0), 2)
            
            # 在AprilTag中心添加ID标签
            center = np.mean(tag_corners, axis=0).astype(int)
            cv2.putText(img, f'ID:{tag_id}', tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return img
    
    def _add_info_overlay(self, img, frame_id, apriltag_success, apriltag_id, 
                         reprojection_error, roll, pitch, yaw):
        """在图像左上角添加信息覆盖层"""
        h, w = img.shape[:2]
        
        # 创建半透明背景
        overlay = img.copy()
        
        # 准备文本信息
        apriltag_status = "OK" if apriltag_success else "Failed"
        apriltag_color = (0, 255, 0) if apriltag_success else (0, 0, 255)  # 绿色/红色
        
        info_lines = [
            f"Name: {frame_id}",
            f"AprilTag: {apriltag_status} (ID: {apriltag_id})",
            f"Error: {reprojection_error:.3f}px",
            f"Roll: {roll:+.2f}° (X-axis, 前后仰)",
            f"Pitch: {pitch:+.2f}° (Z-axis, 平面旋)",
            f"Yaw: {yaw:+.2f}° (Y-axis, 左右歪)"
        ]
        
        # 文本参数
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        line_height = 25
        margin = 10
        
        # 计算背景矩形大小
        max_width = 0
        for line in info_lines:
            (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_width = max(max_width, text_width)
        
        bg_width = max_width + 2 * margin
        bg_height = len(info_lines) * line_height + 2 * margin
        
        # 绘制半透明背景
        cv2.rectangle(overlay, (0, 0), (bg_width, bg_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # 绘制文本
        for i, line in enumerate(info_lines):
            y_pos = margin + (i + 1) * line_height
            
            # 根据内容选择颜色
            if "AprilTag:" in line:
                color = apriltag_color
            elif "Error:" in line:
                # 根据误差大小选择颜色
                if reprojection_error > 10.0:
                    color = (0, 0, 255)  # 红色 - 高误差
                elif reprojection_error > 5.0:
                    color = (0, 165, 255)  # 橙色 - 中等误差
                else:
                    color = (0, 255, 0)  # 绿色 - 低误差
            elif any(angle_name in line for angle_name in ["Roll:", "Pitch:", "Yaw:"]):
                # 根据角度大小选择颜色
                angle_value = abs(float(line.split(':')[1].split('°')[0].strip()))
                if angle_value > 2.0:
                    color = (0, 0, 255)  # 红色 - 大角度
                elif angle_value > 1.0:
                    color = (0, 165, 255)  # 橙色 - 中等角度
                else:
                    color = (0, 255, 0)  # 绿色 - 小角度
            else:
                color = (255, 255, 255)  # 白色 - 默认
            
            cv2.putText(img, line, (margin, y_pos), font, font_scale, color, thickness)
        
        return img
    
    def image_callback(self, msg: Image):
        """处理 ROS2 图像消息"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            frame_id = msg.header.frame_id or f'frame_{self.frame_count:06d}'
            
            result = self.process_frame(cv_image, frame_id, timestamp)
            
            # 变换参数发布已在process_frame中处理
            pass
                
        except Exception as e:
            self.get_logger().error(f'处理图像消息失败: {e}')
    
    def process_image_directory(self, image_dir: str, recursive: bool = True):
        """从图像目录批量处理帧"""
        image_path = Path(image_dir)
        if not image_path.exists():
            self.get_logger().error(f'图像目录不存在: {image_path}')
            return
        
        search_iter = image_path.rglob('*') if recursive else image_path.glob('*')
        valid_ext = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        image_files = sorted([p for p in search_iter if p.is_file() and p.suffix.lower() in valid_ext])
        
        if not image_files:
            self.get_logger().error(f'在目录 {image_path} 中未找到任何图像文件 (支持: {sorted(valid_ext)})')
            return
        
        skip_frames = getattr(self, 'skip_frames', 1)
        max_frames = getattr(self, 'max_frames', None)
        
        self.get_logger().info(f'🚀 开始处理图像目录: {image_path} (共 {len(image_files)} 张)')
        processed = 0
        
        for idx, img_path in enumerate(image_files):
            if idx % skip_frames != 0:
                continue
            if max_frames is not None and processed >= max_frames:
                self.get_logger().info(f'已达到最大处理帧数 ({max_frames})，停止处理')
                break
            
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                self.get_logger().warn(f'无法读取图像: {img_path}')
                continue
            
            frame_id = img_path.stem
            timestamp = img_path.stat().st_mtime
            
            try:
                self.process_frame(img, frame_id, timestamp)
                processed += 1
                if processed % 5 == 0:
                    self.get_logger().info(
                        f'📊 已处理 {processed} 张图像，成功率: '
                        f'{(self.success_count / max(processed, 1)) * 100:.1f}%'
                    )
            except Exception as exc:
                self.get_logger().warn(f'处理图像 {img_path} 失败: {exc}')
                continue
        
        self.get_logger().info(f'✅ 图像目录处理完成，共处理 {processed} 张图像')
    
    def process_rosbag(self, bag_path: str):
        """从 rosbag 处理所有帧"""
        try:
            from rclpy.serialization import serialize_message, deserialize_message
            from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
            
            storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
            converter_options = ConverterOptions(
                input_serialization_format='cdr',
                output_serialization_format='cdr'
            )
            
            reader = SequentialReader()
            reader.open(storage_options, converter_options)
            
            topic_types = reader.get_all_topics_and_types()
            image_topic_found = False
            
            for topic_metadata in topic_types:
                if topic_metadata.name == self.image_topic:
                    image_topic_found = True
                    break
            
            if not image_topic_found:
                self.get_logger().error(f'在 rosbag 中未找到话题: {self.image_topic}')
                self.get_logger().info(f'可用话题: {[t.name for t in topic_types]}')
                return
            
            self.get_logger().info(f'🚀 开始处理 rosbag: {bag_path}')
            
            frame_idx = 0
            processed_count = 0
            skip_frames = getattr(self, 'skip_frames', 1)
            max_frames = getattr(self, 'max_frames', None)
            
            while reader.has_next():
                (topic, data, timestamp) = reader.read_next()
                
                if topic == self.image_topic:
                    if frame_idx % skip_frames != 0:
                        frame_idx += 1
                        continue
                    
                    if max_frames is not None and processed_count >= max_frames:
                        self.get_logger().info(f'已达到最大处理帧数 ({max_frames})，停止处理')
                        break
                    
                    try:
                        msg_type = None
                        for topic_metadata in topic_types:
                            if topic_metadata.name == self.image_topic:
                                msg_type = topic_metadata.type
                                break
                        
                        if msg_type == 'sensor_msgs/msg/Image':
                            msg = deserialize_message(data, Image)
                            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                        else:
                            if isinstance(data, Image):
                                cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
                            else:
                                self.get_logger().warn(f'未知的消息类型: {msg_type}')
                                frame_idx += 1
                                continue
                        
                        frame_id = f'frame_{frame_idx:06d}'
                        ts = timestamp / 1e9
                        
                        result = self.process_frame(cv_image, frame_id, ts)
                        frame_idx += 1
                        processed_count += 1
                        
                        if processed_count % 5 == 0:
                            self.get_logger().info(f'📊 已处理 {processed_count} 帧，成功率: {self.success_count/processed_count*100:.1f}%')
                            
                    except Exception as e:
                        self.get_logger().warn(f'处理帧失败: {e}')
                        frame_idx += 1
                        continue
            
            reader = None
            self.get_logger().info(f'✅ rosbag 处理完成，共处理 {processed_count} 帧')
            
        except ImportError:
            self.get_logger().error('rosbag2_py 不可用，请安装: sudo apt install ros-humble-rosbag2-py')
        except Exception as e:
            self.get_logger().error(f'处理 rosbag 失败: {e}')
    
    def save_results_to_files(self):
        """保存所有结果到文件"""
        if not self.save_results or not self.all_results:
            return
        
        self.get_logger().info('💾 保存结果到文件...')
        
        # 保存 JSON
        def convert_numpy_types(obj):
            """递归转换NumPy类型为Python原生类型"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        json_path = os.path.join(self.output_dir, 'robust_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json_data = {
                'system_info': {
                    'tag_family': self.tag_family,
                    'tag_size_mm': self.tag_size,
                    'board_spacing_mm': self.board_spacing,
                    'max_reprojection_error_px': self.max_reprojection_error,
                    'grid_size': f'{self.rows}x{self.cols}'
                },
                'summary': {
                    'total_frames': self.frame_count,
                    'success_count': self.success_count,
                    'failure_count': self.failure_count,
                    'rejected_by_error_count': self.rejected_by_error_count,
                    'apriltag_success_count': self.apriltag_success_count,
                    'apriltag_failure_count': self.apriltag_failure_count,
                    'high_error_count': self.high_error_count,
                    'fixed_error_count': self.fixed_error_count,
                    'success_rate': self.success_count / self.frame_count if self.frame_count > 0 else 0.0,
                    'apriltag_success_rate': self.apriltag_success_count / (self.apriltag_success_count + self.apriltag_failure_count) if (self.apriltag_success_count + self.apriltag_failure_count) > 0 else 0.0,
                    'rejection_rate': self.rejected_by_error_count / self.frame_count if self.frame_count > 0 else 0.0
                },
                'results': convert_numpy_types(self.all_results)
            }
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        self.get_logger().info(f'✅ 已保存 JSON 结果: {json_path}')
        
        # 保存 CSV
        csv_path = os.path.join(self.output_dir, 'detailed_results.csv')
        if self.all_results:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'frame_id', 'timestamp', 'success', 'apriltag_success', 'method_used',
                    'center_u_mean', 'center_v_mean', 'center_u_mid', 'center_v_mid',
                    'roll', 'pitch', 'yaw',
                    'reprojection_error_mean', 'has_tilt',
                    'apriltag_id', 'origin_idx', 'total_solutions_tried'
                ])
                writer.writeheader()
                for r in self.all_results:
                    apriltag_info = r.get('apriltag_info', {}) or {}
                    writer.writerow({
                        'frame_id': r['frame_id'],
                        'timestamp': r['timestamp'],
                        'success': r['success'],
                        'apriltag_success': r['apriltag_success'],
                        'method_used': r['method_used'],
                        'center_u_mean': r['board_center_px']['mean']['u'],
                        'center_v_mean': r['board_center_px']['mean']['v'],
                        'center_u_mid': r['board_center_px']['mid']['u'],
                        'center_v_mid': r['board_center_px']['mid']['v'],
                        'roll': r['camera_tilt_angles']['roll'],
                        'pitch': r['camera_tilt_angles']['pitch'],
                        'yaw': r['camera_tilt_angles']['yaw'],
                        'reprojection_error_mean': r['reprojection_error']['mean'],
                        'has_tilt': r['tilt_detection']['has_tilt'],
                        'apriltag_id': apriltag_info.get('tag_id', ''),
                        'origin_idx': apriltag_info.get('origin_idx', ''),
                        'total_solutions_tried': r['robust_info']['total_solutions_tried']
                    })
            self.get_logger().info(f'✅ 已保存 CSV 结果: {csv_path}')
        
        # 生成统计报告
        self._generate_summary_report()
    
    def _generate_summary_report(self):
        """生成统计报告"""
        if not self.all_results:
            return
        
        report_path = os.path.join(self.output_dir, 'summary_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('='*80 + '\n')
            f.write('基于鲁棒AprilTag系统的相机倾斜检测统计报告\n')
            f.write('='*80 + '\n\n')
            
            f.write('系统配置:\n')
            f.write(f'  AprilTag家族: {self.tag_family}\n')
            f.write(f'  AprilTag尺寸: {self.tag_size}mm\n')
            f.write(f'  圆点间距: {self.board_spacing}mm\n')
            f.write(f'  最大重投影误差: {self.max_reprojection_error}px\n')
            f.write(f'  网格尺寸: {self.rows}×{self.cols}\n\n')
            
            f.write('处理统计:\n')
            f.write(f'  总帧数: {self.frame_count}\n')
            f.write(f'  成功检测: {self.success_count}\n')
            f.write(f'  失败检测: {self.failure_count}\n')
            f.write(f'    - 因重投影误差超过阈值被淘汰: {self.rejected_by_error_count}\n')
            f.write(f'    - 其他原因失败: {self.failure_count - self.rejected_by_error_count}\n')
            f.write(f'  成功率: {self.success_count / self.frame_count * 100:.2f}%\n')
            f.write(f'  淘汰率: {self.rejected_by_error_count / self.frame_count * 100:.2f}%\n\n')
            
            f.write('AprilTag检测:\n')
            f.write(f'  AprilTag成功检测: {self.apriltag_success_count}\n')
            f.write(f'  AprilTag失败检测: {self.apriltag_failure_count}\n')
            total_apriltag = self.apriltag_success_count + self.apriltag_failure_count
            if total_apriltag > 0:
                f.write(f'  AprilTag成功率: {self.apriltag_success_count / total_apriltag * 100:.2f}%\n\n')
            
            if self.success_count > 0:
                # 统计角度
                rolls = [r['camera_tilt_angles']['roll'] for r in self.all_results]
                pitches = [r['camera_tilt_angles']['pitch'] for r in self.all_results]
                yaws = [r['camera_tilt_angles']['yaw'] for r in self.all_results]
                
                f.write('相机倾斜角度统计:\n')
                f.write(f'  Roll:  平均={np.mean(rolls):+.2f}°, 标准差={np.std(rolls):.2f}°, 最大={np.max(np.abs(rolls)):.2f}°\n')
                f.write(f'  Pitch: 平均={np.mean(pitches):+.2f}°, 标准差={np.std(pitches):.2f}°, 最大={np.max(np.abs(pitches)):.2f}°\n')
                f.write(f'  Yaw:   平均={np.mean(yaws):+.2f}°, 标准差={np.std(yaws):.2f}°, 最大={np.max(np.abs(yaws)):.2f}°\n\n')
                
                # 统计重投影误差（只统计通过阈值的帧）
                errors = [r['reprojection_error']['mean'] for r in self.all_results]
                f.write('重投影误差统计（仅包含通过阈值的帧）:\n')
                f.write(f'  误差阈值: {self.max_reprojection_error} 像素\n')
                f.write(f'  通过阈值的帧数: {len(errors)}\n')
                f.write(f'  被淘汰的帧数: {self.rejected_by_error_count}\n')
                f.write(f'  平均误差: {np.mean(errors):.4f} 像素\n')
                f.write(f'  最大误差: {np.max(errors):.4f} 像素\n')
                f.write(f'  最小误差: {np.min(errors):.4f} 像素\n')
                f.write(f'  标准差: {np.std(errors):.4f} 像素\n\n')
                
                # 统计使用的方法
                methods = [r['method_used'] for r in self.all_results]
                method_counts = {}
                for method in methods:
                    method_counts[method] = method_counts.get(method, 0) + 1
                
                f.write('使用的PnP方法统计:\n')
                for method, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True):
                    f.write(f'  {method}: {count} 次 ({count/len(methods)*100:.1f}%)\n')
                f.write('\n')
                
                # 统计歪斜情况
                tilted_frames = sum(1 for r in self.all_results if r['tilt_detection']['has_tilt'])
                f.write('歪斜检测:\n')
                f.write(f'  存在歪斜的帧数: {tilted_frames} ({tilted_frames/self.success_count*100:.2f}%)\n')
                f.write(f'  正常帧数: {self.success_count - tilted_frames}\n\n')
                
                # 247像素问题修复效果
                f.write('247像素重投影误差问题修复效果:\n')
                f.write(f'  高误差帧数(>50px): {self.high_error_count}\n')
                f.write(f'  修复成功帧数: {self.fixed_error_count}\n')
                if self.high_error_count > 0:
                    f.write(f'  修复成功率: {self.fixed_error_count/self.high_error_count*100:.2f}%\n')
                f.write(f'  平均重投影误差: {np.mean(errors):.3f}px (目标: <{self.max_reprojection_error}px)\n')
                
                if np.mean(errors) < self.max_reprojection_error:
                    f.write('  ✅ 成功解决247像素重投影误差问题！\n')
                else:
                    f.write('  ⚠️ 仍需进一步优化重投影误差\n')
        
        self.get_logger().info(f'✅ 已生成统计报告: {report_path}')


def main(args=None):
    """主函数"""
    parser = argparse.ArgumentParser(description='基于鲁棒AprilTag系统的相机倾斜检测节点')
    parser.add_argument('--image-topic', type=str, default='/camera/color/image_raw',
                       help='图像话题名称')
    parser.add_argument('--camera-yaml', type=str, default='config/camera_info.yaml',
                       help='相机内参 YAML 文件路径')
    parser.add_argument('--rows', type=int, default=15,
                       help='圆点行数（默认15）')
    parser.add_argument('--cols', type=int, default=15,
                       help='圆点列数（默认15）')
    parser.add_argument('--tag-family', type=str, default='tagStandard41h12',
                       help='AprilTag家族')
    parser.add_argument('--tag-size', type=float, default=0.071,
                       help='AprilTag尺寸（m，默认0.0071）')
    parser.add_argument('--board-spacing', type=float, default=0.065,
                       help='标定板圆点间距（m，默认0.065）')
    parser.add_argument('--max-error', type=float, default=1.0,
                       help='最大允许重投影误差（px，默认1.0）')
    parser.add_argument('--rosbag', type=str, default=None,
                       help='rosbag 文件路径')
    parser.add_argument('--image-dir', type=str, default=None,
                       help='图像目录（提供时将批量读取该目录的图片）')
    parser.add_argument('--output-dir', type=str, default='outputs/robust_apriltag_results',
                       help='输出目录')
    parser.add_argument('--save-images', action='store_true',
                       help='保存检测结果图像')
    parser.add_argument('--no-save-results', action='store_true',
                       help='不保存结果文件')
    parser.add_argument('--publish-results', action='store_true',
                       help='发布结果到 ROS 话题')
    parser.add_argument('--skip-frames', type=int, default=1,
                       help='跳帧处理（默认1，即处理所有帧）')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='最大处理帧数')
    
    if args is None:
        cli_args, _ = parser.parse_known_args()
    else:
        cli_args, _ = parser.parse_known_args(args)
    
    rclpy.init(args=args)
    
    node = RobustTiltCheckerNode(
        image_topic=cli_args.image_topic,
        camera_yaml_path=cli_args.camera_yaml,
        rows=cli_args.rows,
        cols=cli_args.cols,
        tag_family=cli_args.tag_family,
        tag_size=cli_args.tag_size,
        board_spacing=cli_args.board_spacing,
        max_reprojection_error=cli_args.max_error,
        output_dir=cli_args.output_dir,
        save_images=cli_args.save_images,
        save_results=not cli_args.no_save_results,
        rosbag_path=cli_args.rosbag  
    )
    
    node.skip_frames = cli_args.skip_frames
    node.max_frames = cli_args.max_frames
    
    try:
        if cli_args.rosbag:
            node.get_logger().info('检测到 --rosbag 参数，切换为 rosbag 批处理模式')
            node.process_rosbag(cli_args.rosbag)
            node.save_results_to_files()
        elif cli_args.image_dir:
            node.get_logger().info('检测到 --image-dir 参数，切换为图像目录批处理模式')
            node.process_image_directory(cli_args.image_dir)
            node.save_results_to_files()
        else:
            node.get_logger().info('未指定 rosbag 或 image-dir，进入实时 ROS 订阅模式')
            node.create_subscription(
                Image,
                cli_args.image_topic,
                node.image_callback,
                10
            )
            node.get_logger().info('等待图像消息...')
            rclpy.spin(node)
            
    except KeyboardInterrupt:
        node.get_logger().info('接收到中断信号，保存结果...')
        node.save_results_to_files()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
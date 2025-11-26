#!/usr/bin/env python3
"""
基于AprilTag坐标系的相机倾斜检测节点

功能：
1. 从 ROS2 话题或 rosbag 读取图像
2. 检测AprilTag建立统一坐标系
3. 检测圆点网格并重新排列
4. 计算相机倾斜角度（基于统一坐标系）
5. 计算重投影误差
6. 发布检测结果

使用方法:
    # 从实时话题
    python src/tilt_checker_with_apriltag.py --image-topic /camera/image_raw --camera-yaml config/camera_info.yaml
    
    # 从 rosbag
    python src/tilt_checker_with_apriltag.py --rosbag /path/to/bag --image-topic /camera/image_raw --camera-yaml config/camera_info.yaml
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
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
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import load_camera_intrinsics, scale_camera_intrinsics, get_camera_intrinsics
from src.detect_grid_improved import try_find_adaptive, refine, auto_search
from src.estimate_tilt import solve_pose_with_guess, build_obj_points
from src.calibration_and_reprojection import calculate_reprojection_errors
from src.apriltag_coordinate_system import AprilTagCoordinateSystem


class TiltCheckerWithAprilTagNode(Node):
    """基于AprilTag坐标系的相机倾斜检测节点"""
    
    def __init__(self, 
                 image_topic: str = '/camera/image_raw',
                 camera_yaml_path: str = 'config/camera_info.yaml',
                 rows: int = 15,
                 cols: int = 15,
                 spacing: float = 0.065,
                 tag_family: str = 'tagStandard41h12',
                 tag_size: float = 0.0071,
                 output_dir: str = 'outputs/apriltag_results',
                 save_images: bool = True,
                 save_results: bool = True,
                 publish_results: bool = False):
        super().__init__('tilt_checker_with_apriltag_node')
        
        self.bridge = CvBridge()
        self.image_topic = image_topic
        self.camera_yaml_path = camera_yaml_path
        self.rows = rows
        self.cols = cols
        self.spacing = spacing
        self.output_dir = output_dir
        self.save_images = save_images
        self.save_results = save_results
        self.publish_results = publish_results
        
        # 初始化AprilTag坐标系建立器
        self.coord_system = AprilTagCoordinateSystem(
            tag_family=tag_family,
            tag_size=tag_size,
            board_spacing=spacing
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
        self.all_results = []
        
        # 创建输出目录
        if self.save_results or self.save_images:
            os.makedirs(self.output_dir, exist_ok=True)
            if self.save_images:
                os.makedirs(os.path.join(self.output_dir, 'images'), exist_ok=True)
        
        self.get_logger().info('='*60)
        self.get_logger().info('基于AprilTag的相机倾斜检测节点已启动')
        self.get_logger().info(f'  图像话题: {self.image_topic}')
        self.get_logger().info(f'  相机内参: {self.camera_yaml_path}')
        self.get_logger().info(f'  网格尺寸: {self.rows} x {self.cols}')
        self.get_logger().info(f'  AprilTag家族: {tag_family}')
        self.get_logger().info(f'  AprilTag尺寸: {tag_size}mm')
        self.get_logger().info(f'  输出目录: {self.output_dir}')
        self.get_logger().info('='*60)
    
    def _load_camera_intrinsics(self):
        """加载相机内参"""
        try:
            K, dist, image_size = load_camera_intrinsics(self.camera_yaml_path)
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
        处理单帧图像（基于AprilTag坐标系）
        
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
        else:
            undistorted = cv_image.copy()
            K_used, dist_used = get_camera_intrinsics(h, w, yaml_path=None, f_scale=1.0)
            self.get_logger().debug(f'[{frame_id}] 使用默认内参')
        
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        
        # 2. 检测圆点网格（初步检测）
        grid_rows = self.rows
        grid_cols = self.cols
        grid_symmetric = True
        detection_source = 'direct'
        
        try:
            ok, corners, blob_keypoints = try_find_adaptive(gray, grid_rows, grid_cols, symmetric=grid_symmetric)
            
            if (not ok) or (corners is None):
                self.get_logger().warn(f'[{frame_id}] 未检测到完整网格，尝试降级搜索...')
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
                        f'[{frame_id}] ✅ 降级搜索成功，使用 {grid_rows}x{grid_cols} 网格'
                    )
                else:
                    ok = False
            
            if not ok or corners is None:
                self.get_logger().warn(f'[{frame_id}] 未检测到网格')
                self.failure_count += 1
                return None
            
            # 精化角点
            corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
            corners_refined = refine(gray, corners)
            board_corners_2d = corners_refined.reshape(-1, 2)
            
        except Exception as e:
            self.get_logger().error(f'[{frame_id}] 网格检测失败: {e}')
            self.failure_count += 1
            return None
        
        # 3. 基于AprilTag建立坐标系
        try:
            coord_success, origin_2d, x_direction, y_direction, coord_info = self.coord_system.establish_coordinate_system(
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
                ordered_corners = coord_info['reordered_corners'].reshape(-1, 1, 2)
                
        except Exception as e:
            self.get_logger().warn(f'[{frame_id}] AprilTag处理失败: {e}，使用原始检测结果')
            self.apriltag_failure_count += 1
            ordered_corners = corners_refined
            coord_info = None
        
        # 4. 计算位姿和角度
        try:
            rvec, tvec, angles_dict, K_final, dist_final, _ = solve_pose_with_guess(
                gray, ordered_corners, grid_rows, grid_cols, 
                spacing=self.spacing, symmetric=grid_symmetric, 
                K=K_used, dist=dist_used
            )
        except Exception as e:
            self.get_logger().error(f'[{frame_id}] 位姿计算失败: {e}')
            self.failure_count += 1
            return None

        pts2d = ordered_corners.reshape(-1, 2)
        center_mean = pts2d.mean(axis=0)
        center_idx = (grid_rows // 2) * grid_cols + (grid_cols // 2)
        center_mid = pts2d[min(center_idx, pts2d.shape[0]-1)]
        
        roll_euler, pitch_euler, yaw_euler = angles_dict['euler']
        roll_tilt, pitch_tilt, yaw_tilt = angles_dict['camera_tilt']
        symmetry_info = angles_dict.get('symmetry', {})
        
        # 5. 计算重投影误差
        all_errors = []
        try:
            objp = build_obj_points(grid_rows, grid_cols, self.spacing, symmetric=grid_symmetric)
            objpoints = [objp]
            imgpoints = [ordered_corners]
            rvecs = [rvec]
            tvecs = [tvec]
            
            errors_per_image, all_errors, residual_vectors = calculate_reprojection_errors(
                objpoints, imgpoints, rvecs, tvecs, K_final, dist_final
            )
            
            mean_error = errors_per_image[0] if errors_per_image else 0.0
            max_error = max(all_errors) if all_errors else 0.0
            
            # 异常重投影误差检测和警告
            if coord_success and mean_error > 50.0:  # 基于AprilTag的帧误差应该很小
                self.get_logger().warn(
                    f'[{frame_id}] ⚠️ AprilTag坐标系下重投影误差异常大: {mean_error:.1f}px '
                    f'(最大: {max_error:.1f}px) - 可能存在角点重排问题'
                )
            elif not coord_success and mean_error > 200.0:  # 传统方法的异常阈值
                self.get_logger().warn(
                    f'[{frame_id}] ⚠️ 传统方法重投影误差异常大: {mean_error:.1f}px '
                    f'(最大: {max_error:.1f}px) - 可能存在检测错误'
                )
            
        except Exception as e:
            self.get_logger().warn(f'[{frame_id}] 重投影误差计算失败: {e}')
            mean_error = 0.0
            max_error = 0.0
            residual_vectors = []
        
        # 6. 歪斜判断
        tol = 0.5
        has_tilt = (abs(roll_tilt) > tol) or (abs(pitch_tilt) > tol) or (abs(yaw_tilt) > tol)
        
        # 7. 构建结果
        result = {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'success': True,
            'apriltag_success': coord_success,
            'grid': {
                'rows_requested': self.rows,
                'cols_requested': self.cols,
                'rows_used': grid_rows,
                'cols_used': grid_cols,
                'symmetric': bool(grid_symmetric),
                'detection_source': detection_source
            },
            'apriltag_info': coord_info,
            'board_center_px': {
                'mean': {'u': float(center_mean[0]), 'v': float(center_mean[1])},
                'mid': {'u': float(center_mid[0]), 'v': float(center_mid[1])}
            },
            'euler_angles': {
                'roll': float(roll_euler),
                'pitch': float(pitch_euler),
                'yaw': float(yaw_euler)
            },
            'camera_tilt_angles': {
                'roll': float(roll_tilt),
                'pitch': float(pitch_tilt),
                'yaw': float(yaw_tilt)
            },
            'reprojection_error': {
                'mean': float(mean_error),
                'max': float(max_error),
                'point_count': len(all_errors)
            },
            'symmetry_transform': symmetry_info,
            'tilt_detection': {
                'has_tilt': bool(has_tilt),
                'roll_offset': float(roll_tilt),
                'pitch_offset': float(pitch_tilt),
                'yaw_offset': float(yaw_tilt),
                'threshold': float(tol)
            }
        }
        
        self.success_count += 1
        self.all_results.append(result)
        
        # 8. 保存图像（如果需要）
        if self.save_images:
            try:
                img_save_path = os.path.join(self.output_dir, 'images', f'{frame_id}_result.png')
                
                # 创建可视化图像
                vis_image = undistorted.copy()
                
                # 绘制检测到的角点
                for corner in pts2d:
                    cv2.circle(vis_image, tuple(corner.astype(int)), 3, (255, 0, 0), -1)
                
                # 如果有AprilTag信息，绘制坐标系
                if coord_info is not None:
                    vis_image = self.coord_system.visualize_coordinate_system(vis_image, coord_info)
                
                # 绘制中心点
                cv2.circle(vis_image, tuple(center_mid.astype(int)), 8, (0, 255, 255), -1)
                
                # 添加文本信息
                info_text = [
                    f"Frame: {frame_id}",
                    f"AprilTag: {'OK' if coord_success else 'FAIL'}",
                    f"Roll: {roll_tilt:+.2f}°",
                    f"Pitch: {pitch_tilt:+.2f}°",
                    f"Yaw: {yaw_tilt:+.2f}°",
                    f"Error: {mean_error:.3f}px"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(vis_image, text, (10, 30 + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imwrite(img_save_path, vis_image)
                
            except Exception as e:
                self.get_logger().warn(f'保存图像失败: {e}')
        
        # 9. 日志输出
        status = "✅ 正常" if not has_tilt else "⚠️ 存在歪斜"
        apriltag_status = "✅ AprilTag" if coord_success else "❌ AprilTag"
        
        self.get_logger().info(
            f'[{frame_id}] {status} | {apriltag_status} | '
            f'中心: ({center_mid[0]:.1f}, {center_mid[1]:.1f}) | '
            f'重投影误差: {mean_error:.3f}px'
        )
        self.get_logger().info(
            f'   相机倾斜角: Roll={roll_tilt:+.2f}° Pitch={pitch_tilt:+.2f}° Yaw={yaw_tilt:+.2f}°'
        )
        
        if coord_info is not None:
            self.get_logger().info(
                f'   AprilTag ID={coord_info["tag_id"]}, 原点索引={coord_info["origin_idx"]}'
            )
        
        return result
    
    def image_callback(self, msg: Image):
        """处理 ROS2 图像消息"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            frame_id = msg.header.frame_id or f'frame_{self.frame_count:06d}'
            
            result = self.process_frame(cv_image, frame_id, timestamp)
            
            if self.publish_results and result:
                # TODO: 发布自定义消息类型
                pass
                
        except Exception as e:
            self.get_logger().error(f'处理图像消息失败: {e}')
    
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
            
            self.get_logger().info(f'开始处理 rosbag: {bag_path}')
            
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
                        
                        if processed_count % 10 == 0:
                            self.get_logger().info(f'已处理 {processed_count} 帧...')
                            
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
        
        # 保存 JSON (处理NumPy数组序列化)
        def convert_numpy_types(obj):
            """递归转换NumPy类型为Python原生类型"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        json_path = os.path.join(self.output_dir, 'results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json_data = {
                'summary': {
                    'total_frames': self.frame_count,
                    'success_count': self.success_count,
                    'failure_count': self.failure_count,
                    'apriltag_success_count': self.apriltag_success_count,
                    'apriltag_failure_count': self.apriltag_failure_count,
                    'success_rate': self.success_count / self.frame_count if self.frame_count > 0 else 0.0,
                    'apriltag_success_rate': self.apriltag_success_count / (self.apriltag_success_count + self.apriltag_failure_count) if (self.apriltag_success_count + self.apriltag_failure_count) > 0 else 0.0
                },
                'results': convert_numpy_types(self.all_results)
            }
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        self.get_logger().info(f'✅ 已保存 JSON 结果: {json_path}')
        
        # 保存 CSV
        csv_path = os.path.join(self.output_dir, 'results.csv')
        if self.all_results:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'frame_id', 'timestamp', 'success', 'apriltag_success',
                    'center_u_mean', 'center_v_mean', 'center_u_mid', 'center_v_mid',
                    'roll_euler', 'pitch_euler', 'yaw_euler',
                    'roll_tilt', 'pitch_tilt', 'yaw_tilt',
                    'reprojection_error_mean', 'reprojection_error_max',
                    'has_tilt', 'roll_offset', 'pitch_offset', 'yaw_offset',
                    'apriltag_id', 'origin_idx'
                ])
                writer.writeheader()
                for r in self.all_results:
                    apriltag_info = r.get('apriltag_info', {}) or {}
                    writer.writerow({
                        'frame_id': r['frame_id'],
                        'timestamp': r['timestamp'],
                        'success': r['success'],
                        'apriltag_success': r['apriltag_success'],
                        'center_u_mean': r['board_center_px']['mean']['u'],
                        'center_v_mean': r['board_center_px']['mean']['v'],
                        'center_u_mid': r['board_center_px']['mid']['u'],
                        'center_v_mid': r['board_center_px']['mid']['v'],
                        'roll_euler': r['euler_angles']['roll'],
                        'pitch_euler': r['euler_angles']['pitch'],
                        'yaw_euler': r['euler_angles']['yaw'],
                        'roll_tilt': r['camera_tilt_angles']['roll'],
                        'pitch_tilt': r['camera_tilt_angles']['pitch'],
                        'yaw_tilt': r['camera_tilt_angles']['yaw'],
                        'reprojection_error_mean': r['reprojection_error']['mean'],
                        'reprojection_error_max': r['reprojection_error']['max'],
                        'has_tilt': r['tilt_detection']['has_tilt'],
                        'roll_offset': r['tilt_detection']['roll_offset'],
                        'pitch_offset': r['tilt_detection']['pitch_offset'],
                        'yaw_offset': r['tilt_detection']['yaw_offset'],
                        'apriltag_id': apriltag_info.get('tag_id', ''),
                        'origin_idx': apriltag_info.get('origin_idx', '')
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
            f.write('='*60 + '\n')
            f.write('基于AprilTag的相机倾斜检测统计报告\n')
            f.write('='*60 + '\n\n')
            
            f.write(f'总帧数: {self.frame_count}\n')
            f.write(f'成功检测: {self.success_count}\n')
            f.write(f'失败检测: {self.failure_count}\n')
            f.write(f'成功率: {self.success_count / self.frame_count * 100:.2f}%\n\n')
            
            f.write(f'AprilTag成功检测: {self.apriltag_success_count}\n')
            f.write(f'AprilTag失败检测: {self.apriltag_failure_count}\n')
            total_apriltag = self.apriltag_success_count + self.apriltag_failure_count
            if total_apriltag > 0:
                f.write(f'AprilTag成功率: {self.apriltag_success_count / total_apriltag * 100:.2f}%\n\n')
            
            if self.success_count > 0:
                # 分别统计有AprilTag和无AprilTag的结果
                apriltag_results = [r for r in self.all_results if r['apriltag_success']]
                no_apriltag_results = [r for r in self.all_results if not r['apriltag_success']]
                
                f.write('相机倾斜角度统计:\n')
                
                if apriltag_results:
                    roll_tilts = [r['camera_tilt_angles']['roll'] for r in apriltag_results]
                    pitch_tilts = [r['camera_tilt_angles']['pitch'] for r in apriltag_results]
                    yaw_tilts = [r['camera_tilt_angles']['yaw'] for r in apriltag_results]
                    
                    f.write(f'  基于AprilTag坐标系 ({len(apriltag_results)} 帧):\n')
                    f.write(f'    Roll:  平均={np.mean(roll_tilts):+.2f}°, 最大={np.max(np.abs(roll_tilts)):.2f}°\n')
                    f.write(f'    Pitch: 平均={np.mean(pitch_tilts):+.2f}°, 最大={np.max(np.abs(pitch_tilts)):.2f}°\n')
                    f.write(f'    Yaw:   平均={np.mean(yaw_tilts):+.2f}°, 最大={np.max(np.abs(yaw_tilts)):.2f}°\n\n')
                
                if no_apriltag_results:
                    roll_tilts = [r['camera_tilt_angles']['roll'] for r in no_apriltag_results]
                    pitch_tilts = [r['camera_tilt_angles']['pitch'] for r in no_apriltag_results]
                    yaw_tilts = [r['camera_tilt_angles']['yaw'] for r in no_apriltag_results]
                    
                    f.write(f'  传统方法 ({len(no_apriltag_results)} 帧):\n')
                    f.write(f'    Roll:  平均={np.mean(roll_tilts):+.2f}°, 最大={np.max(np.abs(roll_tilts)):.2f}°\n')
                    f.write(f'    Pitch: 平均={np.mean(pitch_tilts):+.2f}°, 最大={np.max(np.abs(pitch_tilts)):.2f}°\n')
                    f.write(f'    Yaw:   平均={np.mean(yaw_tilts):+.2f}°, 最大={np.max(np.abs(yaw_tilts)):.2f}°\n\n')
                
                # 统计重投影误差
                errors = [r['reprojection_error']['mean'] for r in self.all_results]
                f.write('重投影误差统计:\n')
                f.write(f'  平均误差: {np.mean(errors):.4f} 像素\n')
                f.write(f'  最大误差: {np.max(errors):.4f} 像素\n')
                f.write(f'  最小误差: {np.min(errors):.4f} 像素\n\n')
                
                # 统计歪斜情况
                tilted_frames = sum(1 for r in self.all_results if r['tilt_detection']['has_tilt'])
                f.write('歪斜检测:\n')
                f.write(f'  存在歪斜的帧数: {tilted_frames} ({tilted_frames/self.success_count*100:.2f}%)\n')
                f.write(f'  正常帧数: {self.success_count - tilted_frames}\n')
        
        self.get_logger().info(f'✅ 已生成统计报告: {report_path}')


def main(args=None):
    """主函数"""
    parser = argparse.ArgumentParser(description='基于AprilTag的相机倾斜检测节点')
    parser.add_argument('--image-topic', type=str, default='/camera/image_raw',
                       help='图像话题名称')
    parser.add_argument('--camera-yaml', type=str, default='config/camera_info.yaml',
                       help='相机内参 YAML 文件路径')
    parser.add_argument('--rows', type=int, default=15,
                       help='圆点行数（默认15）')
    parser.add_argument('--cols', type=int, default=15,
                       help='圆点列数（默认15）')
    parser.add_argument('--spacing', type=float, default=0.065,
                       help='圆点间距（m，默认0.065')
    parser.add_argument('--tag-family', type=str, default='tagStandard41h12',
                       choices=['tag36h11', 'tag25h9', 'tag16h5', 'tagStandard41h12', 'TagStandard41h12'],
                       help='AprilTag家族')
    parser.add_argument('--tag-size', type=float, default=0.0071,
                       help='AprilTag尺寸（m，默认0.0071）')
    parser.add_argument('--rosbag', type=str, default=None,
                       help='rosbag 文件路径')
    parser.add_argument('--output-dir', type=str, default='outputs/apriltag_results',
                       help='输出目录')
    parser.add_argument('--save-images', action='store_true',
                       help='保存检测结果图像')
    parser.add_argument('--no-save-results', action='store_true',
                       help='不保存结果文件')
    parser.add_argument('--publish-results', action='store_true',
                       help='发布结果到 ROS 话题')
    parser.add_argument('--skip-frames', type=int, default=1,
                       help='跳帧处理')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='最大处理帧数')
    
    if args is None:
        cli_args, _ = parser.parse_known_args()
    else:
        cli_args, _ = parser.parse_known_args(args)
    
    rclpy.init(args=args)
    
    node = TiltCheckerWithAprilTagNode(
        image_topic=cli_args.image_topic,
        camera_yaml_path=cli_args.camera_yaml,
        rows=cli_args.rows,
        cols=cli_args.cols,
        spacing=cli_args.spacing,
        tag_family=cli_args.tag_family,
        tag_size=cli_args.tag_size,
        output_dir=cli_args.output_dir,
        save_images=cli_args.save_images,
        save_results=not cli_args.no_save_results,
        publish_results=cli_args.publish_results
    )
    
    node.skip_frames = cli_args.skip_frames
    node.max_frames = cli_args.max_frames
    
    try:
        if cli_args.rosbag:
            node.process_rosbag(cli_args.rosbag)
            node.save_results_to_files()
        else:
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
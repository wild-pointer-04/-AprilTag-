#!/usr/bin/env python3
"""
相机倾斜检测 ROS2 节点

功能：
1. 从 ROS2 话题或 rosbag 读取图像
2. 对图像进行畸变矫正
3. 检测圆点网格
4. 计算相机倾斜角度
5. 计算重投影误差
6. 发布检测结果（ROS 消息或保存文件）

使用方法:
    # 从实时话题
    python src/tilt_checker_node.py --image-topic /camera/image_raw --camera-yaml config/camera_info.yaml
    
    # 从 rosbag
    python src/tilt_checker_node.py --rosbag /path/to/bag --image-topic /camera/image_raw --camera-yaml config/camera_info.yaml
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


class TiltCheckerNode(Node):
    """相机倾斜检测节点"""
    
    def __init__(self, 
                 image_topic: str = '/camera/image_raw',
                 camera_yaml_path: str = 'config/camera_info.yaml',
                 rows: int = 15,
                 cols: int = 15,
                 spacing: float = 10.0,
                 output_dir: str = 'outputs/rosbag_results',
                 save_images: bool = True,
                 save_results: bool = True,
                 publish_results: bool = False):
        super().__init__('tilt_checker_node')
        
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
        
        # 加载相机内参
        self.K = None
        self.dist = None
        self.image_size = None  # YAML 中记录的图像尺寸 (width, height)
        self._load_camera_intrinsics()
        
        # 统计信息
        self.frame_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.all_results = []
        
        # 创建输出目录
        if self.save_results or self.save_images:
            os.makedirs(self.output_dir, exist_ok=True)
            if self.save_images:
                os.makedirs(os.path.join(self.output_dir, 'images'), exist_ok=True)
        
        self.get_logger().info('='*60)
        self.get_logger().info('相机倾斜检测节点已启动')
        self.get_logger().info(f'  图像话题: {self.image_topic}')
        self.get_logger().info(f'  相机内参: {self.camera_yaml_path}')
        self.get_logger().info(f'  网格尺寸: {self.rows} x {self.cols}')
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
                self.image_size = image_size  # 保存 YAML 中记录的图像尺寸
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
        处理单帧图像
        
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
            timestamp = self.frame_count * 0.1  # 假设 10 FPS
        
        h, w = cv_image.shape[:2]
        actual_size = (w, h)  # 实际图像尺寸 (width, height)
        
        # 1. 获取并自动缩放相机内参（如果需要）
        if self.K is not None and self.dist is not None:
            # 检查图像尺寸是否与 YAML 中记录的不匹配
            if self.image_size is not None:
                yaml_size = self.image_size  # (width, height)
                if yaml_size[0] != w or yaml_size[1] != h:
                    # 图像尺寸不匹配，自动缩放内参
                    self.get_logger().info(
                        f'[{frame_id}] 检测到图像尺寸不匹配: '
                        f'YAML中为 {yaml_size[0]} x {yaml_size[1]}, '
                        f'实际为 {w} x {h}'
                    )
                    K_used, dist_used = scale_camera_intrinsics(
                        self.K, self.dist, yaml_size, actual_size
                    )
                    self.get_logger().info(
                        f'[{frame_id}] 已自动缩放内参矩阵以适应新分辨率 '
                        f'(缩放比例: {w/yaml_size[0]:.3f} x {h/yaml_size[1]:.3f})'
                    )
                else:
                    # 尺寸匹配，直接使用原始内参
                    K_used = self.K.copy()
                    dist_used = self.dist.copy()
            else:
                # YAML 中未记录图像尺寸，直接使用（假设匹配）
                K_used = self.K.copy()
                dist_used = self.dist.copy()
                self.get_logger().debug(
                    f'[{frame_id}] YAML中未记录图像尺寸，直接使用内参 '
                    f'(假设图像尺寸为 {w} x {h})'
                )
            
            # 使用缩放后的内参进行畸变矫正
            undistorted = cv2.undistort(cv_image, K_used, dist_used)
        else:
            # 如果没有内参，使用原始图像和默认内参
            undistorted = cv_image.copy()
            K_used, dist_used = get_camera_intrinsics(h, w, yaml_path=None, f_scale=1.0)
            self.get_logger().debug(f'[{frame_id}] 使用默认内参')
        
        # 转换为灰度图
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        
        # 2. 检测圆点网格
        grid_rows = self.rows
        grid_cols = self.cols
        grid_symmetric = True
        detection_source = 'direct'
        try:
            ok, corners, blob_keypoints = try_find_adaptive(gray, grid_rows, grid_cols, symmetric=grid_symmetric)
            
            if (not ok) or (corners is None):
                self.get_logger().warn(f'[{frame_id}] 未检测到完整 {self.rows*self.cols} 网格，尝试降级搜索局部子网格...')
                rows_range = (max(4, self.rows - 6), self.rows)
                cols_range = (max(4, self.cols - 6), self.cols)
                auto_ok, auto_corners, meta, blob_keypoints = auto_search(
                    gray,
                    rows_range=rows_range,
                    cols_range=cols_range
                )
                if auto_ok and auto_corners is not None and meta is not None:
                    ok = True
                    corners = auto_corners.reshape(-1, 1, 2)
                    grid_rows, grid_cols, grid_symmetric = meta
                    detection_source = 'fallback'
                    self.get_logger().info(
                        f'[{frame_id}] ✅ 降级搜索成功，使用 {grid_rows}x{grid_cols} 网格 (symmetric={grid_symmetric}) '
                        f'({len(corners)} 个点)'
                    )
                else:
                    ok = False
            
            if not ok or corners is None:
                self.get_logger().warn(f'[{frame_id}] 未检测到网格')
                self.failure_count += 1
                return None
            
            expected_pts = grid_rows * grid_cols
            if len(corners) != expected_pts:
                self.get_logger().warn(
                    f'[{frame_id}] 检测到 {len(corners)} 个点，但当前网格设置为 {grid_rows}x{grid_cols}={expected_pts} 个，'
                    ' 无法建立稳定坐标系。'
                )
                self.failure_count += 1
                return None
            
            # 精化角点
            corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
            corners_refined = refine(gray, corners)
            
        except Exception as e:
            self.get_logger().error(f'[{frame_id}] 检测失败: {e}')
            self.failure_count += 1
            return None
        
        # 3. 计算位姿和角度（使用已缩放的内参，内部自动搜索对称排列）
        try:
            rvec, tvec, angles_dict, K_final, dist_final, ordered_corners = solve_pose_with_guess(
                gray, corners_refined, grid_rows, grid_cols, 
                spacing=self.spacing, symmetric=grid_symmetric, 
                K=K_used, dist=dist_used  # 直接使用已缩放的内参
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
        
        # 对称圆点网格存在 90° 方向歧义：对 yaw 和 pitch 做 mod 90 归一化，仅用于显示与判定
        def _normalize_mod90(angle_deg: float):
            k = int(round(angle_deg / 90.0))
            return angle_deg - 90.0 * k
        yaw_tilt_mod90 = _normalize_mod90(yaw_tilt)
        pitch_tilt_mod90 = _normalize_mod90(pitch_tilt)
        
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
            
        except Exception as e:
            self.get_logger().warn(f'[{frame_id}] 重投影误差计算失败: {e}')
            mean_error = 0.0
            max_error = 0.0
            residual_vectors = []
        
        # 6. 歪斜判断（使用归一化后的角度）
        tol = 0.5  # 阈值：0.5度
        has_tilt = (abs(roll_tilt) > tol) or (abs(pitch_tilt_mod90) > tol) or (abs(yaw_tilt_mod90) > tol)
        
        # 7. 构建结果
        result = {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'success': True,
            'grid': {
                'rows_requested': self.rows,
                'cols_requested': self.cols,
                'rows_used': grid_rows,
                'cols_used': grid_cols,
                'symmetric': bool(grid_symmetric),
                'detection_source': detection_source
            },
            'board_center_px': {
                'mean': {'u': float(center_mean[0]), 'v': float(center_mean[1])},
                'mid': {'u': float(center_mid[0]), 'v': float(center_mid[1])}
            },
            'euler_angles': {  # 板子相对于相机
                'roll': float(roll_euler),
                'pitch': float(pitch_euler),
                'yaw': float(yaw_euler)
            },
            'camera_tilt_angles': {  # 相机相对于水平面（假设板子水平）
                'roll': float(roll_tilt),
                'pitch': float(pitch_tilt),
                'yaw': float(yaw_tilt)
            },
            'camera_tilt_angles_mod': {  # 归一化后的角度（pitch 和 yaw 按 90° 归一化，仅用于对称网格）
                'roll': float(roll_tilt),
                'pitch': float(pitch_tilt),
                'pitch_mod90': float(pitch_tilt_mod90),
                'yaw': float(yaw_tilt),
                'yaw_mod90': float(yaw_tilt_mod90)
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
                'pitch_offset_mod90': float(pitch_tilt_mod90),
                'yaw_offset': float(yaw_tilt),
                'yaw_offset_mod90': float(yaw_tilt_mod90),
                'threshold': float(tol)
            }
        }
        
        self.success_count += 1
        self.all_results.append(result)
        
        # 8. 保存图像（如果需要）
        if self.save_images:
            try:
                from src.estimate_tilt import visualize_and_save
                img_save_path = os.path.join(self.output_dir, 'images', f'{frame_id}_result.png')
                visualize_and_save(
                    undistorted, ordered_corners, K_final, dist_final, 
                    rvec, tvec, img_save_path,
                    center_px=center_mid,
                    center_mean_px=center_mean,
                    blob_keypoints=blob_keypoints
                )
            except Exception as e:
                self.get_logger().warn(f'保存图像失败: {e}')
        
        # 9. 日志输出
        status = "✅ 正常" if not has_tilt else "⚠️ 存在歪斜"
        # 中心点说明：
        # - 均值中心：所有检测到的角点的平均值（算术平均）
        # - 中心(mid)：网格中心位置的实际角点（中间行、中间列的那个点）
        center_mean_str = f'均值中心(所有角点平均)(u,v)=({center_mean[0]:.1f}, {center_mean[1]:.1f})'
        center_mid_str = f'中心(mid)(网格中心角点)(u,v)=({center_mid[0]:.1f}, {center_mid[1]:.1f})'
        symmetry_desc = symmetry_info.get('transform', 'N/A')
        self.get_logger().info(
            f'[{frame_id}] {status} | {center_mean_str} | {center_mid_str} | 对称解:{symmetry_desc} | '
            f'平均重投影误差: {mean_error:.3f}px'
        )
        self.get_logger().info('   标准欧拉角（板子相对于相机，XYZ顺序）：')
        self.get_logger().info(f'      Roll(前后仰,绕X轴): {roll_euler:+.2f}°')
        self.get_logger().info(f'      Pitch(平面旋,绕Z轴): {pitch_euler:+.2f}°')
        self.get_logger().info(f'      Yaw(左右歪,绕Y轴): {yaw_euler:+.2f}°')
        self.get_logger().info('   相机倾斜角（假设板子水平，相机相对于水平面）：')
        self.get_logger().info(f'      Roll(前后仰,绕X轴): {roll_tilt:+.2f}°')
        self.get_logger().info(f'      Pitch(平面旋,绕Z轴): {pitch_tilt:+.2f}° (mod90={pitch_tilt_mod90:+.2f}°)')
        self.get_logger().info(f'      Yaw(左右歪,绕Y轴): {yaw_tilt:+.2f}° (mod90={yaw_tilt_mod90:+.2f}°)')
        if has_tilt:
            self.get_logger().info(
                f'   -> Roll偏移(前后仰,绕X轴): {roll_tilt:+.3f}° | '
                f'Pitch偏移(平面旋,绕Z轴): {pitch_tilt:+.3f}° (mod90={pitch_tilt_mod90:+.3f}°) | '
                f'Yaw偏移(左右歪,绕Y轴): {yaw_tilt:+.3f}° (mod90={yaw_tilt_mod90:+.3f}°) | '
                f'阈值: ±{tol:.1f}°'
            )
        
        return result
    
    def image_callback(self, msg: Image):
        """处理 ROS2 图像消息"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            frame_id = msg.header.frame_id or f'frame_{self.frame_count:06d}'
            
            result = self.process_frame(cv_image, frame_id, timestamp)
            
            # 如果启用发布，可以在这里发布结果消息
            if self.publish_results and result:
                # TODO: 发布自定义消息类型
                pass
                
        except Exception as e:
            self.get_logger().error(f'处理图像消息失败: {e}')
    
    def process_rosbag(self, bag_path: str):
        """从 rosbag 处理所有帧"""
        try:
            # 尝试使用 rosbag2 API
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
                self.get_logger().info(f'图像话题: {self.image_topic}')
                
                frame_idx = 0
                processed_count = 0
                skip_frames = getattr(self, 'skip_frames', 1)
                max_frames = getattr(self, 'max_frames', None)
                
                while reader.has_next():
                    (topic, data, timestamp) = reader.read_next()
                    
                    if topic == self.image_topic:
                        # 跳帧逻辑
                        if frame_idx % skip_frames != 0:
                            frame_idx += 1
                            continue
                        
                        # 最大帧数限制
                        if max_frames is not None and processed_count >= max_frames:
                            self.get_logger().info(f'已达到最大处理帧数 ({max_frames})，停止处理')
                            break
                        
                        try:
                            # rosbag2_py 返回的 data 是序列化的字节
                            # 需要根据消息类型反序列化
                            msg_type = None
                            for topic_metadata in topic_types:
                                if topic_metadata.name == self.image_topic:
                                    msg_type = topic_metadata.type
                                    break
                            
                            if msg_type == 'sensor_msgs/msg/Image':
                                # 反序列化 Image 消息
                                msg = deserialize_message(data, Image)
                                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                            else:
                                # 尝试直接使用 cv_bridge（如果 data 已经是消息对象）
                                if isinstance(data, Image):
                                    cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
                                else:
                                    self.get_logger().warn(f'未知的消息类型: {msg_type}')
                                    frame_idx += 1
                                    continue
                            
                            frame_id = f'frame_{frame_idx:06d}'
                            ts = timestamp / 1e9  # 转换为秒
                            
                            result = self.process_frame(cv_image, frame_id, ts)
                            frame_idx += 1
                            processed_count += 1
                            
                            if processed_count % 10 == 0:
                                self.get_logger().info(f'已处理 {processed_count} 帧 (总共读取 {frame_idx} 帧)...')
                                
                        except Exception as e:
                            self.get_logger().warn(f'处理帧失败: {e}')
                            import traceback
                            self.get_logger().debug(traceback.format_exc())
                            frame_idx += 1
                            continue
                
                # rosbag2_py SequentialReader 在部分版本中无 close() 方法，依赖析构释放资源
                reader = None
                self.get_logger().info(f'✅ rosbag 处理完成，共读取 {frame_idx} 帧，实际处理 {processed_count} 帧')
                
            except ImportError:
                # 回退到使用 rosbag2 命令行工具
                self.get_logger().warn('rosbag2_py 不可用，尝试使用 rosbag2 命令行工具')
                self._process_rosbag_with_cli(bag_path)
                
        except Exception as e:
            self.get_logger().error(f'处理 rosbag 失败: {e}')
            import traceback
            traceback.print_exc()
    
    def _process_rosbag_with_cli(self, bag_path: str):
        """使用 rosbag2 命令行工具处理（回退方案）"""
        import subprocess
        import tempfile
        
        self.get_logger().info('使用 rosbag2 命令行工具提取图像...')
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 使用 rosbag2 提取图像话题
            # 注意：这需要 rosbag2 命令行工具支持
            # 实际实现可能需要根据具体需求调整
            
            self.get_logger().warn('rosbag2 命令行提取功能需要根据实际环境实现')
            self.get_logger().info('建议安装: sudo apt install ros-humble-rosbag2-py')
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def save_results_to_files(self):
        """保存所有结果到文件"""
        if not self.save_results or not self.all_results:
            return
        
        # 保存 JSON
        json_path = os.path.join(self.output_dir, 'results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_frames': self.frame_count,
                    'success_count': self.success_count,
                    'failure_count': self.failure_count,
                    'success_rate': self.success_count / self.frame_count if self.frame_count > 0 else 0.0
                },
                'results': self.all_results
            }, f, indent=2, ensure_ascii=False)
        self.get_logger().info(f'✅ 已保存 JSON 结果: {json_path}')
        
        # 保存 CSV
        csv_path = os.path.join(self.output_dir, 'results.csv')
        if self.all_results:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'frame_id', 'timestamp', 'success',
                    'center_u_mean', 'center_v_mean', 'center_u_mid', 'center_v_mid',
                    'roll_euler', 'pitch_euler', 'yaw_euler',
                    'roll_tilt', 'pitch_tilt', 'yaw_tilt',
                    'pitch_tilt_mod90', 'yaw_tilt_mod90',
                    'reprojection_error_mean', 'reprojection_error_max',
                    'has_tilt', 'roll_offset', 'pitch_offset', 'pitch_offset_mod90', 'yaw_offset', 'yaw_offset_mod90'
                ])
                writer.writeheader()
                for r in self.all_results:
                    writer.writerow({
                        'frame_id': r['frame_id'],
                        'timestamp': r['timestamp'],
                        'success': r['success'],
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
                        'pitch_tilt_mod90': r['camera_tilt_angles_mod']['pitch_mod90'],
                        'yaw_tilt_mod90': r['camera_tilt_angles_mod']['yaw_mod90'],
                        'reprojection_error_mean': r['reprojection_error']['mean'],
                        'reprojection_error_max': r['reprojection_error']['max'],
                        'has_tilt': r['tilt_detection']['has_tilt'],
                        'roll_offset': r['tilt_detection']['roll_offset'],
                        'pitch_offset': r['tilt_detection']['pitch_offset'],
                        'pitch_offset_mod90': r['tilt_detection']['pitch_offset_mod90'],
                        'yaw_offset': r['tilt_detection']['yaw_offset'],
                        'yaw_offset_mod90': r['tilt_detection']['yaw_offset_mod90']
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
            f.write('相机倾斜检测统计报告\n')
            f.write('='*60 + '\n\n')
            
            f.write(f'总帧数: {self.frame_count}\n')
            f.write(f'成功检测: {self.success_count}\n')
            f.write(f'失败检测: {self.failure_count}\n')
            f.write(f'成功率: {self.success_count / self.frame_count * 100:.2f}%\n\n')
            
            if self.success_count > 0:
                # 统计角度（使用归一化后的角度）
                roll_tilts = [r['camera_tilt_angles']['roll'] for r in self.all_results]
                pitch_tilts = [r['camera_tilt_angles_mod']['pitch_mod90'] for r in self.all_results]
                yaw_tilts = [r['camera_tilt_angles_mod']['yaw_mod90'] for r in self.all_results]
                
                f.write('相机倾斜角度统计（假设板子水平，Pitch和Yaw已做90°归一化）:\n')
                f.write(f'  Roll (前后仰, 绕X轴):  平均={np.mean(roll_tilts):+.2f}°, 最大={np.max(np.abs(roll_tilts)):.2f}°\n')
                f.write(f'  Pitch (平面旋, 绕Z轴): 平均={np.mean(pitch_tilts):+.2f}°, 最大={np.max(np.abs(pitch_tilts)):.2f}°\n')
                f.write(f'  Yaw (左右歪, 绕Y轴):   平均={np.mean(yaw_tilts):+.2f}°, 最大={np.max(np.abs(yaw_tilts)):.2f}°\n\n')
                
                # 统计重投影误差
                errors = [r['reprojection_error']['mean'] for r in self.all_results]
                f.write('重投影误差统计:\n')
                f.write(f'  平均误差: {np.mean(errors):.4f} 像素\n')
                f.write(f'  最大误差: {np.max(errors):.4f} 像素\n')
                f.write(f'  最小误差: {np.min(errors):.4f} 像素\n')
                f.write(f'  标准差: {np.std(errors):.4f} 像素\n\n')
                
                # 统计歪斜情况
                tilted_frames = sum(1 for r in self.all_results if r['tilt_detection']['has_tilt'])
                f.write('歪斜检测:\n')
                f.write(f'  存在歪斜的帧数: {tilted_frames} ({tilted_frames/self.success_count*100:.2f}%)\n')
                f.write(f'  正常帧数: {self.success_count - tilted_frames} ({(self.success_count-tilted_frames)/self.success_count*100:.2f}%)\n')
                f.write(f'  阈值: ±0.5°\n')
        
        self.get_logger().info(f'✅ 已生成统计报告: {report_path}')


def main(args=None):
    """主函数"""
    parser = argparse.ArgumentParser(description='相机倾斜检测 ROS2 节点')
    parser.add_argument('--image-topic', type=str, default='/camera/image_raw',
                       help='图像话题名称')
    parser.add_argument('--camera-yaml', type=str, default='config/camera_info.yaml',
                       help='相机内参 YAML 文件路径')
    parser.add_argument('--rows', type=int, default=15,
                       help='圆点行数（默认15）')
    parser.add_argument('--cols', type=int, default=15,
                       help='圆点列数（默认15）')
    parser.add_argument('--spacing', type=float, default=10.0,
                       help='圆点间距（mm，默认60.0）')
    parser.add_argument('--rosbag', type=str, default=None,
                       help='rosbag 文件路径（如果提供，将从 rosbag 读取）')
    parser.add_argument('--output-dir', type=str, default='outputs/rosbag_results',
                       help='输出目录')
    parser.add_argument('--save-images', action='store_true',
                       help='保存检测结果图像')
    parser.add_argument('--no-save-results', action='store_true',
                       help='不保存结果文件（JSON/CSV）')
    parser.add_argument('--publish-results', action='store_true',
                       help='发布结果到 ROS 话题（需要自定义消息类型）')
    parser.add_argument('--skip-frames', type=int, default=1,
                       help='跳帧处理（默认1=处理所有帧，2=每隔一帧处理一次）')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='最大处理帧数（默认None=处理所有帧）')
    
    # 解析参数（支持 ROS2 参数）
    if args is None:
        cli_args, _ = parser.parse_known_args()
    else:
        cli_args, _ = parser.parse_known_args(args)
    
    # 初始化 ROS2
    rclpy.init(args=args)
    
    # 创建节点
    node = TiltCheckerNode(
        image_topic=cli_args.image_topic,
        camera_yaml_path=cli_args.camera_yaml,
        rows=cli_args.rows,
        cols=cli_args.cols,
        spacing=cli_args.spacing,
        output_dir=cli_args.output_dir,
        save_images=cli_args.save_images,
        save_results=not cli_args.no_save_results,
        publish_results=cli_args.publish_results
    )
    
    # 添加跳帧和最大帧数参数
    node.skip_frames = cli_args.skip_frames
    node.max_frames = cli_args.max_frames
    
    try:
        if cli_args.rosbag:
            # 从 rosbag 处理
            node.process_rosbag(cli_args.rosbag)
            node.save_results_to_files()
        else:
            # 订阅实时话题
            node.create_subscription(
                Image,
                cli_args.image_topic,
                node.image_callback,
                10
            )
            node.get_logger().info('等待图像消息...')
            node.get_logger().info('按 Ctrl+C 退出')
            rclpy.spin(node)
            
    except KeyboardInterrupt:
        node.get_logger().info('接收到中断信号，保存结果...')
        node.save_results_to_files()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


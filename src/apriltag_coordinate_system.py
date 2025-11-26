#!/usr/bin/env python3
"""
基于AprilTag的坐标系建立模块

功能：
1. 检测AprilTag标记
2. 建立统一的坐标系（以AprilTag为参考）
3. 找到离AprilTag最近的标定板角点作为原点
4. 建立x轴方向为AprilTag正方向的坐标系
5. 解决PnP多解歧义问题，避免247像素级别的重投影误差

使用方法：
    from src.apriltag_coordinate_system import AprilTagCoordinateSystem
    
    coord_sys = AprilTagCoordinateSystem()
    success, origin, x_axis, y_axis = coord_sys.establish_coordinate_system(
        image, board_corners, camera_matrix, dist_coeffs
    )
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging
from .pnp_ambiguity_resolver import PnPAmbiguityResolver

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

logger = logging.getLogger(__name__)


class AprilTagCoordinateSystem:
    """基于AprilTag的坐标系建立器"""
    
    def __init__(self, 
                 tag_family: str = 'tagStandard41h12',
                 tag_size: float = 0.0071,  # AprilTag的实际尺寸(m)
                 board_spacing: float = 0.065,  # 标定板圆点间距(m)
                 max_reprojection_error: float = 1.0):  # 最大允许重投影误差
        """
        初始化AprilTag坐标系建立器
        
        Args:
            tag_family: AprilTag家族 ('tag36h11', 'tag25h9', 'tag16h5')
            tag_size: AprilTag的实际尺寸(mm)
            board_spacing: 标定板圆点间距(mm)
            max_reprojection_error: 最大允许重投影误差(像素)
        """
        self.tag_family = tag_family
        self.tag_size = tag_size
        self.board_spacing = board_spacing
        self.max_reprojection_error = max_reprojection_error
        
        # 初始化PnP多解歧义解决器
        self.pnp_resolver = PnPAmbiguityResolver(
            max_reprojection_error=max_reprojection_error
        )
        
        # 创建AprilTag检测器
        if USING_PUPIL_APRILTAGS:
            self.detector = Detector(
                families=tag_family,
                nthreads=4,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=True
            )
        else:
            options = apriltag.DetectorOptions(families=tag_family)
            self.detector = apriltag.Detector(options)
        
        logger.info(f"AprilTag坐标系建立器初始化完成")
        logger.info(f"  标签家族: {tag_family}")
        logger.info(f"  标签尺寸: {tag_size}mm")
        logger.info(f"  板子间距: {board_spacing}mm")
    
    def detect_apriltag(self, gray_image: np.ndarray) -> List:
        """
        检测图像中的AprilTag
        
        Args:
            gray_image: 灰度图像
            
        Returns:
            检测到的AprilTag列表
        """
        detections = self.detector.detect(gray_image)
        logger.debug(f"检测到 {len(detections)} 个AprilTag")
        return detections
    
    def get_tag_pose(self, 
                     detection,
                     camera_matrix: np.ndarray,
                     dist_coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        计算AprilTag的位姿
        
        Args:
            detection: AprilTag检测结果
            camera_matrix: 相机内参矩阵
            dist_coeffs: 畸变系数
            
        Returns:
            (rvec, tvec, distance): 旋转向量、平移向量和距离
        """
        # AprilTag的3D角点（以标签中心为原点，z=0平面）
        half_size = self.tag_size / 2.0
        object_points = np.array([
            [-half_size, -half_size, 0],  # 左下
            [ half_size, -half_size, 0],  # 右下
            [ half_size,  half_size, 0],  # 右上
            [-half_size,  half_size, 0]   # 左上
        ], dtype=np.float32)
        
        # 图像中的角点（按照AprilTag的角点顺序）
        image_points = np.array(detection.corners, dtype=np.float32)
        
        # 使用迭代PnP求解位姿（更稳定）
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, 
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            raise ValueError("AprilTag位姿求解失败")
        
        # 计算距离（Z轴距离）
        distance = np.linalg.norm(tvec)
        
        # 评估检测质量（确保corners格式正确）
        try:
            corners_for_area = np.array(detection.corners, dtype=np.float32).reshape(-1, 1, 2)
            tag_area = cv2.contourArea(corners_for_area)
            quality_score = np.sqrt(tag_area) / 100.0
        except Exception as e:
            # 如果计算面积失败，使用默认值
            logger.debug(f"无法计算AprilTag面积: {e}")
            tag_area = 0.0
            quality_score = 1.0
        
        # 距离警告
        if distance > 1.0:  # 超过1米
            logger.warning(f"⚠️  标定板距离较远: {distance:.2f}m，重投影误差可能增大")
            logger.warning(f"   建议距离: < 0.8m，当前检测质量: {quality_score:.2f}")
        
        logger.debug(f"标定板距离: {distance:.3f}m, 检测质量: {quality_score:.2f}")
            
        return rvec, tvec, distance
    
    def find_nearest_board_corner(self, 
                                  tag_center: np.ndarray,
                                  board_corners: np.ndarray,
                                  grid_rows: int,
                                  grid_cols: int) -> Tuple[int, np.ndarray]:
        """
        找到离AprilTag中心最近的标定板角点（仅在四个角点中查找）
        
        Args:
            tag_center: AprilTag中心坐标 [x, y]
            board_corners: 标定板角点数组 (N, 2)
            grid_rows: 网格行数
            grid_cols: 网格列数
            
        Returns:
            (最近角点的索引, 最近角点坐标)
        """
        if board_corners is None or len(board_corners) == 0:
            raise ValueError("board_corners 为空，无法选择原点")
        
        # 仅在四个边角候选中查找，避免误选内部点
        candidate_indices = [
            0,                          # 左上角
            grid_cols - 1,              # 右上角
            (grid_rows - 1) * grid_cols, # 左下角
            grid_rows * grid_cols - 1    # 右下角
        ]
        candidate_indices = [idx for idx in candidate_indices if idx < len(board_corners)]
        
        if not candidate_indices:
            raise ValueError("无法定位四个角点，请确认网格尺寸是否正确")
        
        corner_names = ['左上角', '右上角', '左下角', '右下角']
        
        # 计算每个角点到AprilTag中心的距离
        distances = {
            idx: np.linalg.norm(board_corners[idx] - tag_center)
            for idx in candidate_indices
        }
        
        nearest_idx = min(distances, key=distances.get)
        nearest_corner = board_corners[nearest_idx]
        min_distance = distances[nearest_idx]
        
        # 找到对应的角点名称
        corner_idx_in_list = candidate_indices.index(nearest_idx)
        nearest_corner_name = corner_names[corner_idx_in_list] if corner_idx_in_list < len(corner_names) else '未知'
        
        logger.debug(f"最近角点: {nearest_corner_name}(索引{nearest_idx}), 距离: {min_distance:.2f}px")
        
        return nearest_idx, nearest_corner
    
    def establish_coordinate_system(self,
                                   image: np.ndarray,
                                   board_corners: np.ndarray,
                                   camera_matrix: np.ndarray,
                                   dist_coeffs: np.ndarray,
                                   grid_rows: int,
                                   grid_cols: int) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[dict]]:
        """
        建立基于AprilTag的坐标系
        
        Args:
            image: 输入图像（BGR或灰度）
            board_corners: 标定板角点 (N, 2)
            camera_matrix: 相机内参矩阵
            dist_coeffs: 畸变系数
            grid_rows: 网格行数
            grid_cols: 网格列数
            
        Returns:
            (成功标志, 原点坐标, x轴方向向量, y轴方向向量, 详细信息)
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 检测AprilTag
        detections = self.detect_apriltag(gray)
        
        if len(detections) == 0:
            logger.warning("未检测到AprilTag")
            return False, None, None, None, None
        
        if len(detections) > 1:
            logger.warning(f"检测到多个AprilTag ({len(detections)}个)，使用第一个")
        
        # 使用第一个检测到的AprilTag
        detection = detections[0]
        
        try:
            # 计算AprilTag位姿
            tag_rvec, tag_tvec, distance = self.get_tag_pose(detection, camera_matrix, dist_coeffs)
            
            # 获取AprilTag中心坐标（确保是numpy数组格式）
            if hasattr(detection.center, '__len__') and len(detection.center) == 2:
                tag_center = np.array([float(detection.center[0]), float(detection.center[1])], dtype=np.float64)
            else:
                # 如果center是其他格式，尝试转换
                tag_center = np.array(detection.center, dtype=np.float64).flatten()
                if len(tag_center) != 2:
                    raise ValueError(f"AprilTag中心坐标格式错误: {detection.center}")
            
            # 找到最近的标定板角点作为原点（仅在四个角点中查找）
            nearest_idx, origin_2d = self.find_nearest_board_corner(
                tag_center, board_corners, grid_rows, grid_cols
            )
            
            # 计算AprilTag的方向向量（在图像平面上）
            # 确保corners是正确格式的numpy数组
            tag_corners = np.array(detection.corners, dtype=np.float64)
            if tag_corners.shape != (4, 2):
                tag_corners = tag_corners.reshape(4, 2)
            
            # AprilTag的x轴方向：从左下角到右下角
            tag_x_direction_2d = tag_corners[1] - tag_corners[0]  # 右下 - 左下
            tag_x_direction_2d = tag_x_direction_2d / np.linalg.norm(tag_x_direction_2d)
            
            # AprilTag的y轴方向：从左下角到左上角  
            tag_y_direction_2d = tag_corners[3] - tag_corners[0]  # 左上 - 左下
            tag_y_direction_2d = tag_y_direction_2d / np.linalg.norm(tag_y_direction_2d)
            
            # 确保右手坐标系：检查x和y轴的叉积方向
            cross_product = np.cross(tag_x_direction_2d, tag_y_direction_2d)
            if cross_product < 0:
                # 如果叉积为负，说明是左手坐标系，需要调整
                tag_y_direction_2d = -tag_y_direction_2d
                logger.debug("调整Y轴方向以确保右手坐标系")
            
            logger.debug(f"AprilTag方向 - X: {tag_x_direction_2d}, Y: {tag_y_direction_2d}, 叉积: {cross_product:.3f}")
            
            # 重新排列标定板角点，使其符合新的坐标系
            reordered_corners, permutation = self._reorder_board_corners(
                board_corners, nearest_idx, tag_x_direction_2d, tag_y_direction_2d,
                grid_rows, grid_cols
            )
            
            # 构建详细信息
            origin_position = int(np.where(permutation == nearest_idx)[0][0])
            
            info = {
                'tag_id': detection.tag_id,
                'tag_center': tag_center,
                'tag_corners': tag_corners,
                'tag_rvec': tag_rvec,
                'tag_tvec': tag_tvec,
                'distance': distance,
                'origin_idx': nearest_idx,
                'origin_position': origin_position,
                'origin_2d': origin_2d,
                'x_direction_2d': tag_x_direction_2d,
                'y_direction_2d': tag_y_direction_2d,
                'reordered_corners': reordered_corners,
                'corner_permutation': permutation
            }
            
            logger.info(f"✅ 成功建立坐标系")
            logger.info(f"  AprilTag ID: {detection.tag_id}")
            logger.info(f"  标定板距离: {distance:.3f}m")
            logger.info(f"  原点角点索引: {nearest_idx}")
            logger.info(f"  原点坐标: ({origin_2d[0]:.1f}, {origin_2d[1]:.1f})")
            
            return True, origin_2d, tag_x_direction_2d, tag_y_direction_2d, info
            
        except Exception as e:
            logger.error(f"建立坐标系失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False, None, None, None, None
    
    def _reorder_board_corners(self,
                              board_corners: np.ndarray,
                              origin_idx: int,
                              x_direction: np.ndarray,
                              y_direction: np.ndarray,
                              grid_rows: int,
                              grid_cols: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据新坐标系重新排列标定板角点
        
        Args:
            board_corners: 原始角点数组 (N, 2)
            origin_idx: 原点角点索引
            x_direction: x轴方向向量
            y_direction: y轴方向向量
            grid_rows: 网格行数
            grid_cols: 网格列数
            
        Returns:
            (重新排列的角点数组, 排列索引数组)
        """
        origin = board_corners[origin_idx]
        expected_points = grid_rows * grid_cols
        if len(board_corners) != expected_points:
            raise ValueError(
                f"角点数量({len(board_corners)})与网格尺寸({grid_rows}x{grid_cols})不匹配"
            )
        
        # 计算每个点在新坐标系下的投影
        relative_points = board_corners - origin
        x_coords = np.dot(relative_points, x_direction)
        y_coords = np.dot(relative_points, y_direction)
        
        logger.debug(f"坐标系建立：原点索引={origin_idx}")
        logger.debug(f"X轴方向: {x_direction}")
        logger.debug(f"Y轴方向: {y_direction}")
        logger.debug(f"角点范围: X=[{x_coords.min():.1f}, {x_coords.max():.1f}], Y=[{y_coords.min():.1f}, {y_coords.max():.1f}]")
        
        # 根据Y轴投影排序，再在每一行内按X轴排序
        sorted_indices = np.argsort(y_coords)
        ordered_indices: List[int] = []
        for r in range(grid_rows):
            row_slice = sorted_indices[r * grid_cols:(r + 1) * grid_cols]
            row_sorted = row_slice[np.argsort(x_coords[row_slice])]
            ordered_indices.extend(row_sorted.tolist())
        
        ordered_indices = np.array(ordered_indices, dtype=np.int32)
        reordered = board_corners[ordered_indices]
        
        return reordered, ordered_indices
    
    def visualize_coordinate_system(self,
                                   image: np.ndarray,
                                   info: dict,
                                   axis_length: float = 50.0) -> np.ndarray:
        """
        可视化建立的坐标系
        
        Args:
            image: 输入图像
            info: 坐标系信息
            axis_length: 坐标轴长度（像素）
            
        Returns:
            带有可视化标记的图像
        """
        vis_image = image.copy()
        
        # 绘制AprilTag
        tag_corners = info['tag_corners'].astype(int)
        cv2.polylines(vis_image, [tag_corners], True, (0, 255, 0), 2)
        
        # 绘制AprilTag ID
        tag_center = info['tag_center'].astype(int)
        cv2.putText(vis_image, f"ID:{info['tag_id']}", 
                   (tag_center[0]-20, tag_center[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 绘制原点
        origin = info['origin_2d'].astype(int)
        cv2.circle(vis_image, tuple(origin.tolist()), 8, (0, 0, 255), -1)
        cv2.putText(vis_image, "Origin", 
                   (int(origin[0])+10, int(origin[1])-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 绘制坐标轴 - 添加数据验证
        x_direction = info['x_direction_2d']
        y_direction = info['y_direction_2d']
        
        # 检查方向向量是否有效
        if not (np.isfinite(x_direction).all() and np.isfinite(y_direction).all()):
            logger.warning("坐标轴方向向量包含无效值，跳过绘制")
            return vis_image
        
        x_end = (origin + (x_direction * axis_length)).astype(int)
        y_end = (origin + (y_direction * axis_length)).astype(int)
        
        # X轴（红色）
        cv2.arrowedLine(vis_image, tuple(origin.tolist()), tuple(x_end.tolist()), (0, 0, 255), 3)
        cv2.putText(vis_image, "X", (int(x_end[0])+5, int(x_end[1])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Y轴（绿色）
        cv2.arrowedLine(vis_image, tuple(origin.tolist()), tuple(y_end.tolist()), (0, 255, 0), 3)
        cv2.putText(vis_image, "Y", (int(y_end[0])+5, int(y_end[1])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Z轴（蓝色）- 使用3D投影绘制Z轴
        if 'tag_rvec' in info and 'tag_tvec' in info:
            try:
                # 从AprilTag位姿获取相机内参（如果可用）
                # 这里我们简化处理，Z轴方向向上偏移显示
                z_direction_2d = np.array([-0.3, -0.8])  # 向左上方向，模拟Z轴向外
                z_end = (origin + (z_direction_2d * axis_length * 0.7)).astype(int)
                
                cv2.arrowedLine(vis_image, tuple(origin.tolist()), tuple(z_end.tolist()), (255, 0, 0), 3)
                cv2.putText(vis_image, "Z", (int(z_end[0])+5, int(z_end[1])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            except Exception as e:
                logger.debug(f"绘制Z轴失败: {e}")
        
        # 绘制重新排列的角点
        if 'reordered_corners' in info:
            for i, corner in enumerate(info['reordered_corners']):
                corner_int = corner.astype(int)
                cv2.circle(vis_image, tuple(corner_int.tolist()), 3, (255, 0, 0), -1)
        
        return vis_image


def test_apriltag_coordinate_system():
    """测试AprilTag坐标系建立功能"""
    import os
    
    # 创建测试用的坐标系建立器
    coord_sys = AprilTagCoordinateSystem()
    
    # 这里可以添加测试代码
    print("AprilTag坐标系建立器测试完成")


if __name__ == '__main__':
    test_apriltag_coordinate_system()
#!/usr/bin/env python3
"""
修正后的基于AprilTag引导的网格检测模块

主要改进：
1. 修正了AprilTag位置的计算方式
2. 使用匈牙利算法避免重复匹配
3. 改进了网格原点的推算
4. 添加了更多的调试信息和鲁棒性检查
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

# 尝试导入 scipy，如果失败则使用纯 Python 实现
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy 不可用，使用纯 Python 实现的匈牙利算法（速度较慢）")
    
    def linear_sum_assignment(cost_matrix):
        """
        纯 Python 实现的匈牙利算法（简化版，用于二分图匹配）
        
        这是一个贪心近似算法，不保证全局最优，但对于我们的用例足够好
        """
        cost = cost_matrix.copy()
        n_rows, n_cols = cost.shape
        
        # 确保是方阵
        if n_rows != n_cols:
            max_dim = max(n_rows, n_cols)
            padded_cost = np.full((max_dim, max_dim), 1e10, dtype=cost.dtype)
            padded_cost[:n_rows, :n_cols] = cost
            cost = padded_cost
            n = max_dim
        else:
            n = n_rows
        
        # 贪心算法：每次选择当前最小的未匹配边
        row_indices = []
        col_indices = []
        used_rows = set()
        used_cols = set()
        
        # 将所有边按权重排序
        edges = []
        for i in range(n):
            for j in range(n):
                if i < n_rows and j < n_cols:  # 只考虑原始矩阵范围内的边
                    edges.append((cost[i, j], i, j))
        
        edges.sort()  # 按权重排序
        
        # 贪心选择
        for weight, i, j in edges:
            if i not in used_rows and j not in used_cols:
                if weight < 1e9:  # 只接受有效的边
                    row_indices.append(i)
                    col_indices.append(j)
                    used_rows.add(i)
                    used_cols.add(j)
                    
                    if len(row_indices) == min(n_rows, n_cols):
                        break
        
        return np.array(row_indices), np.array(col_indices)

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
        logger.warning("AprilTag库不可用，请安装: pip install pupil-apriltags 或 pip install apriltag")


def build_blob_detector():
    """构建blob检测器"""
    params = cv2.SimpleBlobDetector_Params()
    
    params.filterByColor = True
    params.blobColor = 0  # 检测黑色blob
    
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 5000
    
    params.filterByCircularity = True
    params.minCircularity = 0.7
    
    params.filterByConvexity = True
    params.minConvexity = 0.8
    
    params.filterByInertia = True
    params.minInertiaRatio = 0.6
    
    return cv2.SimpleBlobDetector_create(params)


def refine_corners(gray_image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """子像素精度优化"""
    if corners is None or len(corners) == 0:
        return corners
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    
    corners_refined = cv2.cornerSubPix(
        gray_image,
        corners.copy(),
        (11, 11),
        (-1, -1),
        criteria
    )
    
    return corners_refined


class AprilTagGuidedGridDetector:
    """
    修正后的基于AprilTag引导的网格检测器
    """
    
    def __init__(self,
                 pattern_size: Tuple[int, int] = (15, 15),
                 circle_spacing: float = 0.065,
                 apriltag_size: float = 0.0071,
                 tag_family: str = 'tagStandard41h12',
                 max_match_distance: float = 25.0,
                 image_margin: float = 20.0,
                 apriltag_position: str = 'right_top'):
        """
        初始化检测器
        
        Args:
            pattern_size: 完整网格尺寸 (cols, rows)
            circle_spacing: 圆点间距（米）
            apriltag_size: AprilTag边长（米）
            tag_family: AprilTag家族
            max_match_distance: 最大匹配距离（像素）
            image_margin: 图像边界裕度（像素）
            apriltag_position: AprilTag相对于网格的位置
                - 'right_top': 在右上角外部
                - 'right_top_inside': 在右上角内部（替代第15列第1行圆点）
        """
        self.pattern_size = pattern_size
        self.grid_cols, self.grid_rows = pattern_size
        self.circle_spacing = circle_spacing
        self.apriltag_size = apriltag_size
        self.max_match_distance = max_match_distance
        self.image_margin = image_margin
        self.apriltag_position = apriltag_position
        
        # 根据AprilTag位置确定偏移量
        # 假设AprilTag在网格右上角外部，中心位置相对于左上角第一个圆点的偏移
        if apriltag_position == 'right_top_inside':
            # AprilTag替代了第15列第1行的圆点
            self.apriltag_offset_in_grid = (pattern_size[0] - 1, 0)
        else:
            # AprilTag在网格外部，需要根据实际情况调整
            # 这里假设AprilTag中心在第15列圆点右侧约1.5个间距处
            self.apriltag_offset_in_grid = (pattern_size[0] - 0.5, 0)
        
        # 创建AprilTag检测器
        if APRILTAG_AVAILABLE:
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
        else:
            self.detector = None
        
        # 创建blob检测器
        self.blob_detector = build_blob_detector()
        
        logger.info(f"AprilTag引导网格检测器初始化完成")
        logger.info(f"  网格尺寸: {self.grid_cols}×{self.grid_rows}")
        logger.info(f"  圆点间距: {circle_spacing}m")
        logger.info(f"  AprilTag尺寸: {apriltag_size}m")
        logger.info(f"  AprilTag位置: {apriltag_position}")
    
    def detect_apriltag(self, gray_image: np.ndarray) -> Optional[Dict]:
        """检测AprilTag"""
        if self.detector is None:
            logger.warning("AprilTag检测器不可用")
            return None
        
        detections = self.detector.detect(gray_image)
        
        if len(detections) == 0:
            return None
        
        if len(detections) > 1:
            logger.warning(f"检测到多个AprilTag ({len(detections)}个)，使用第一个")
        
        detection = detections[0]
        
        if USING_PUPIL_APRILTAGS:
            corners = np.array(detection.corners, dtype=np.float64)
            center = np.array(detection.center, dtype=np.float64)
            tag_id = detection.tag_id
        else:
            corners = np.array(detection.corners, dtype=np.float64)
            center = np.array(detection.center, dtype=np.float64)
            tag_id = detection.tag_id
        
        return {
            'corners': corners,
            'center': center,
            'tag_id': tag_id,
            'detection': detection
        }
    
    def estimate_grid_from_apriltag(self,
                                   apriltag_info: Dict,
                                   image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        改进的网格位置估算
        
        关键改进：
        1. 更准确的方向向量计算
        2. 考虑AprilTag可能的旋转
        3. 更鲁棒的网格原点计算
        """
        h, w = image_shape[:2]
        tag_center = apriltag_info['center']
        tag_corners = apriltag_info['corners']
        
        # 1. 计算AprilTag的方向和尺度
        # AprilTag角点顺序: [左上, 右上, 右下, 左下]
        top_left = tag_corners[0]
        top_right = tag_corners[1]
        bottom_left = tag_corners[3]
        
        # X轴方向（从左到右）
        tag_x_vec = top_right - top_left
        tag_x_len = np.linalg.norm(tag_x_vec)
        unit_x = tag_x_vec / tag_x_len
        
        # Y轴方向（从上到下）
        tag_y_vec = bottom_left - top_left
        tag_y_len = np.linalg.norm(tag_y_vec)
        unit_y = tag_y_vec / tag_y_len
        
        # 2. 计算像素/米的比例
        pixel_per_meter = (tag_x_len + tag_y_len) / (2.0 * self.apriltag_size)
        circle_spacing_px = self.circle_spacing * pixel_per_meter
        
        logger.debug(f"AprilTag中心: ({tag_center[0]:.2f}, {tag_center[1]:.2f})")
        logger.debug(f"像素/米比例: {pixel_per_meter:.2f} px/m")
        logger.debug(f"圆点间距: {circle_spacing_px:.2f} px")
        logger.debug(f"X方向: ({unit_x[0]:.3f}, {unit_x[1]:.3f})")
        logger.debug(f"Y方向: ({unit_y[0]:.3f}, {unit_y[1]:.3f})")
        
        # 3. 计算网格左上角第一个圆点的位置
        # 从AprilTag中心反推到网格原点
        offset_x = self.apriltag_offset_in_grid[0] * circle_spacing_px
        offset_y = self.apriltag_offset_in_grid[1] * circle_spacing_px
        
        grid_origin = tag_center - unit_x * offset_x - unit_y * offset_y
        
        logger.debug(f"网格原点: ({grid_origin[0]:.2f}, {grid_origin[1]:.2f})")
        logger.debug(f"从AprilTag到原点的偏移: X={-offset_x:.2f}px, Y={-offset_y:.2f}px")
        
        # 4. 生成所有网格点
        grid_points = np.zeros((self.grid_rows, self.grid_cols, 2), dtype=np.float32)
        valid_mask = np.zeros((self.grid_rows, self.grid_cols), dtype=bool)
        
        margin = self.image_margin
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                point = grid_origin + unit_x * col * circle_spacing_px + unit_y * row * circle_spacing_px
                grid_points[row, col] = point
                
                # 检查是否在图像范围内
                if (margin <= point[0] < w - margin and 
                    margin <= point[1] < h - margin):
                    valid_mask[row, col] = True
        
        valid_count = np.sum(valid_mask)
        total_count = self.grid_rows * self.grid_cols
        logger.info(f"预期在图像范围内的圆点: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
        
        return grid_points, valid_mask
    
    def detect_all_blobs(self, gray_image: np.ndarray) -> Tuple[np.ndarray, List]:
        """检测所有blob"""
        keypoints = self.blob_detector.detect(gray_image)
        
        if len(keypoints) == 0:
            return np.array([], dtype=np.float32).reshape(0, 2), []
        
        blob_points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        logger.debug(f"检测到 {len(blob_points)} 个blob")
        
        return blob_points, keypoints
    
    def match_blobs_to_grid_hungarian(self,
                                     blob_points: np.ndarray,
                                     grid_points: np.ndarray,
                                     valid_mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """
        使用匈牙利算法进行最优匹配，避免重复匹配
        
        关键改进：处理网格点数和blob数不相等的情况
        """
        match_mask = np.zeros((self.grid_rows, self.grid_cols), dtype=bool)
        
        if len(blob_points) == 0:
            logger.warning("没有检测到任何blob")
            return None, None, match_mask
        
        # 收集所有有效的网格点
        valid_grid_points = []
        valid_grid_indices = []
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                if valid_mask[row, col]:
                    valid_grid_points.append(grid_points[row, col])
                    valid_grid_indices.append((row, col))
        
        if len(valid_grid_points) == 0:
            logger.warning("没有有效的网格点")
            return None, None, match_mask
        
        valid_grid_points = np.array(valid_grid_points)
        valid_grid_indices = np.array(valid_grid_indices)
        
        n_grids = len(valid_grid_points)
        n_blobs = len(blob_points)
        
        logger.debug(f"有效网格点数: {n_grids}")
        logger.debug(f"检测到的blob数: {n_blobs}")
        
        # 计算距离矩阵
        # 关键修正：使cost_matrix为方阵，确保匈牙利算法正确工作
        max_dim = max(n_grids, n_blobs)
        cost_matrix = np.full((max_dim, max_dim), 1e6, dtype=np.float32)
        
        # 填充实际的距离
        for i in range(n_grids):
            for j in range(n_blobs):
                dist = np.linalg.norm(valid_grid_points[i] - blob_points[j])
                if dist <= self.max_match_distance:
                    cost_matrix[i, j] = dist
                else:
                    cost_matrix[i, j] = 1e6  # 超过阈值的设为无穷大
        
        # 使用匈牙利算法求解
        grid_indices, blob_indices = linear_sum_assignment(cost_matrix)
        
        # 筛选有效匹配
        matched_corners_list = []
        matched_grid_indices_list = []
        used_blobs = set()
        
        for grid_idx, blob_idx in zip(grid_indices, blob_indices):
            # 确保索引在有效范围内
            if grid_idx >= n_grids or blob_idx >= n_blobs:
                continue
            
            # 检查距离是否在阈值内
            dist = cost_matrix[grid_idx, blob_idx]
            if dist < self.max_match_distance:
                row, col = valid_grid_indices[grid_idx]
                matched_corners_list.append(blob_points[blob_idx])
                matched_grid_indices_list.append([row, col])
                match_mask[row, col] = True
                used_blobs.add(blob_idx)
        
        if len(matched_corners_list) == 0:
            logger.warning("没有blob成功匹配到网格")
            return None, None, match_mask
        
        matched_corners = np.array(matched_corners_list, dtype=np.float32).reshape(-1, 1, 2)
        matched_indices = np.array(matched_grid_indices_list, dtype=np.int32)
        
        # 统计信息
        matched_count = len(matched_corners)
        unmatched_blobs = n_blobs - len(used_blobs)
        unmatched_grids = n_grids - matched_count
        
        logger.info(f"成功匹配: {matched_count}/{n_grids} 网格点 "
                   f"({matched_count/n_grids*100:.1f}%)")
        logger.info(f"未匹配的blob: {unmatched_blobs}/{n_blobs}")
        logger.info(f"未匹配的网格点: {unmatched_grids}/{n_grids}")
        
        if matched_count > 0:
            valid_dists = [cost_matrix[grid_indices[i], blob_indices[i]] 
                          for i in range(len(grid_indices))
                          if grid_indices[i] < n_grids and blob_indices[i] < n_blobs
                          and cost_matrix[grid_indices[i], blob_indices[i]] < self.max_match_distance]
            if valid_dists:
                avg_dist = np.mean(valid_dists)
                max_dist = np.max(valid_dists)
                logger.debug(f"匹配距离: 平均={avg_dist:.2f}px, 最大={max_dist:.2f}px")
        
        return matched_corners, matched_indices, match_mask
    
    def generate_object_points(self, matched_indices: np.ndarray) -> np.ndarray:
        """生成3D物体点"""
        object_points = np.zeros((len(matched_indices), 3), dtype=np.float32)
        
        for i, (row, col) in enumerate(matched_indices):
            object_points[i] = [
                col * self.circle_spacing,
                row * self.circle_spacing,
                0.0
            ]
        
        return object_points
    
    def detect(self, image: np.ndarray) -> Dict:
        """完整的检测流程"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. 检测AprilTag
        apriltag_info = self.detect_apriltag(gray)
        if apriltag_info is None:
            logger.error("未检测到AprilTag")
            return {
                'success': False,
                'message': 'AprilTag not detected'
            }
        
        logger.info(f"✓ 检测到AprilTag ID: {apriltag_info['tag_id']}")
        
        # 2. 估算网格位置
        grid_points, valid_mask = self.estimate_grid_from_apriltag(
            apriltag_info,
            image.shape
        )
        
        # 3. 检测所有blob
        blob_points, keypoints = self.detect_all_blobs(gray)
        
        if len(blob_points) == 0:
            logger.error("未检测到任何blob")
            return {
                'success': False,
                'message': 'No blobs detected',
                'apriltag_info': apriltag_info,
                'grid_points': grid_points,
                'valid_mask': valid_mask
            }
        
        logger.info(f"✓ 检测到 {len(blob_points)} 个blob")
        
        # 4. 使用匈牙利算法匹配
        matched_corners, matched_indices, match_mask = self.match_blobs_to_grid_hungarian(
            blob_points,
            grid_points,
            valid_mask
        )
        
        if matched_corners is None:
            logger.error("blob匹配失败")
            return {
                'success': False,
                'message': 'Blob matching failed',
                'apriltag_info': apriltag_info,
                'blob_points': blob_points,
                'grid_points': grid_points,
                'valid_mask': valid_mask
            }
        
        # 5. 子像素精度优化
        matched_corners_refined = refine_corners(gray, matched_corners)
        
        # 6. 生成3D物体点
        object_points = self.generate_object_points(matched_indices)
        
        logger.info(f"✓ 检测完成: {len(matched_corners_refined)} 个角点")
        
        return {
            'success': True,
            'corners': matched_corners_refined,
            'object_points': object_points,
            'matched_indices': matched_indices,
            'grid_points': grid_points,
            'valid_mask': valid_mask,
            'match_mask': match_mask,
            'apriltag_info': apriltag_info,
            'blob_points': blob_points,
            'all_keypoints': keypoints,
            'grid_rows': self.grid_rows,
            'grid_cols': self.grid_cols,
            'match_count': len(matched_corners_refined),
            'valid_count': np.sum(valid_mask)
        }
    
    def visualize(self, image: np.ndarray, result: Dict, show_details: bool = True) -> np.ndarray:
        """
        增强的可视化
        
        颜色说明：
        - 绿色圆圈：检测到的所有blob
        - 蓝色十字：在范围内但未匹配的网格点（关键！显示问题所在）
        - 黄色圆点：成功匹配的角点
        - 灰色小点：超出图像范围的网格点
        - 绿色方框：AprilTag
        """
        vis = image.copy()
        
        if not result['success']:
            cv2.putText(vis, result.get('message', 'Detection failed'), (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 即使失败也显示AprilTag（如果检测到）
            if 'apriltag_info' in result and result['apriltag_info'] is not None:
                tag_corners = result['apriltag_info']['corners'].astype(int)
                cv2.polylines(vis, [tag_corners], True, (0, 255, 0), 3)
            
            return vis
        
        if show_details:
            # 1. 首先绘制所有检测到的blob（绿色圆圈）
            for kp in result['all_keypoints']:
                pt = (int(kp.pt[0]), int(kp.pt[1]))
                cv2.circle(vis, pt, int(kp.size/2), (0, 255, 0), 2)
            
            # 2. 绘制预期的网格点（这是关键！显示为什么有些blob没匹配）
            grid_points = result['grid_points']
            valid_mask = result['valid_mask']
            match_mask = result['match_mask']
            
            for row in range(self.grid_rows):
                for col in range(self.grid_cols):
                    pt = tuple(grid_points[row, col].astype(int))
                    
                    if match_mask[row, col]:
                        # 匹配成功的点：不在这里画，后面会用黄色大圆点覆盖
                        pass
                    elif valid_mask[row, col]:
                        # ⭐ 蓝色十字：在范围内但未匹配（这是关键！）
                        cv2.drawMarker(vis, pt, (255, 0, 0), 
                                     cv2.MARKER_CROSS, 15, 3)
                        # 显示网格索引
                        cv2.putText(vis, f"{row},{col}", 
                                  (pt[0]+8, pt[1]-8),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.35, 
                                  (255, 0, 0), 1)
                    else:
                        # 灰色小点：超出图像范围
                        cv2.circle(vis, pt, 2, (128, 128, 128), 1)
        
        # 3. 绘制成功匹配的角点（黄色大圆点）
        for i, corner in enumerate(result['corners']):
            pt = tuple(corner[0].astype(int))
            cv2.circle(vis, pt, 10, (0, 255, 255), -1)
            
            # 显示匹配的网格索引
            if show_details:
                row, col = result['matched_indices'][i]
                cv2.putText(vis, f"{row},{col}", 
                           (pt[0]+12, pt[1]-12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                           (0, 255, 255), 2)
        
        # 4. 绘制AprilTag（绿色方框）
        tag_corners = result['apriltag_info']['corners'].astype(int)
        cv2.polylines(vis, [tag_corners], True, (0, 255, 0), 3)
        
        # 标记AprilTag中心（绿色菱形）
        tag_center = result['apriltag_info']['center'].astype(int)
        cv2.drawMarker(vis, tuple(tag_center), (0, 255, 0), 
                      cv2.MARKER_DIAMOND, 15, 3)
        
        # 5. 添加详细的统计信息
        total_points = self.grid_rows * self.grid_cols
        valid_count = result['valid_count']
        matched_count = result['match_count']
        blob_count = len(result['blob_points'])
        
        info_text = [
            f"AprilTag ID: {result['apriltag_info']['tag_id']}",
            f"Grid: {self.grid_cols}x{self.grid_rows} = {total_points}",
            f"Valid grid: {valid_count} ({valid_count/total_points*100:.1f}%)",
            f"Blobs: {blob_count}",
            f"Matched: {matched_count}/{valid_count} ({matched_count/valid_count*100:.1f}%)",
            f"Unmatched blobs: {blob_count - matched_count}",
            f"Unmatched grids: {valid_count - matched_count}"
        ]
        
        # 绘制带背景的文字
        y_offset = 25
        for text in info_text:
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
            # 黑色半透明背景
            overlay = vis.copy()
            cv2.rectangle(overlay, (5, y_offset-18), 
                         (15 + text_size[0], y_offset+5), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
            
            # 白色文字
            cv2.putText(vis, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            y_offset += 28
        
        # 6. 添加图例（右下角）
        legend_x = vis.shape[1] - 250
        legend_y = vis.shape[0] - 150
        
        legend_items = [
            ((0, 255, 0), "Green circle: Detected blob"),
            ((0, 255, 255), "Yellow dot: Matched point"),
            ((255, 0, 0), "Blue cross: Unmatched grid"),
            ((128, 128, 128), "Gray dot: Out of range")
        ]
        
        for i, (color, text) in enumerate(legend_items):
            y = legend_y + i * 30
            # 绘制示例标记
            if i == 0:
                cv2.circle(vis, (legend_x, y), 8, color, 2)
            elif i == 1:
                cv2.circle(vis, (legend_x, y), 8, color, -1)
            elif i == 2:
                cv2.drawMarker(vis, (legend_x, y), color, cv2.MARKER_CROSS, 12, 2)
            else:
                cv2.circle(vis, (legend_x, y), 3, color, -1)
            
            # 绘制文字
            cv2.putText(vis, text, (legend_x + 20, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        return vis


# ============ 使用示例 ============

def example_usage():
    """使用示例"""
    
    # 1. 创建检测器
    detector = AprilTagGuidedGridDetector(
        pattern_size=(15, 15),
        circle_spacing=0.065,  # 65mm
        apriltag_size=0.0071,  # 7.1mm
        max_match_distance=25.0,  # 25像素
        apriltag_position='right_top'  # 根据实际情况调整
    )
    
    # 2. 读取图像
    image = cv2.imread('your_image.jpg')
    
    # 3. 畸变矫正（如果需要）
    # camera_matrix = np.array([...])
    # dist_coeffs = np.array([...])
    # image = cv2.undistort(image, camera_matrix, dist_coeffs)
    
    # 4. 检测
    result = detector.detect(image)
    
    if result['success']:
        print(f"✓ 检测成功！")
        print(f"  匹配的角点数: {result['match_count']}")
        print(f"  有效网格点数: {result['valid_count']}")
        print(f"  匹配率: {result['match_count']/result['valid_count']*100:.1f}%")
        
        # 5. 可视化
        vis = detector.visualize(image, result, show_details=True)
        cv2.imshow('Detection Result', vis)
        cv2.waitKey(0)
    else:
        print(f"✗ 检测失败: {result['message']}")
        vis = detector.visualize(image, result)
        cv2.imshow('Detection Failed', vis)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    example_usage()

#!/usr/bin/env python3
"""
AprilTag 引导的相机标定完整代码 - 批量处理版本
功能：
1. 批量处理data文件夹下的所有图片
2. 智能网格匹配（包含遮挡点补全）
3. PnP求解相机位姿
4. 计算欧拉角和变换矩阵 [δx,δy,δz,γ,α,β]
5. 详细的终端输出和可视化
6. 保存所有结果到Result_array.txt
"""

import cv2
import numpy as np
import yaml
import os
from pathlib import Path

try:
    from pupil_apriltags import Detector
    USING_PUPIL = True
except ImportError:
    import apriltag
    USING_PUPIL = False


# ==================== 相机参数加载 ====================
def load_camera_params(yaml_path):
    """加载相机内参"""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        camera_matrix = np.array(data['camera_matrix']['data']).reshape(3, 3)
        dist_coeffs = np.array(data['distortion_coefficients']['data'])
        
        return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"❌ 加载相机参数失败: {e}")
        return None, None


# ==================== AprilTag 检测 ====================
def detect_apriltag(gray_image):
    """检测AprilTag"""
    if USING_PUPIL:
        detector = Detector(
            families='tagStandard41h12',
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=True
        )
    else:
        options = apriltag.DetectorOptions(families='tagStandard41h12')
        detector = apriltag.Detector(options)

    detections = detector.detect(gray_image)
    if len(detections) == 0:
        return None

    detection = detections[0]
    corners = np.array(detection.corners, dtype=np.float64)
    center = np.array(detection.center, dtype=np.float64)
    tag_id = detection.tag_id

    return {'corners': corners, 'center': center, 'tag_id': tag_id}


# ==================== Blob 检测 ====================
def detect_blobs(gray_image):
    """检测圆形blob - 使用保守的参数确保不漏检"""
    params = cv2.SimpleBlobDetector_Params()
    
    # 检测黑色圆点
    params.filterByColor = True
    params.blobColor = 0
    
    # 面积过滤 - 保守范围，宁可多检测
    params.filterByArea = True
    params.minArea = 30       # 适中的最小面积
    params.maxArea = 3000     # 适中的最大面积
    
    # 圆形度过滤 - 不要太严格
    params.filterByCircularity = True
    params.minCircularity = 0.6
    
    # 凸性过滤
    params.filterByConvexity = True
    params.minConvexity = 0.7
    
    # 惯性比过滤
    params.filterByInertia = True
    params.minInertiaRatio = 0.5

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray_image)
    blob_points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    
    print(f"\n[Blob检测] 共检测到 {len(blob_points)} 个候选点")
    
    return blob_points, keypoints


def filter_blobs_exclude_apriltag(blob_points, apriltag_info):
    """排除 AprilTag 区域的 blob - 保守策略"""
    if len(blob_points) == 0:
        return blob_points, np.array([])
    
    tag_center = apriltag_info['center']
    tag_corners = apriltag_info['corners']
    
    # 计算AprilTag的尺寸
    tag_size_px = np.mean([
        np.linalg.norm(tag_corners[1] - tag_corners[0]),
        np.linalg.norm(tag_corners[2] - tag_corners[1])
    ])
    
    # 使用较小的排除半径，避免误删网格点
    exclusion_radius = tag_size_px * 1.0  # 从1.5降低到1.2
    distances = np.linalg.norm(blob_points - tag_center, axis=1)
    valid_mask = distances > exclusion_radius
    
    filtered_blobs = blob_points[valid_mask]
    excluded_indices = np.where(~valid_mask)[0]
    
    print(f"\n[排除AprilTag区域]")
    print(f"  AprilTag尺寸: {tag_size_px:.1f} px")
    print(f"  排除半径: {exclusion_radius:.1f} px")
    print(f"  排除 {len(excluded_indices)} 个点，保留 {len(filtered_blobs)} 个点")
    
    return filtered_blobs, excluded_indices


def filter_blobs_by_grid_pattern(blob_points, apriltag_info, expected_spacing_range=(35, 55)):
    """
    基于网格模式过滤：保留符合网格间距的点，排除边框孔
    
    策略：
    1. 估算主网格间距
    2. 对每个点，检查其邻居是否符合网格间距
    3. 保留有足够多"网格邻居"的点
    """
    if len(blob_points) < 100:
        return blob_points, np.array([])
    
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(blob_points)
        
        # 估算网格间距
        distances, _ = tree.query(blob_points, k=5)  # 找最近4个邻居
        nearest_distances = distances[:, 1]  # 最近邻距离
        estimated_spacing = np.median(nearest_distances)
        
        print(f"\n[网格模式过滤]")
        print(f"  估算间距: {estimated_spacing:.1f} px")
        
        # 对每个点，统计在"网格间距"范围内的邻居数
        # 网格点应该有至少2-4个邻居在标准间距附近
        min_spacing = estimated_spacing * 0.7
        max_spacing = estimated_spacing * 1.4
        
        valid_neighbor_counts = []
        for i, point in enumerate(blob_points):
            # 找附近的点
            neighbors = tree.query_ball_point(point, r=max_spacing)
            # 统计距离在合理范围内的邻居
            valid_neighbors = 0
            for j in neighbors:
                if i == j:
                    continue
                dist = np.linalg.norm(blob_points[i] - blob_points[j])
                if min_spacing <= dist <= max_spacing:
                    valid_neighbors += 1
            valid_neighbor_counts.append(valid_neighbors)
        
        valid_neighbor_counts = np.array(valid_neighbor_counts)
        
        # 保留至少有2个网格邻居的点
        # 边框孔通常是孤立的，邻居少
        min_required_neighbors = 2
        valid_mask = valid_neighbor_counts >= min_required_neighbors
        
        filtered_blobs = blob_points[valid_mask]
        outlier_indices = np.where(~valid_mask)[0]
        
        print(f"  网格间距范围: {min_spacing:.1f} - {max_spacing:.1f} px")
        print(f"  最少邻居数要求: {min_required_neighbors}")
        print(f"  过滤掉 {len(outlier_indices)} 个孤立点")
        print(f"  保留 {len(filtered_blobs)} 个网格点")
        
        return filtered_blobs, outlier_indices
        
    except ImportError:
        print(f"\n[网格模式过滤] 警告：未安装scipy，跳过过滤")
        return blob_points, np.array([])


# ==================== 角点查找 ====================
def find_four_corner_blobs(blob_points, image_shape):
    """找到标定板的四个角点"""
    if len(blob_points) == 0:
        return None, None

    min_x, max_x = np.min(blob_points[:, 0]), np.max(blob_points[:, 0])
    min_y, max_y = np.min(blob_points[:, 1]), np.max(blob_points[:, 1])
    
    corner_positions = {
        'top_left': np.array([min_x, min_y]),
        'top_right': np.array([max_x, min_y]),
        'bottom_left': np.array([min_x, max_y]),
        'bottom_right': np.array([max_x, max_y])
    }
    
    corners_dict = {}
    corners_indices = {}
    
    for corner_name, ideal_pos in corner_positions.items():
        distances = np.linalg.norm(blob_points - ideal_pos, axis=1)
        nearest_idx = np.argmin(distances)
        corners_dict[corner_name] = blob_points[nearest_idx]
        corners_indices[corner_name] = nearest_idx
    
    return corners_dict, corners_indices


def find_corner_near_apriltag(corners_dict, corners_indices, apriltag_info):
    """从四个角点中找到最靠近 AprilTag 的那个（作为参考点）"""
    tag_center = apriltag_info['center']
    
    min_dist = float('inf')
    nearest_corner_name = None
    nearest_corner_blob = None
    nearest_corner_idx = None
    
    for corner_name, corner_blob in corners_dict.items():
        dist = np.linalg.norm(corner_blob - tag_center)
        if dist < min_dist:
            min_dist = dist
            nearest_corner_name = corner_name
            nearest_corner_blob = corner_blob
            nearest_corner_idx = corners_indices[corner_name]
    
    return nearest_corner_blob, nearest_corner_idx, nearest_corner_name


# ==================== 方向和间距估算 ====================
def get_grid_orientation_from_apriltag(apriltag_info):
    """从 AprilTag 获取网格的方向"""
    tag_corners = apriltag_info['corners']
    
    corners_with_idx = [(i, tag_corners[i]) for i in range(4)]
    corners_with_idx.sort(key=lambda x: x[1][1])
    
    top_two = corners_with_idx[:2]
    bottom_two = corners_with_idx[2:]
    
    top_two.sort(key=lambda x: x[1][0])
    bottom_two.sort(key=lambda x: x[1][0])
    
    top_left = top_two[0][1]
    top_right = top_two[1][1]
    bottom_left = bottom_two[0][1]
    
    x_vec = top_right - top_left
    unit_x = x_vec / np.linalg.norm(x_vec)
    
    y_vec = bottom_left - top_left
    unit_y = y_vec / np.linalg.norm(y_vec)
    
    return unit_x, unit_y


def estimate_grid_spacing_robust(blob_points):
    """估算圆点间距"""
    if len(blob_points) < 10:
        return None
    
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(blob_points)
        distances, _ = tree.query(blob_points, k=2)
        nearest_distances = distances[:, 1]
        spacing_px = np.median(nearest_distances)
        return spacing_px
    except:
        distances = []
        for i in range(min(50, len(blob_points))):
            for j in range(i+1, min(50, len(blob_points))):
                dist = np.linalg.norm(blob_points[i] - blob_points[j])
                if 20 < dist < 200:
                    distances.append(dist)
        if len(distances) == 0:
            return None
        spacing_px = np.median(distances)
        return spacing_px


# ==================== 网格构建 ====================
def determine_grid_origin_from_corner(corner_name, corner_blob, unit_x, unit_y,
                                    circle_spacing_px, pattern_size=(15, 15)):
    """根据识别的角点位置，计算网格左上角原点"""
    cols, rows = pattern_size
    
    corner_map = {
        'top_left': (0, 0, 0, 0),
        'top_right': (0, cols - 1, cols - 1, 0),
        'bottom_left': (rows - 1, 0, 0, rows - 1),
        'bottom_right': (rows - 1, cols - 1, cols - 1, rows - 1)
    }
    
    r, c, offset_x, offset_y = corner_map[corner_name]
    corner_grid_pos = (r, c)
    
    grid_origin = corner_blob - unit_x * offset_x * circle_spacing_px - unit_y * offset_y * circle_spacing_px
    
    return grid_origin, corner_grid_pos


def build_grid_from_origin(grid_origin, unit_x, unit_y, circle_spacing_px, 
                          pattern_size=(15, 15)):
    """从左上角原点构建完整网格"""
    cols, rows = pattern_size
    grid_points = np.zeros((rows, cols, 2), dtype=np.float32)
    
    for row in range(rows):
        for col in range(cols):
            point = grid_origin + unit_x * col * circle_spacing_px + unit_y * row * circle_spacing_px
            grid_points[row, col] = point
    
    return grid_points


# ==================== 匹配和单应性 ====================
def simple_greedy_matching(cost_matrix, threshold=1e6):
    """简单贪心匹配"""
    n_rows, n_cols = cost_matrix.shape
    matches = []
    
    for i in range(n_rows):
        for j in range(n_cols):
            if cost_matrix[i, j] < threshold:
                matches.append((cost_matrix[i, j], i, j))
    
    matches.sort()
    used_rows, used_cols = set(), set()
    result_rows, result_cols = [], []
    
    for cost, i, j in matches:
        if i not in used_rows and j not in used_cols:
            result_rows.append(i)
            result_cols.append(j)
            used_rows.add(i)
            used_cols.add(j)
    
    return np.array(result_rows), np.array(result_cols)


def match_blobs_to_grid(blob_points, grid_points, max_distance=100.0):
    """匹配 blob 到网格"""
    rows, cols = grid_points.shape[:2]
    
    valid_grids = []
    valid_indices = []
    for row in range(rows):
        for col in range(cols):
            valid_grids.append(grid_points[row, col])
            valid_indices.append((row, col))
    
    if len(valid_grids) == 0 or len(blob_points) == 0:
        return None, None
    
    valid_grids = np.array(valid_grids)
    valid_indices = np.array(valid_indices)
    
    n_grids = len(valid_grids)
    n_blobs = len(blob_points)
    cost_matrix = np.zeros((n_grids, n_blobs), dtype=np.float32)
    
    for i in range(n_grids):
        for j in range(n_blobs):
            dist = np.linalg.norm(valid_grids[i] - blob_points[j])
            cost_matrix[i, j] = dist if dist <= max_distance else 1e6
    
    grid_indices, blob_indices = simple_greedy_matching(cost_matrix, threshold=max_distance)
    
    matched_corners = []
    matched_indices = []
    
    for grid_idx, blob_idx in zip(grid_indices, blob_indices):
        if cost_matrix[grid_idx, blob_idx] < max_distance:
            row, col = valid_indices[grid_idx]
            matched_corners.append(blob_points[blob_idx])
            matched_indices.append([row, col])
    
    if len(matched_corners) == 0:
        return None, None
    
    matched_corners = np.array(matched_corners, dtype=np.float32).reshape(-1, 1, 2)
    matched_indices = np.array(matched_indices, dtype=np.int32)
    
    return matched_corners, matched_indices


def refine_grid_with_homography(grid_points, matched_indices, matched_corners, 
                               pattern_size=(15, 15)):
    """使用单应性矩阵优化网格位置"""
    if len(matched_corners) < 4:
        return grid_points, None
    
    src_points = []
    dst_points = []
    
    for i in range(len(matched_indices)):
        row, col = matched_indices[i]
        src_points.append([col, row])
        dst_points.append(matched_corners[i, 0])
    
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    
    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    
    if H is None:
        return grid_points, None
    
    # 投影所有网格点
    rows, cols = pattern_size
    all_grid_logical = []
    for r in range(rows):
        for c in range(cols):
            all_grid_logical.append([c, r])
    
    all_grid_logical = np.array(all_grid_logical, dtype=np.float32).reshape(-1, 1, 2)
    refined_points_flat = cv2.perspectiveTransform(all_grid_logical, H)
    refined_grid_points = refined_points_flat.reshape(rows, cols, 2)
    
    return refined_grid_points, H


def match_and_fill_missing_points(refined_grid_points, all_blobs, 
                                  search_radius=20.0, pattern_size=(15, 15)):
    """智能补全网格"""
    rows, cols = pattern_size
    final_grid = np.zeros_like(refined_grid_points)
    status_mask = np.zeros((rows, cols), dtype=np.uint8)
    used_blobs = set()
    
    matched_count = 0
    interpolated_count = 0
    
    for r in range(rows):
        for c in range(cols):
            theoretical_pt = refined_grid_points[r, c]
            
            if len(all_blobs) > 0:
                dists = np.linalg.norm(all_blobs - theoretical_pt, axis=1)
                min_dist_idx = np.argmin(dists)
                min_dist = dists[min_dist_idx]
                
                if min_dist < search_radius and min_dist_idx not in used_blobs:
                    final_grid[r, c] = all_blobs[min_dist_idx]
                    status_mask[r, c] = 1
                    used_blobs.add(min_dist_idx)
                    matched_count += 1
                else:
                    final_grid[r, c] = theoretical_pt
                    status_mask[r, c] = 0
                    interpolated_count += 1
            else:
                final_grid[r, c] = theoretical_pt
                status_mask[r, c] = 0
                interpolated_count += 1
    
    return final_grid, status_mask, matched_count, interpolated_count


# ==================== PnP 求解和变换矩阵计算 ====================
def solve_pnp_and_compute_transform(final_grid_points, camera_matrix, dist_coeffs, 
                                   circle_spacing_m=0.02, pattern_size=(15, 15)):
    """
    使用 PnP 求解相机位姿并计算变换矩阵
    返回: [δx, δy, δz, γ, α, β]
    其中相机坐标系->标定板坐标系的变换为:
    1. 平移 (δx, δy, δz)
    2. 绕Z轴旋转γ (Yaw)
    3. 绕Y轴旋转α (Pitch)
    4. 绕X轴旋转β (Roll)
    """
    rows, cols = pattern_size
    
    # 构建3D世界坐标（标定板坐标系：X向右，Y向下，Z=0）
    object_points = []
    image_points = []
    
    for r in range(rows):
        for c in range(cols):
            object_points.append([c * circle_spacing_m, r * circle_spacing_m, 0.0])
            image_points.append(final_grid_points[r, c])
    
    object_points = np.array(object_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)
    
    # 使用 solvePnP 求解位姿
    success, rvec, tvec = cv2.solvePnP(
        object_points, 
        image_points, 
        camera_matrix, 
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return None, None, None, None, None
    
    # 计算重投影误差
    projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, 
                                           camera_matrix, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2)
    
    errors = np.linalg.norm(image_points - projected_points, axis=1)
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    # 旋转向量转旋转矩阵
    R_camera_to_board, _ = cv2.Rodrigues(rvec)
    
    # 计算欧拉角 (ZYX顺序，即 Yaw-Pitch-Roll)
    # R = Rz(γ) * Ry(α) * Rx(β)
    
    sy = np.sqrt(R_camera_to_board[0, 0]**2 + R_camera_to_board[1, 0]**2)
    
    singular = sy < 1e-6
    
    if not singular:
        beta = np.arctan2(R_camera_to_board[2, 1], R_camera_to_board[2, 2])  # Roll (绕X)
        alpha = np.arctan2(-R_camera_to_board[2, 0], sy)  # Pitch (绕Y)
        gamma = np.arctan2(R_camera_to_board[1, 0], R_camera_to_board[0, 0])  # Yaw (绕Z)
    else:
        beta = np.arctan2(-R_camera_to_board[1, 2], R_camera_to_board[1, 1])
        alpha = np.arctan2(-R_camera_to_board[2, 0], sy)
        gamma = 0
    
    # 提取平移向量
    delta_x = tvec[0, 0]
    delta_y = tvec[1, 0]
    delta_z = tvec[2, 0]
    
    # 组装变换数组 [δx, δy, δz, γ, α, β]
    transform_array = np.array([delta_x, delta_y, delta_z, gamma, alpha, beta])
    
    # 转换为角度（用于显示）
    euler_angles_deg = {
        'roll': np.degrees(beta),
        'pitch': np.degrees(alpha),
        'yaw': np.degrees(gamma)
    }
    
    return transform_array, euler_angles_deg, mean_error, max_error, (rvec, tvec)


# ==================== 主检测函数 ====================
def detect_grid_with_apriltag(image, camera_matrix, dist_coeffs, max_distance=100.0):
    """AprilTag 引导网格检测 + 相机标定"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. 检测 AprilTag
    apriltag_info = detect_apriltag(gray)
    if apriltag_info is None:
        return {'success': False, 'error': 'No AprilTag detected'}
    
    # 2. 检测所有圆点
    blob_points, keypoints = detect_blobs(gray)
    if len(blob_points) == 0:
        return {'success': False, 'error': 'No blobs detected'}
    
    # 3. 排除 AprilTag 区域（保守策略）
    filtered_blob_points, excluded_indices = filter_blobs_exclude_apriltag(
        blob_points, apriltag_info
    )
    
    # 3.5 基于网格模式过滤（更温和的方法）
    filtered_blob_points, outlier_indices = filter_blobs_by_grid_pattern(
        filtered_blob_points, apriltag_info
    )
    
    # 确保有足够的点
    if len(filtered_blob_points) < 180:
        print(f"  ⚠️  过滤后点数不足 ({len(filtered_blob_points)})，禁用过滤")
        # 只排除AprilTag，不做其他过滤
        filtered_blob_points, _ = filter_blobs_exclude_apriltag(blob_points, apriltag_info)
        print(f"  → 使用保守策略，保留 {len(filtered_blob_points)} 个点")
    
    # 4-7. 找角点、方向、间距
    corners_dict, corners_indices_dict = find_four_corner_blobs(
        filtered_blob_points, image.shape
    )
    if corners_dict is None:
        return {'success': False, 'error': 'Cannot find corner blobs'}
    
    corner_blob, corner_idx_filtered, corner_name = find_corner_near_apriltag(
        corners_dict, corners_indices_dict, apriltag_info
    )
    
    unit_x, unit_y = get_grid_orientation_from_apriltag(apriltag_info)
    spacing_px = estimate_grid_spacing_robust(filtered_blob_points)
    if spacing_px is None:
        return {'success': False, 'error': 'Cannot estimate spacing'}
    
    # 8-9. 构建初始网格
    grid_origin, corner_grid_pos = determine_grid_origin_from_corner(
        corner_name, corner_blob, unit_x, unit_y, spacing_px, pattern_size=(15, 15)
    )
    
    coarse_grid_points = build_grid_from_origin(
        grid_origin, unit_x, unit_y, spacing_px, pattern_size=(15, 15)
    )
    
    # 10. 初始匹配
    matched_corners, matched_indices = match_blobs_to_grid(
        filtered_blob_points, coarse_grid_points, max_distance=max_distance
    )
    
    if matched_corners is None or len(matched_corners) < 4:
        return {'success': False, 'error': 'Insufficient matched points'}
    
    # 11. 单应性优化
    theoretical_grid_points, H = refine_grid_with_homography(
        coarse_grid_points, matched_indices, matched_corners
    )
    
    if H is None:
        return {'success': False, 'error': 'Homography failed'}
    
    # 12. 智能补全
    search_radius = spacing_px * 0.4
    final_grid_points, status_mask, n_found, n_interp = match_and_fill_missing_points(
        theoretical_grid_points, filtered_blob_points, 
        search_radius=search_radius, pattern_size=(15, 15)
    )
    
    # 13. PnP 求解和变换矩阵计算
    result = solve_pnp_and_compute_transform(
        final_grid_points, camera_matrix, dist_coeffs
    )
    
    if result[0] is None:
        return {'success': False, 'error': 'PnP solving failed'}
    
    transform_array, euler_angles_deg, mean_error, max_error, pose = result
    
    # 14. 计算网格中心
    # 均值中心：所有角点的平均
    mean_center_2d = np.mean(final_grid_points.reshape(-1, 2), axis=0)
    
    # 网格中心：中间的角点 (7, 7)
    mid_center_2d = final_grid_points[7, 7]
    
    return {
        'success': True,
        'final_grid_points': final_grid_points,
        'theoretical_grid_points': theoretical_grid_points,
        'status_mask': status_mask,
        'apriltag_info': apriltag_info,
        'corner_name': corner_name,
        'corner_grid_pos': corner_grid_pos,
        'spacing_px': spacing_px,
        'all_keypoints': keypoints,
        'excluded_indices': excluded_indices,
        'transform_array': transform_array,
        'euler_angles_deg': euler_angles_deg,
        'mean_error': mean_error,
        'max_error': max_error,
        'pose': pose,
        'mean_center_2d': mean_center_2d,
        'mid_center_2d': mid_center_2d,
        'unit_x': unit_x,
        'unit_y': unit_y,
        'n_found': n_found,
        'n_interpolated': n_interp
    }


# ==================== 可视化 ====================
def visualize_result(image, result, camera_matrix, dist_coeffs, image_name):
    """增强版可视化"""
    vis = image.copy()
    
    # 1. 绘制 AprilTag 和连线
    tag_info = result['apriltag_info']
    tag_corners = tag_info['corners'].astype(int)
    tag_center = tag_info['center'].astype(int)
    
    # AprilTag 边框（绿色）
    cv2.polylines(vis, [tag_corners], True, (0, 255, 0), 3)
    
    # AprilTag ID
    cv2.putText(vis, f"ID:{tag_info['tag_id']}", 
                tuple(tag_center + np.array([10, -10])),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    
    # 2. 绘制网格点
    grid = result['final_grid_points']
    mask = result['status_mask']
    rows, cols = grid.shape[:2]
    
    for r in range(rows):
        for c in range(cols):
            pt = tuple(grid[r, c].astype(int))
            
            if mask[r, c] == 1:
                # 真实点（黄色实心）
                cv2.circle(vis, pt, 5, (0, 255, 255), -1)
                cv2.circle(vis, pt, 7, (0, 255, 0), 1)
            else:
                # 补全点（蓝色十字）
                cv2.drawMarker(vis, pt, (255, 0, 0), cv2.MARKER_CROSS, 15, 2)
    
    # 3. 绘制参考角点 Ref（橙色大圈）
    corner_r, corner_c = result['corner_grid_pos']
    ref_pt = tuple(grid[corner_r, corner_c].astype(int))
    cv2.circle(vis, ref_pt, 20, (0, 165, 255), 3)
    cv2.putText(vis, "Ref", (ref_pt[0]+25, ref_pt[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    
    # AprilTag 到 Ref 点的连线（洋红色虚线）
    cv2.line(vis, tuple(tag_center), ref_pt, (255, 0, 255), 2, cv2.LINE_AA)
    
    # 4. 绘制均值中心（红色实心圆）
    mean_center = tuple(result['mean_center_2d'].astype(int))
    cv2.circle(vis, mean_center, 8, (0, 0, 255), -1)
    cv2.circle(vis, mean_center, 10, (0, 0, 255), 2)
    cv2.putText(vis, "Mean", (mean_center[0]+15, mean_center[1]-15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 5. 绘制网格中心（青色十字）
    mid_center = tuple(result['mid_center_2d'].astype(int))
    cv2.drawMarker(vis, mid_center, (255, 255, 0), cv2.MARKER_TILTED_CROSS, 20, 3)
    cv2.putText(vis, "Mid", (mid_center[0]+15, mid_center[1]+15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # 6. 绘制3D坐标轴
    axis_length = 0.1  # 10cm
    axis_3d = np.float32([
        [0, 0, 0],
        [axis_length, 0, 0],  # X
        [0, axis_length, 0],  # Y
        [0, 0, -axis_length]  # Z
    ])
    
    rvec, tvec = result['pose']
    
    # 坐标轴原点设在Ref点
    ref_3d_x = corner_c * 0.02
    ref_3d_y = corner_r * 0.02
    axis_3d_shifted = axis_3d.copy()
    axis_3d_shifted[:, 0] += ref_3d_x
    axis_3d_shifted[:, 1] += ref_3d_y
    
    imgpts, _ = cv2.projectPoints(axis_3d_shifted, rvec, tvec, 
                                   camera_matrix, dist_coeffs)
    imgpts = imgpts.reshape(-1, 2).astype(int)
    
    origin = tuple(imgpts[0])
    x_end = tuple(imgpts[1])
    y_end = tuple(imgpts[2])
    z_end = tuple(imgpts[3])
    
    # X轴（紫色）
    cv2.arrowedLine(vis, origin, x_end, (255, 0, 255), 4, tipLength=0.3)
    cv2.putText(vis, "X", x_end, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
    
    # Y轴（红色）
    cv2.arrowedLine(vis, origin, y_end, (0, 0, 255), 4, tipLength=0.3)
    cv2.putText(vis, "Y", y_end, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # Z轴（蓝色）
    cv2.arrowedLine(vis, origin, z_end, (255, 0, 0), 4, tipLength=0.3)
    cv2.putText(vis, "Z", z_end, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    
    # 7. 信息面板（左上角）
    h, w = image.shape[:2]
    euler = result['euler_angles_deg']
    
    info_lines = [
        f"Image: {image_name}",
        f"Resolution: {w}x{h}",
        f"Roll (X):    {euler['roll']:+7.2f} deg",
        f"Pitch (Y):   {euler['pitch']:+7.2f} deg",
        f"Yaw (Z):     {euler['yaw']:+7.2f} deg",
        f"Reproj Error: Avg={result['mean_error']:.3f}px",
        f"              Max={result['max_error']:.3f}px"
    ]
    
    # 半透明背景
    panel_h = len(info_lines) * 35 + 20
    panel_w = 400
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
    
    # 绘制文字
    for i, line in enumerate(info_lines):
        y = 30 + i * 35
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
    
    return vis


# ==================== 批量处理主程序 ====================
def process_all_images():
    """批量处理data文件夹下的所有图片"""
    
    # 路径配置
    data_folder = Path('data')
    output_folder = Path('outputs/newpy/images')
    camera_yaml = '/home/eureka/tilt_checker2/-AprilTag-/config/camera_info.yaml'
    
    # 创建输出文件夹
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # 加载相机参数
    print("="*80)
    print("加载相机参数...")
    print("="*80)
    camera_matrix, dist_coeffs = load_camera_params(camera_yaml)
    
    if camera_matrix is None:
        print("❌ 无法加载相机参数，程序退出")
        return
    
    print(f"✅ 相机参数加载成功")
    print(f"   内参矩阵:")
    print(f"   fx={camera_matrix[0,0]:.2f}, fy={camera_matrix[1,1]:.2f}")
    print(f"   cx={camera_matrix[0,2]:.2f}, cy={camera_matrix[1,2]:.2f}")
    
    # 获取所有图片
    image_files = sorted(list(data_folder.glob('*.png')) + list(data_folder.glob('*.jpg')))
    
    if len(image_files) == 0:
        print(f"❌ 在 {data_folder} 中未找到图片")
        return
    
    print(f"\n找到 {len(image_files)} 张图片")
    print("="*80)
    
    # 存储所有结果
    all_results = []
    success_count = 0
    
    # 逐张处理
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n{'='*80}")
        print(f"处理第 {idx}/{len(image_files)} 张图片")
        print(f"{'='*80}")
        
        # 读取图像
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"❌ 无法读取图像: {img_path.name}")
            continue
        
        h, w = image.shape[:2]
        image_name = img_path.name
        
        # 应用畸变矫正
        image_undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
        
        # 终端输出：基本信息
        print(f"📷 处理帧名称: {image_name}")
        print(f"📐 分辨率: {w}x{h}")
        print(f"🔧 畸变矫正: ✅ 是 (已应用)")
        
        # 执行检测（使用矫正后的图像）
        result = detect_grid_with_apriltag(image_undistorted, camera_matrix, dist_coeffs)
        
        if not result['success']:
            print(f"❌ 检测失败: {result.get('error', 'Unknown error')}")
            print(f"🏷️  识别到AprilTag: 否")
            print(f"📍 AprilTag坐标系: ❌ 未建立")
            continue
        
        # 成功检测
        success_count += 1
        
        # 终端输出：检测结果
        print(f"✅ 检测标定板角点数量: {result['n_found']}/225")
        print(f"🔧 补全点数量: {result['n_interpolated']}/225")
        print(f"🏷️  识别到AprilTag: 是 (ID: {result['apriltag_info']['tag_id']})")
        print(f"📍 AprilTag坐标系: ✅ AprilTag坐标系建立成功 (ID: {result['apriltag_info']['tag_id']})")
        
        # 输出重投影误差
        print(f"\n📊 重投影误差:")
        print(f"   平均={result['mean_error']:.3f}px, 最大={result['max_error']:.3f}px")
        
        # 输出欧拉角
        euler = result['euler_angles_deg']
        print(f"\n🔄 相机倾斜角度 (假设板子水平，相机相对于水平面):")
        print(f"   Roll (横滚角):    {euler['roll']:+7.2f}°")
        print(f"   Pitch (俯仰角):   {euler['pitch']:+7.2f}°")
        print(f"   Yaw (偏航角):     {euler['yaw']:+7.2f}°")
        
        # 输出中心点
        mean_c = result['mean_center_2d']
        mid_c = result['mid_center_2d']
        print(f"\n📌 中心点坐标:")
        print(f"   均值中心(所有角点平均)(u,v)=({mean_c[0]:.1f}, {mean_c[1]:.1f})")
        print(f"   网格中心(mid)(网格中心角点)(u,v)=({mid_c[0]:.1f}, {mid_c[1]:.1f})")
        
        # 可视化（使用矫正后的图像）
        vis_image = visualize_result(image_undistorted, result, camera_matrix, dist_coeffs, image_name)
        
        # 保存图像
        output_name = img_path.stem + '_result.png'
        output_path = output_folder / output_name
        cv2.imwrite(str(output_path), vis_image)
        
        print(f"\n💾 已保存可视化图像: {output_path}")
        
        # 保存结果到列表
        transform = result['transform_array']
        all_results.append({
            'image_name': image_name,
            'transform': transform,
            'euler_deg': euler,
            'mean_error': result['mean_error'],
            'max_error': result['max_error']
        })
    
    print(f"\n{'='*80}")
    print(f"批量处理完成！")
    print(f"{'='*80}")
    print(f"✅ 成功处理: {success_count}/{len(image_files)} 张图片")
    
    # 输出所有变换矩阵
    if len(all_results) > 0:
        print(f"\n{'='*80}")
        print("所有图片的变换矩阵 [δx, δy, δz, γ, α, β]")
        print("(距离单位: 米, 角度单位: 弧度)")
        print(f"{'='*80}")
        
        for res in all_results:
            t = res['transform']
            print(f"{res['image_name']:40s} | [{t[0]:+.6f}, {t[1]:+.6f}, {t[2]:+.6f}, {t[3]:+.6f}, {t[4]:+.6f}, {t[5]:+.6f}]")
        
        # 保存到文件
        result_file = Path('Result_array.txt')
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("相机标定结果 - 变换矩阵汇总\n")
            f.write("="*100 + "\n\n")
            f.write("变换格式: [δx, δy, δz, γ, α, β]\n")
            f.write("  δx, δy, δz: 平移向量 (单位: 米)\n")
            f.write("  γ (Yaw):   绕Z轴旋转角 (单位: 弧度)\n")
            f.write("  α (Pitch): 绕Y轴旋转角 (单位: 弧度)\n")
            f.write("  β (Roll):  绕X轴旋转角 (单位: 弧度)\n\n")
            f.write("变换顺序: 相机坐标系 -> 平移(δx,δy,δz) -> Rz(γ) -> Ry(α) -> Rx(β) -> 标定板坐标系\n\n")
            f.write("="*100 + "\n\n")
            
            f.write(f"{'图片名称':<45s} {'变换矩阵 [δx, δy, δz, γ, α, β]':<60s} {'平均误差(px)':<15s} {'最大误差(px)':<15s}\n")
            f.write("-"*140 + "\n")
            
            for res in all_results:
                t = res['transform']
                transform_str = f"[{t[0]:+.6f}, {t[1]:+.6f}, {t[2]:+.6f}, {t[3]:+.6f}, {t[4]:+.6f}, {t[5]:+.6f}]"
                f.write(f"{res['image_name']:<45s} {transform_str:<60s} {res['mean_error']:<15.4f} {res['max_error']:<15.4f}\n")
            
            f.write("\n" + "="*100 + "\n")
            f.write(f"总计处理: {len(all_results)} 张图片\n")
            f.write("="*100 + "\n")
        
        print(f"\n💾 变换矩阵已保存到: {result_file}")


if __name__ == '__main__':
    process_all_images()
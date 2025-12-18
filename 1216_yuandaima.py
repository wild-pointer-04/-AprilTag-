#!/usr/bin/env python3
"""
AprilTag 引导的相机标定 - 完美融合版
结合Version4的单应性补全 + Version8的Ref点原点坐标系
"""

import cv2
import numpy as np
import yaml
from pathlib import Path

try:
    from pupil_apriltags import Detector
    USING_PUPIL = True
except ImportError:
    import apriltag
    USING_PUPIL = False


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


def detect_blobs(gray_image):
    """检测圆形blob"""
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = 30
    params.maxArea = 3000
    params.filterByCircularity = True
    params.minCircularity = 0.6
    params.filterByConvexity = True
    params.minConvexity = 0.7
    params.filterByInertia = True
    params.minInertiaRatio = 0.5

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray_image)
    blob_points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    print(f"\n[Blob检测] 共检测到 {len(blob_points)} 个候选点")
    return blob_points, keypoints


def filter_blobs_exclude_apriltag(blob_points, apriltag_info):
    """排除AprilTag区域"""
    if len(blob_points) == 0:
        return blob_points, np.array([])
    
    tag_center = apriltag_info['center']
    tag_corners = apriltag_info['corners']
    tag_size_px = np.mean([
        np.linalg.norm(tag_corners[1] - tag_corners[0]),
        np.linalg.norm(tag_corners[2] - tag_corners[1])
    ])
    
    exclusion_radius = tag_size_px * 1.1
    distances = np.linalg.norm(blob_points - tag_center, axis=1)
    valid_mask = distances > exclusion_radius
    filtered_blobs = blob_points[valid_mask]
    excluded_indices = np.where(~valid_mask)[0]
    return filtered_blobs, excluded_indices


def filter_blobs_by_grid_pattern(blob_points, apriltag_info):
    """基于网格模式过滤"""
    if len(blob_points) < 100:
        return blob_points, np.array([])
    
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(blob_points)
        distances, _ = tree.query(blob_points, k=5)
        nearest_distances = distances[:, 1]
        estimated_spacing = np.median(nearest_distances)
        
        min_spacing = estimated_spacing * 0.7
        max_spacing = estimated_spacing * 1.4
        
        valid_neighbor_counts = []
        for i, point in enumerate(blob_points):
            neighbors = tree.query_ball_point(point, r=max_spacing)
            valid_neighbors = 0
            for j in neighbors:
                if i == j:
                    continue
                dist = np.linalg.norm(blob_points[i] - blob_points[j])
                if min_spacing <= dist <= max_spacing:
                    valid_neighbors += 1
            valid_neighbor_counts.append(valid_neighbors)
        
        valid_neighbor_counts = np.array(valid_neighbor_counts)
        valid_mask = valid_neighbor_counts >= 2
        filtered_blobs = blob_points[valid_mask]
        outlier_indices = np.where(~valid_mask)[0]
        return filtered_blobs, outlier_indices
    except ImportError:
        return blob_points, np.array([])


def find_four_corner_blobs(blob_points):
    """找到标定板的四个角点"""
    if len(blob_points) < 4:
        return None, None

    rect = cv2.minAreaRect(blob_points)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="float32")

    sorted_y_indices = np.argsort(box[:, 1])
    top_2_indices = sorted_y_indices[:2]
    bottom_2_indices = sorted_y_indices[2:]
    
    top_2 = box[top_2_indices]
    bottom_2 = box[bottom_2_indices]
    
    if top_2[0][0] < top_2[1][0]:
        tl_ideal, tr_ideal = top_2[0], top_2[1]
    else:
        tl_ideal, tr_ideal = top_2[1], top_2[0]
        
    if bottom_2[0][0] < bottom_2[1][0]:
        bl_ideal, br_ideal = bottom_2[0], bottom_2[1]
    else:
        bl_ideal, br_ideal = bottom_2[1], bottom_2[0]

    corner_positions = {
        'top_left': tl_ideal,
        'top_right': tr_ideal,
        'bottom_left': bl_ideal,
        'bottom_right': br_ideal
    }
    
    corners_dict = {}
    corners_indices = {}
    used_indices = set()
    
    print(f"\n[查找四个角点]")
    for corner_name, ideal_pos in corner_positions.items():
        distances = np.linalg.norm(blob_points - ideal_pos, axis=1)
        sorted_dist_indices = np.argsort(distances)
        
        found_idx = -1
        for idx in sorted_dist_indices:
            if idx not in used_indices:
                found_idx = idx
                break
        
        if found_idx == -1:
            found_idx = sorted_dist_indices[0]
            
        nearest_blob = blob_points[found_idx]
        used_indices.add(found_idx)
        corners_dict[corner_name] = nearest_blob
        corners_indices[corner_name] = found_idx
        print(f"  {corner_name}: ({nearest_blob[0]:.1f}, {nearest_blob[1]:.1f})")
    
    return corners_dict, corners_indices


def find_corner_near_apriltag(corners_dict, corners_indices, apriltag_info):
    """
    找到最靠近AprilTag的角点作为Ref（保持向后兼容）
    如果 AprilTag 固定在右上，则优先返回 top_right
    """
    # 固定场景：AprilTag在右上角，直接返回 top_right，更稳
    if 'top_right' in corners_dict:
        return corners_dict['top_right'], corners_indices['top_right'], 'top_right'
    
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


def get_grid_orientation_from_corners(corners_dict, corner_name):
    """
    从标定板的四个角点计算网格方向
    比从AprilTag计算更准确
    """
    tl = corners_dict['top_left']
    tr = corners_dict['top_right']
    bl = corners_dict['bottom_left']
    br = corners_dict['bottom_right']
    
    # 计算X方向（水平）：从左到右
    x_vec_top = tr - tl
    x_vec_bottom = br - bl
    x_vec = (x_vec_top + x_vec_bottom) / 2  # 取平均
    unit_x = x_vec / np.linalg.norm(x_vec)
    
    # 计算Y方向（垂直）：从上到下
    y_vec_left = bl - tl
    y_vec_right = br - tr
    y_vec = (y_vec_left + y_vec_right) / 2  # 取平均
    unit_y = y_vec / np.linalg.norm(y_vec)
    
    print(f"\n[方向计算] 使用标定板四角点计算网格方向")
    print(f"  X方向(水平): {unit_x}")
    print(f"  Y方向(垂直): {unit_y}")
    
    return unit_x, unit_y


def estimate_spacing_from_corners(corners_dict, pattern_size=(15, 15)):
    """
    从四个角点估算圆点间距
    更准确，因为考虑了整个标定板的尺寸
    """
    cols, rows = pattern_size
    
    tl = corners_dict['top_left']
    tr = corners_dict['top_right']
    bl = corners_dict['bottom_left']
    br = corners_dict['bottom_right']
    
    # 水平方向：top_left 到 top_right 的距离 / (cols-1)
    horizontal_dist_top = np.linalg.norm(tr - tl)
    horizontal_dist_bottom = np.linalg.norm(br - bl)
    spacing_horizontal = (horizontal_dist_top + horizontal_dist_bottom) / 2 / (cols - 1)
    
    # 垂直方向：top_left 到 bottom_left 的距离 / (rows-1)
    vertical_dist_left = np.linalg.norm(bl - tl)
    vertical_dist_right = np.linalg.norm(br - tr)
    spacing_vertical = (vertical_dist_left + vertical_dist_right) / 2 / (rows - 1)
    
    # 取平均
    spacing_px = (spacing_horizontal + spacing_vertical) / 2
    
    print(f"\n[间距估算] 从四角点计算")
    print(f"  水平间距: {spacing_horizontal:.2f}px")
    print(f"  垂直间距: {spacing_vertical:.2f}px")
    print(f"  平均间距: {spacing_px:.2f}px")
    
    return spacing_px


def build_grid_from_ref_point(ref_blob, corner_name, unit_x, unit_y,
                              spacing_px, pattern_size=(15, 15)):
    """
    以Ref点为起点构建网格
    不依赖左上角，直接从Ref点向四周扩展
    """
    cols, rows = pattern_size
    
    # 确定Ref点在网格中的逻辑位置
    ref_positions = {
        'top_left': (0, 0),
        'top_right': (0, cols - 1),
        'bottom_left': (rows - 1, 0),
        'bottom_right': (rows - 1, cols - 1)
    }
    
    ref_row, ref_col = ref_positions.get(corner_name, (0, cols - 1))
    
    grid_points = np.zeros((rows, cols, 2), dtype=np.float32)
    
    # 从Ref点向四周构建
    for r in range(rows):
        for c in range(cols):
            # 相对于Ref点的偏移
            delta_col = c - ref_col
            delta_row = r - ref_row
            
            # 从Ref点出发计算位置
            point = ref_blob + unit_x * delta_col * spacing_px + unit_y * delta_row * spacing_px
            grid_points[r, c] = point
    
    print(f"\n[网格构建] 以{corner_name}的Ref点为起点，向四周构建15x15网格")
    print(f"  Ref点位置: 网格({ref_row},{ref_col}) = 物理坐标{ref_blob}")
    
    return grid_points


def simple_greedy_matching(cost_matrix, threshold=1e6):
    """贪心匹配"""
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
    """匹配blob到网格（简单贪心，确保Ref列行保持）"""
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
                               pattern_size=(15, 15), ref_grid_pos=None, ref_point=None):
    """
    使用单应性矩阵优化网格位置
    这是Version 4的核心 - 保证补全点位置准确
    """
    if len(matched_corners) < 4:
        return grid_points, None
    
    src_points = []
    dst_points = []
    
    for i in range(len(matched_indices)):
        row, col = matched_indices[i]
        src_points.append([col, row])
        dst_points.append(matched_corners[i, 0])
    
    # 强制加入Ref约束，确保单应性不会偏移原点
    if ref_grid_pos is not None and ref_point is not None:
        ref_r, ref_c = ref_grid_pos
        src_points.append([ref_c, ref_r])
        dst_points.append(ref_point)
    
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    
    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    
    if H is None:
        return grid_points, None
    
    rows, cols = pattern_size
    all_grid_logical = []
    for r in range(rows):
        for c in range(cols):
            all_grid_logical.append([c, r])
    
    all_grid_logical = np.array(all_grid_logical, dtype=np.float32).reshape(-1, 1, 2)
    refined_points_flat = cv2.perspectiveTransform(all_grid_logical, H)
    refined_grid_points = refined_points_flat.reshape(rows, cols, 2)
    
    print(f"[单应性优化] 使用{len(matched_corners)}个匹配点计算单应性矩阵")
    
    return refined_grid_points, H


def match_and_fill_missing_points(refined_grid_points, all_blobs, 
                                  search_radius=20.0, pattern_size=(15, 15)):
    """
    智能补全网格 - Version 4的方法
    返回最终匹配的blob点及其网格索引
    """
    rows, cols = pattern_size
    final_grid = np.zeros_like(refined_grid_points)
    status_mask = np.zeros((rows, cols), dtype=np.uint8)
    used_blobs = set()
    
    matched_count = 0
    interpolated_count = 0
    
    # 记录最终匹配的点
    final_matched_blobs = []
    final_matched_grid_indices = []
    
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
                    # 记录匹配信息
                    final_matched_blobs.append(all_blobs[min_dist_idx])
                    final_matched_grid_indices.append([r, c])
                else:
                    final_grid[r, c] = theoretical_pt
                    status_mask[r, c] = 0
                    interpolated_count += 1
            else:
                final_grid[r, c] = theoretical_pt
                status_mask[r, c] = 0
                interpolated_count += 1
    
    print(f"[智能补全] 匹配: {matched_count}, 插值: {interpolated_count}")
    
    # 转换为numpy数组
    final_matched_blobs = np.array(final_matched_blobs, dtype=np.float32) if final_matched_blobs else np.array([])
    final_matched_grid_indices = np.array(final_matched_grid_indices, dtype=np.int32) if final_matched_grid_indices else np.array([])
    
    return final_grid, status_mask, matched_count, interpolated_count, final_matched_blobs, final_matched_grid_indices


def get_ref_grid_position(corner_name, pattern_size=(15, 15)):
    """获取Ref点在标准网格中的位置"""
    cols, rows = pattern_size
    
    ref_positions = {
        'top_left': (0, 0),
        'top_right': (0, cols - 1),
        'bottom_left': (rows - 1, 0),
        'bottom_right': (rows - 1, cols - 1)
    }
    
    return ref_positions.get(corner_name, (0, cols - 1))


def solve_pnp_with_ref_origin(final_grid_points, ref_grid_pos, camera_matrix,
                               dist_coeffs, circle_spacing_m=0.02, pattern_size=(15, 15)):
    """
    PnP求解 - 以Ref点为原点 (Version 8的方法)
    """
    rows, cols = pattern_size
    ref_row, ref_col = ref_grid_pos
    
    object_points = []
    image_points = []
    
    for r in range(rows):
        for c in range(cols):
            # 3D坐标相对于Ref点
            x_3d = (c - ref_col) * circle_spacing_m
            y_3d = (r - ref_row) * circle_spacing_m
            z_3d = 0.0
            
            object_points.append([x_3d, y_3d, z_3d])
            image_points.append(final_grid_points[r, c])
    
    object_points = np.array(object_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)
    
    print(f"\n[PnP求解] Ref点网格位置: ({ref_row}, {ref_col}) -> 3D原点(0,0,0)")
    print(f"[PnP求解] 3D范围: X=[{np.min(object_points[:,0]):.3f}, {np.max(object_points[:,0]):.3f}]m, "
          f"Y=[{np.min(object_points[:,1]):.3f}, {np.max(object_points[:,1]):.3f}]m")
    
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return None, None, None, None, None
    
    projected_points, _ = cv2.projectPoints(object_points, rvec, tvec,
                                           camera_matrix, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2)
    
    errors = np.linalg.norm(image_points - projected_points, axis=1)
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    R_camera_to_board, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R_camera_to_board[0, 0]**2 + R_camera_to_board[1, 0]**2)
    singular = sy < 1e-6
    
    if not singular:
        beta = np.arctan2(R_camera_to_board[2, 1], R_camera_to_board[2, 2])
        alpha = np.arctan2(-R_camera_to_board[2, 0], sy)
        gamma = np.arctan2(R_camera_to_board[1, 0], R_camera_to_board[0, 0])
    else:
        beta = np.arctan2(-R_camera_to_board[1, 2], R_camera_to_board[1, 1])
        alpha = np.arctan2(-R_camera_to_board[2, 0], sy)
        gamma = 0
    
    delta_x = tvec[0, 0]
    delta_y = tvec[1, 0]
    delta_z = tvec[2, 0]
    
    transform_array = np.array([delta_x, delta_y, delta_z, gamma, alpha, beta])
    euler_angles_deg = {
        'roll': np.degrees(beta),
        'pitch': np.degrees(alpha),
        'yaw': np.degrees(gamma)
    }
    
    return transform_array, euler_angles_deg, mean_error, max_error, (rvec, tvec)


def detect_grid_with_apriltag(image, camera_matrix, dist_coeffs, max_distance=100.0):
    """主检测函数 - 融合Version 4和Version 8"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    apriltag_info = detect_apriltag(gray)
    if apriltag_info is None:
        return {'success': False, 'error': 'No AprilTag detected'}
    
    blob_points, keypoints = detect_blobs(gray)
    if len(blob_points) == 0:
        return {'success': False, 'error': 'No blobs detected'}
    
    # 过滤
    filtered_blob_points, excluded_indices = filter_blobs_exclude_apriltag(
        blob_points, apriltag_info
    )
    
    filtered_blob_points, outlier_indices = filter_blobs_by_grid_pattern(
        filtered_blob_points, apriltag_info
    )
    
    if len(filtered_blob_points) < 180:
        print(f"  ⚠️  过滤后点数不足，使用保守策略")
        filtered_blob_points, _ = filter_blobs_exclude_apriltag(blob_points, apriltag_info)
    
    # 找角点
    corners_dict, corners_indices_dict = find_four_corner_blobs(filtered_blob_points)
    if corners_dict is None:
        return {'success': False, 'error': 'Cannot find corner blobs'}
    
    # 找Ref点
    corner_blob, corner_idx_filtered, corner_name = find_corner_near_apriltag(
        corners_dict, corners_indices_dict, apriltag_info
    )
    
    # 保存corners_dict用于可视化
    result_corners_dict = corners_dict
    
    # 获取方向 - 改用标定板角点计算，而不是AprilTag
    unit_x, unit_y = get_grid_orientation_from_corners(corners_dict, corner_name)
    
    # 获取间距 - 改用角点距离计算
    spacing_px = estimate_spacing_from_corners(corners_dict, pattern_size=(15, 15))
    if spacing_px is None or spacing_px < 10:
        return {'success': False, 'error': 'Cannot estimate spacing'}
    
    # ===== Version 4改进: 从Ref点构建网格 =====
    coarse_grid_points = build_grid_from_ref_point(
        corner_blob, corner_name, unit_x, unit_y, spacing_px, pattern_size=(15, 15)
    )
    
    # ===== Version 4: 初始匹配 =====
    matched_corners, matched_indices = match_blobs_to_grid(
        filtered_blob_points, coarse_grid_points, max_distance=max_distance
    )
    
    if matched_corners is None or len(matched_corners) < 4:
        return {'success': False, 'error': 'Insufficient matched points'}
    
    # ===== Version 4: 单应性优化 =====
    theoretical_grid_points, H = refine_grid_with_homography(
        coarse_grid_points, matched_indices, matched_corners,
        pattern_size=(15, 15), ref_grid_pos=get_ref_grid_position(corner_name), ref_point=corner_blob
    )
    
    if H is None:
        return {'success': False, 'error': 'Homography failed'}
    
    # ===== Version 4: 智能补全 =====
    search_radius = spacing_px * 0.5
    final_grid_points, status_mask, n_found, n_interp, final_matched_blobs, final_matched_indices = match_and_fill_missing_points(
        theoretical_grid_points, filtered_blob_points, 
        search_radius=search_radius, pattern_size=(15, 15)
    )
    
    # 强制Ref点使用原始blob
    ref_grid_pos = get_ref_grid_position(corner_name)
    r_ref, c_ref = ref_grid_pos
    final_grid_points[r_ref, c_ref] = corner_blob
    status_mask[r_ref, c_ref] = 1
    print(f"[强制锚点] 网格位置({r_ref},{c_ref}) = Ref点 = {corner_blob}")
    
    # ===== Version 8: PnP以Ref为原点 =====
    result = solve_pnp_with_ref_origin(
        final_grid_points, ref_grid_pos, camera_matrix, dist_coeffs
    )
    
    if result[0] is None:
        return {'success': False, 'error': 'PnP solving failed'}
    
    transform_array, euler_angles_deg, mean_error, max_error, pose = result
    
    return {
        'success': True,
        'final_grid_points': final_grid_points,
        'theoretical_grid_points': theoretical_grid_points,
        'status_mask': status_mask,
        'apriltag_info': apriltag_info,
        'corner_name': corner_name,
        'ref_grid_pos': ref_grid_pos,
        'corners_dict': result_corners_dict,  # 四个角点
        'final_matched_blobs': final_matched_blobs,  # 最终匹配的blob点
        'final_matched_indices': final_matched_indices,  # 最终匹配的网格索引
        'spacing_px': spacing_px,
        'all_keypoints': keypoints,
        'transform_array': transform_array,
        'euler_angles_deg': euler_angles_deg,
        'mean_error': mean_error,
        'max_error': max_error,
        'pose': pose,
        'n_found': n_found,
        'n_interpolated': n_interp
    }


def visualize_result(image, result, camera_matrix, dist_coeffs, image_name):
    """可视化 - 增强版"""
    vis = image.copy()

    # 绘制所有原始blob点（绿色圆圈）
    if 'all_keypoints' in result:
        for kp in result['all_keypoints']:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            radius = int(kp.size / 2) + 2
            cv2.circle(vis, (x, y), radius, (0, 255, 0), 1)
    
    # 绘制AprilTag
    tag_info = result['apriltag_info']
    tag_corners = tag_info['corners'].astype(int)
    tag_center = tag_info['center'].astype(int)
    
    cv2.polylines(vis, [tag_corners], True, (0, 255, 0), 3)
    cv2.putText(vis, f"ID:{tag_info['tag_id']}", 
                tuple(tag_center + np.array([10, -10])),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    
    # ===== 新增：绘制四个角点（蓝色方块）=====
    if 'corners_dict' in result:
        corners_dict = result['corners_dict']
        for corner_name, corner_pt in corners_dict.items():
            pt = tuple(corner_pt.astype(int))
            # 绘制蓝色实心方块
            cv2.rectangle(vis, (pt[0]-8, pt[1]-8), (pt[0]+8, pt[1]+8), (255, 0, 0), -1)
            # 标注角点名称
            label_offset = {'top_left': (-80, -15), 'top_right': (15, -15),
                          'bottom_left': (-80, 25), 'bottom_right': (15, 25)}
            offset = label_offset.get(corner_name, (15, -15))
            cv2.putText(vis, corner_name, (pt[0]+offset[0], pt[1]+offset[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # ===== 新增：绘制最终匹配点（红色圆点）=====
    if 'final_matched_blobs' in result and len(result['final_matched_blobs']) > 0:
        final_matched_blobs = result['final_matched_blobs']
        for i in range(len(final_matched_blobs)):
            pt = tuple(final_matched_blobs[i].astype(int))
            # 绘制红色小圆点
            cv2.circle(vis, pt, 3, (0, 0, 255), -1)
    
    # 绘制网格点
    grid = result['final_grid_points']
    mask = result['status_mask']
    rows, cols = grid.shape[:2]
    
    for r in range(rows):
        for c in range(cols):
            pt = tuple(grid[r, c].astype(int))
            if mask[r, c] == 1:
                # 匹配成功的点（黄色实心圆）
                cv2.circle(vis, pt, 5, (0, 255, 255), -1)
                cv2.circle(vis, pt, 7, (0, 255, 0), 1)
            else:
                # 插值的点（蓝色十字）
                cv2.drawMarker(vis, pt, (255, 0, 0), cv2.MARKER_CROSS, 15, 2)
    
    # 标注Ref点（橙色大圆圈）
    r_ref, c_ref = result['ref_grid_pos']
    ref_pt = tuple(grid[r_ref, c_ref].astype(int))
    cv2.circle(vis, ref_pt, 20, (0, 165, 255), 3)
    cv2.putText(vis, "Ref(0,0,0)", (ref_pt[0]+25, ref_pt[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    
    cv2.line(vis, tuple(tag_center), ref_pt, (255, 0, 255), 2, cv2.LINE_AA)
    
    # 绘制坐标轴
    axis_length = 0.1
    axis_3d = np.float32([
        [0, 0, 0],
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, -axis_length]
    ])
    
    rvec, tvec = result['pose']
    imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.reshape(-1, 2).astype(int)
    
    origin = tuple(imgpts[0])
    cv2.arrowedLine(vis, origin, tuple(imgpts[1]), (255, 0, 255), 5, tipLength=0.3)
    cv2.arrowedLine(vis, origin, tuple(imgpts[2]), (0, 0, 255), 5, tipLength=0.3)
    cv2.arrowedLine(vis, origin, tuple(imgpts[3]), (255, 0, 0), 5, tipLength=0.3)
    
    cv2.putText(vis, "X", tuple(imgpts[1] + [15, 0]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    cv2.putText(vis, "Y", tuple(imgpts[2] + [15, 0]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(vis, "Z", tuple(imgpts[3] + [15, 0]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # 信息面板
    euler = result['euler_angles_deg']
    matched_count = len(result['final_matched_blobs']) if 'final_matched_blobs' in result else 0
    info_lines = [
        f"Image: {image_name}",
        f"Roll: {euler['roll']:+.2f} | Pitch: {euler['pitch']:+.2f} | Yaw: {euler['yaw']:+.2f}",
        f"Error: Avg={result['mean_error']:.3f}px | Matched: {matched_count}/225"
    ]
    
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (650, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
    
    for i, line in enumerate(info_lines):
        cv2.putText(vis, line, (10, 30 + i * 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return vis


def process_all_images():
    """批量处理"""
    data_folder = Path('data')
    output_folder = Path('outputs/newpy/images')
    camera_yaml = '/home/eureka/tilt_checker2/-AprilTag-/config/camera_info.yaml'
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("加载相机参数...")
    camera_matrix, dist_coeffs = load_camera_params(camera_yaml)
    
    if camera_matrix is None:
        return
    
    image_files = sorted(list(data_folder.glob('*.png')) + list(data_folder.glob('*.jpg')))
    if len(image_files) == 0:
        print(f"❌ 在 {data_folder} 中未找到图片")
        return
    
    all_results = []
    success_count = 0
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n{'='*80}")
        print(f"处理第 {idx}/{len(image_files)} 张图片: {img_path.name}")
        
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        image_undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
        result = detect_grid_with_apriltag(image_undistorted, camera_matrix, dist_coeffs)
        
        if not result['success']:
            print(f"❌ 检测失败: {result.get('error', 'Unknown')}")
            continue
        
        success_count += 1
        print(f"✅ 成功! 误差: Avg={result['mean_error']:.3f}px | Ref: {result['corner_name']}")
        
        vis_image = visualize_result(image_undistorted, result, camera_matrix,
                                     dist_coeffs, img_path.name)
        
        output_name = img_path.stem + '_result.png'
        cv2.imwrite(str(output_folder / output_name), vis_image)
        
        all_results.append({
            'image_name': img_path.name,
            'transform': result['transform_array'],
            'mean_error': result['mean_error'],
            'max_error': result['max_error']
        })
    
    print(f"\n{'='*80}")
    print(f"批量处理完成！成功: {success_count}/{len(image_files)}")
    
    if len(all_results) > 0:
        result_file = Path('Result_array.txt')
        with open(result_file, 'w') as f:
            for res in all_results:
                t = res['transform']
                line = f"{res['image_name']} [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}, {t[3]:.6f}, {t[4]:.6f}, {t[5]:.6f}] {res['mean_error']:.4f}\n"
                f.write(line)
        print(f"💾 结果已保存到: {result_file}")


if __name__ == '__main__':
    process_all_images()
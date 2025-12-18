#!/usr/bin/env python3

"""

正确的 AprilTag 引导网格检测



策略：

1. AprilTag 只用来确定方向（X轴、Y轴）

2. 找到距离 AprilTag 最近的实际圆点（blob）

3. 以这个圆点作为右上角 (0, 14)，推算整个网格

"""



import cv2

import numpy as np



try:

    from pupil_apriltags import Detector

    USING_PUPIL = True

except ImportError:

    import apriltag

    USING_PUPIL = False





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

    

    if USING_PUPIL:

        corners = np.array(detection.corners, dtype=np.float64)

        center = np.array(detection.center, dtype=np.float64)

        tag_id = detection.tag_id

    else:

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

    params.minArea = 20

    params.maxArea = 5000

    params.filterByCircularity = True

    params.minCircularity = 0.7

    params.filterByConvexity = True

    params.minConvexity = 0.8

    params.filterByInertia = True

    params.minInertiaRatio = 0.6

    

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(gray_image)

    

    blob_points = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    return blob_points, keypoints





def filter_blobs_exclude_apriltag(blob_points, apriltag_info, apriltag_size_m=0.0714):

    """

    排除 AprilTag 区域的 blob

    

    Args:

        blob_points: 所有检测到的 blob

        apriltag_info: AprilTag 信息

        apriltag_size_m: AprilTag 实际尺寸（米），用于计算排除范围

    

    Returns:

        filtered_blobs: 排除 AprilTag 后的 blob

        excluded_indices: 被排除的 blob 索引

    """

    tag_center = apriltag_info['center']

    tag_corners = apriltag_info['corners']

    

    # 计算 AprilTag 的像素尺寸

    tag_size_px = np.mean([

        np.linalg.norm(tag_corners[1] - tag_corners[0]),

        np.linalg.norm(tag_corners[2] - tag_corners[1])

    ])

    

    # 排除范围：AprilTag 中心周围 2 倍尺寸

    exclusion_radius = tag_size_px * 1.5

    

    print(f"\n[排除 AprilTag 区域]")

    print(f"  AprilTag 中心: ({tag_center[0]:.1f}, {tag_center[1]:.1f})")

    print(f"  AprilTag 尺寸: {tag_size_px:.1f} px")

    print(f"  排除半径: {exclusion_radius:.1f} px")

    

    # 过滤

    distances = np.linalg.norm(blob_points - tag_center, axis=1)

    valid_mask = distances > exclusion_radius

    

    filtered_blobs = blob_points[valid_mask]

    excluded_indices = np.where(~valid_mask)[0]

    

    print(f"  排除了 {len(excluded_indices)} 个 blob")

    print(f"  剩余 {len(filtered_blobs)} 个有效 blob")

    

    return filtered_blobs, excluded_indices





def find_four_corner_blobs(blob_points, image_shape):

    """

    找到标定板的四个角点

    

    策略：

    1. 找到 blob 分布的边界框

    2. 找距离四个角最近的 blob

    

    Returns:

        corners_dict: {'top_left': blob, 'top_right': blob, 'bottom_left': blob, 'bottom_right': blob}

        corners_indices: 对应的索引

    """

    if len(blob_points) == 0:

        return None, None

    

    print(f"\n[查找四个角点]")

    

    # 计算 blob 分布的边界

    min_x = np.min(blob_points[:, 0])

    max_x = np.max(blob_points[:, 0])

    min_y = np.min(blob_points[:, 1])

    max_y = np.max(blob_points[:, 1])

    

    print(f"  Blob 分布范围:")

    print(f"    X: {min_x:.1f} - {max_x:.1f}")

    print(f"    Y: {min_y:.1f} - {max_y:.1f}")

    

    # 定义四个角的理想位置

    corner_positions = {

        'top_left': np.array([min_x, min_y]),

        'top_right': np.array([max_x, min_y]),

        'bottom_left': np.array([min_x, max_y]),

        'bottom_right': np.array([max_x, max_y])

    }

    

    # 找距离每个角最近的 blob

    corners_dict = {}

    corners_indices = {}

    

    for corner_name, ideal_pos in corner_positions.items():

        distances = np.linalg.norm(blob_points - ideal_pos, axis=1)

        nearest_idx = np.argmin(distances)

        nearest_blob = blob_points[nearest_idx]

        nearest_dist = distances[nearest_idx]

        

        corners_dict[corner_name] = nearest_blob

        corners_indices[corner_name] = nearest_idx

        

        print(f"  {corner_name}: ({nearest_blob[0]:.1f}, {nearest_blob[1]:.1f}), dist={nearest_dist:.1f}px")

    

    return corners_dict, corners_indices





def find_corner_near_apriltag(corners_dict, corners_indices, apriltag_info):

    """

    从四个角点中找到最靠近 AprilTag 的那个

    这个点应该是右上角

    """

    tag_center = apriltag_info['center']

    

    print(f"\n[选择最近的角点]")

    print(f"  AprilTag 中心: ({tag_center[0]:.1f}, {tag_center[1]:.1f})")

    

    min_dist = float('inf')

    nearest_corner_name = None

    nearest_corner_blob = None

    nearest_corner_idx = None

    

    for corner_name, corner_blob in corners_dict.items():

        dist = np.linalg.norm(corner_blob - tag_center)

        print(f"  {corner_name}: dist={dist:.1f}px")

        

        if dist < min_dist:

            min_dist = dist

            nearest_corner_name = corner_name

            nearest_corner_blob = corner_blob

            nearest_corner_idx = corners_indices[corner_name]

    

    print(f"  ✓ 选择: {nearest_corner_name}")

    print(f"    位置: ({nearest_corner_blob[0]:.1f}, {nearest_corner_blob[1]:.1f})")

    print(f"    距离: {min_dist:.1f}px")

    

    # 验证：应该是 top_right

    if nearest_corner_name != 'top_right':

        print(f"  ⚠️  警告：最近的角点不是 top_right，可能 AprilTag 位置不对")

    

    return nearest_corner_blob, nearest_corner_idx, nearest_corner_name





def determine_grid_origin_from_corner(corner_name, corner_blob, unit_x, unit_y, 

                                      circle_spacing_px, pattern_size=(15, 15)):

    """

    根据识别的角点位置，计算网格左上角原点

    

    Args:

        corner_name: 'top_left', 'top_right', 'bottom_left', 'bottom_right'

        corner_blob: 角点位置

        unit_x, unit_y: 方向向量

        circle_spacing_px: 间距

        pattern_size: 网格尺寸 (cols, rows)

    

    Returns:

        grid_origin: 左上角 (0, 0) 的位置

        corner_grid_pos: 该角点在网格中的位置 (row, col)

    """

    cols, rows = pattern_size

    

    # 根据角点位置确定其在网格中的坐标

    if corner_name == 'top_left':

        corner_grid_pos = (0, 0)

        offset_x = 0

        offset_y = 0

    elif corner_name == 'top_right':

        corner_grid_pos = (0, cols - 1)

        offset_x = cols - 1

        offset_y = 0

    elif corner_name == 'bottom_left':

        corner_grid_pos = (rows - 1, 0)

        offset_x = 0

        offset_y = rows - 1

    elif corner_name == 'bottom_right':

        corner_grid_pos = (rows - 1, cols - 1)

        offset_x = cols - 1

        offset_y = rows - 1

    else:

        raise ValueError(f"Unknown corner name: {corner_name}")

    

    # 计算左上角 (0, 0) 的位置

    grid_origin = corner_blob - unit_x * offset_x * circle_spacing_px - unit_y * offset_y * circle_spacing_px

    

    print(f"\n[网格构建]")

    print(f"  参考角点: {corner_name} = 网格 {corner_grid_pos}")

    print(f"  参考点位置: ({corner_blob[0]:.1f}, {corner_blob[1]:.1f})")

    print(f"  网格原点 (0,0): ({grid_origin[0]:.1f}, {grid_origin[1]:.1f})")

    print(f"  偏移: 向左 {offset_x} 格, 向上 {offset_y} 格")

    

    return grid_origin, corner_grid_pos





def get_grid_orientation_from_apriltag(apriltag_info):

    """

    从 AprilTag 获取网格的方向

    返回 X 轴（向右）和 Y 轴（向下）的单位向量

    """

    tag_corners = apriltag_info['corners']

    

    # 按 Y 坐标排序找出上下

    corners_with_idx = [(i, tag_corners[i]) for i in range(4)]

    corners_with_idx.sort(key=lambda x: x[1][1])

    

    top_two = corners_with_idx[:2]

    bottom_two = corners_with_idx[2:]

    

    # 在上方找左右

    top_two.sort(key=lambda x: x[1][0])

    bottom_two.sort(key=lambda x: x[1][0])

    

    top_left = top_two[0][1]

    top_right = top_two[1][1]

    bottom_left = bottom_two[0][1]

    

    # X 轴：从左上到右上

    x_vec = top_right - top_left

    x_len = np.linalg.norm(x_vec)

    unit_x = x_vec / x_len

    

    # Y 轴：从左上到左下

    y_vec = bottom_left - top_left

    y_len = np.linalg.norm(y_vec)

    unit_y = y_vec / y_len

    

    print(f"\n[网格方向]")

    print(f"  X轴（向右）: ({unit_x[0]:.3f}, {unit_x[1]:.3f})")

    print(f"  Y轴（向下）: ({unit_y[0]:.3f}, {unit_y[1]:.3f})")

    

    # 验证 Y 轴向下

    if unit_y[1] < 0:

        print(f"  ⚠️  Y轴向上，这不应该发生")

    

    return unit_x, unit_y





def estimate_grid_spacing_robust(blob_points, percentile=100):

    """

    更鲁棒的间距估算

    使用中位数而不是直方图峰值

    """

    if len(blob_points) < 10:

        return None

    

    # 尝试使用 scipy

    try:

        from scipy.spatial import cKDTree

        tree = cKDTree(blob_points)

        # 找每个点的最近邻（k=2 因为第一个是自己）

        distances, _ = tree.query(blob_points, k=2)

        nearest_distances = distances[:, 1]  # 第二近的就是最近邻

        

        # 使用中位数

        spacing_px = np.median(nearest_distances)

        

        print(f"\n[估算圆点间距（中位数法）]")

        print(f"  间距: {spacing_px:.2f} px")

        print(f"  范围: {np.min(nearest_distances):.1f} - {np.max(nearest_distances):.1f} px")

        

        return spacing_px

    except:

        # scipy 不可用，使用简单方法

        return estimate_grid_spacing_simple(blob_points)





def estimate_grid_spacing_simple(blob_points):

    """简单的间距估算（不依赖 scipy）"""

    if len(blob_points) < 10:

        return None

    

    # 计算所有点对的距离

    distances = []

    sample_size = min(100, len(blob_points))

    

    for i in range(sample_size):

        for j in range(i+1, sample_size):

            dist = np.linalg.norm(blob_points[i] - blob_points[j])

            if 20 < dist < 200:

                distances.append(dist)

    

    if len(distances) == 0:

        return None

    

    # 使用中位数

    spacing_px = np.median(distances)

    

    print(f"\n[估算圆点间距（简单法）]")

    print(f"  间距中位数: {spacing_px:.2f} px")

    

    return spacing_px





def build_grid_from_origin(grid_origin, unit_x, unit_y, 

                          circle_spacing_px, pattern_size=(15, 15),

                          image_shape=None):

    """

    从左上角原点构建完整网格

    """

    cols, rows = pattern_size

    

    # 生成网格

    grid_points = np.zeros((rows, cols, 2), dtype=np.float32)

    valid_mask = np.zeros((rows, cols), dtype=bool)

    

    if image_shape is not None:

        h, w = image_shape[:2]

        margin = 20

    else:

        margin = 0

        h, w = 99999, 99999

    

    for row in range(rows):

        for col in range(cols):

            point = grid_origin + unit_x * col * circle_spacing_px + unit_y * row * circle_spacing_px

            grid_points[row, col] = point

            

            if margin <= point[0] < w - margin and margin <= point[1] < h - margin:

                valid_mask[row, col] = True

    

    # 验证关键点

    print(f"\n[关键网格点]")

    print(f"  (0,0) 左上角: {grid_points[0, 0]}")

    print(f"  (0,{cols-1}) 右上角: {grid_points[0, cols-1]}")

    print(f"  ({rows-1},0) 左下角: {grid_points[rows-1, 0]}")

    print(f"  ({rows-1},{cols-1}) 右下角: {grid_points[rows-1, cols-1]}")

    print(f"  有效点数: {np.sum(valid_mask)}/{rows*cols}")

    

    return grid_points, valid_mask





def simple_greedy_matching(cost_matrix, threshold=1e6):

    """简单贪心匹配"""

    n_rows, n_cols = cost_matrix.shape

    

    matches = []

    for i in range(n_rows):

        for j in range(n_cols):

            if cost_matrix[i, j] < threshold:

                matches.append((cost_matrix[i, j], i, j))

    

    matches.sort()

    

    used_rows = set()

    used_cols = set()

    result_rows = []

    result_cols = []

    

    for cost, i, j in matches:

        if i not in used_rows and j not in used_cols:

            result_rows.append(i)

            result_cols.append(j)

            used_rows.add(i)

            used_cols.add(j)

    

    return np.array(result_rows), np.array(result_cols)





def match_blobs_to_grid(blob_points, grid_points, valid_mask, max_distance=100.0):

    """匹配 blob 到网格"""

    rows, cols = grid_points.shape[:2]

    

    # 收集有效网格点

    valid_grids = []

    valid_indices = []

    for row in range(rows):

        for col in range(cols):

            if valid_mask[row, col]:

                valid_grids.append(grid_points[row, col])

                valid_indices.append((row, col))

    

    if len(valid_grids) == 0 or len(blob_points) == 0:

        return None, None, np.zeros((rows, cols), dtype=bool)

    

    valid_grids = np.array(valid_grids)

    valid_indices = np.array(valid_indices)

    

    # 计算距离矩阵

    n_grids = len(valid_grids)

    n_blobs = len(blob_points)

    

    cost_matrix = np.zeros((n_grids, n_blobs), dtype=np.float32)

    

    for i in range(n_grids):

        for j in range(n_blobs):

            dist = np.linalg.norm(valid_grids[i] - blob_points[j])

            cost_matrix[i, j] = dist if dist <= max_distance else 1e6

    

    # 贪心匹配

    grid_indices, blob_indices = simple_greedy_matching(cost_matrix, threshold=max_distance)

    

    # 构建结果

    matched_corners = []

    matched_indices = []

    match_mask = np.zeros((rows, cols), dtype=bool)

    

    for grid_idx, blob_idx in zip(grid_indices, blob_indices):

        if cost_matrix[grid_idx, blob_idx] < max_distance:

            row, col = valid_indices[grid_idx]

            matched_corners.append(blob_points[blob_idx])

            matched_indices.append([row, col])

            match_mask[row, col] = True

    

    if len(matched_corners) == 0:

        return None, None, match_mask

    

    matched_corners = np.array(matched_corners, dtype=np.float32).reshape(-1, 1, 2)

    matched_indices = np.array(matched_indices, dtype=np.int32)

    

    return matched_corners, matched_indices, match_mask



def refine_grid_with_homography(grid_points, matched_indices, 

                                matched_corners, pattern_size=(15, 15)):

    """

    使用单应性矩阵优化网格位置

    这是解决透视畸变和长距离误差累积的关键步骤

    """

    if len(matched_corners) < 4:

        print("⚠️ 匹配点太少，无法计算单应性矩阵")

        return grid_points, None



    print(f"\n[单应性矩阵优化]")

    

    # 1. 准备数据

    # src_points: 网格的逻辑坐标 (0,0), (0,1), (1,0)... 

    # dst_points: 对应的图像像素坐标

    src_points = []

    dst_points = []

    

    for i in range(len(matched_indices)):

        row, col = matched_indices[i]

        # 使用逻辑坐标，单位可以是 1，这代表"格子"

        # 注意：这里我们定义 (col, row) 对应 (x, y)

        src_points.append([col, row]) 

        

        # 修复点：这里原来有乱码，现在修正为提取坐标

        # matched_corners 的形状是 (N, 1, 2)，所以取 [i,0] 得到 [x, y]

        dst_points.append(matched_corners[i, 0])



    src_points = np.array(src_points, dtype=np.float32)

    dst_points = np.array(dst_points, dtype=np.float32)



    # 2. 计算单应性矩阵 H

    # RANSAC 可以剔除错误的匹配点

    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    

    if H is None:

        print("❌ 单应性矩阵计算失败")

        return grid_points, None

        

    print(f"  单应性矩阵计算成功")



    # ==================== ### 替换原来的误差计算部分 ### ====================

    # 1. 投影所有匹配点（包含离群点）

    # src_points 形状 (N, 2), dst_points 形状 (N, 2)

    src_points_reshaped = src_points.reshape(-1, 1, 2)

    dst_points_reshaped = dst_points.reshape(-1, 1, 2)

    

    # 使用 H 矩阵预测这些网格点应该在图像的什么位置

    projected_points = cv2.perspectiveTransform(src_points_reshaped, H)

    

    # 2. 计算每个点的误差向量 (dx, dy) 和 欧氏距离

    diff_vectors = projected_points - dst_points_reshaped

    all_errors = np.linalg.norm(diff_vectors, axis=2).flatten() # 展平为一维数组

    

    # 3. 找出误差最大的那个点 (The Worst Match)

    max_error_idx = np.argmax(all_errors)

    max_error_val = all_errors[max_error_idx]

    

    # 获取该点的具体信息

    bad_grid_pos = matched_indices[max_error_idx] # (Row, Col)

    bad_pixel_pos = dst_points[max_error_idx]     # (x, y)

    should_be_pos = projected_points[max_error_idx][0] # H 预测的位置

    

    print(f"  --------------------------------")

    print(f"  [🔍 异常点定位]")

    print(f"  ❌ 误差最大的匹配点索引: {max_error_idx}")

    print(f"     网格坐标: 第 {bad_grid_pos[0]} 行, 第 {bad_grid_pos[1]} 列")

    print(f"     实际检测位置: ({bad_pixel_pos[0]:.1f}, {bad_pixel_pos[1]:.1f})")

    print(f"     理论预测位置: ({should_be_pos[0]:.1f}, {should_be_pos[1]:.1f})")

    print(f"     偏移距离: {max_error_val:.2f} px")

    print(f"  --------------------------------")



    # 4. 把这个异常点的信息存入 info 字典，以便后续可视化 (可选，为了简单这里直接打印)

    # 如果你想在可视化里画出来，可以将 bad_grid_pos 记下来



    # 5. 接着做常规统计（只统计 RANSAC 认可的内点）

    inlier_mask = mask.ravel() == 1

    valid_errors = all_errors[inlier_mask]

    

    if len(valid_errors) > 0:

        clean_rms = np.sqrt(np.mean(valid_errors ** 2))

        clean_mean = np.mean(valid_errors)

        print(f"  [剔除离群点后的统计]")

        print(f"  ✅ 平均误差: {clean_mean:.4f} px")

        print(f"  ✅ RMS误差:  {clean_rms:.4f} px (如果 < 1.0 则非常完美)")

    # ==================== ### 修改结束 ### ====================



    # 3. 使用 H 重新投影所有理论网格点

    rows, cols = pattern_size

    

    # 生成所有逻辑网格点 (col, row)

    all_grid_logical = []

    for r in range(rows):

        for c in range(cols):

            all_grid_logical.append([c, r])

    

    all_grid_logical = np.array(all_grid_logical, dtype=np.float32).reshape(-1, 1, 2)

    

    # 透视变换

    refined_points_flat = cv2.perspectiveTransform(all_grid_logical, H)

    

    # 重塑回 (rows, cols, 2)

    refined_grid_points = refined_points_flat.reshape(rows, cols, 2)

    

    return refined_grid_points, H



def match_and_fill_missing_points(refined_grid_points, all_blobs, 

                                  search_radius=20.0, pattern_size=(15, 15)):

    """

    智能补全网格：

    遍历每一个理论网格点，在 search_radius 范围内寻找最近的真实 Blob。

    - 如果找到：使用真实 Blob 坐标，标记为 'found'

    - 如果没找到：使用理论网格坐标，标记为 'interpolated' (补全)

    """

    rows, cols = pattern_size

    

    # 最终的网格坐标 (15, 15, 2)

    final_grid = np.zeros_like(refined_grid_points)

    # 状态掩码: 0=补全点(虚拟), 1=真实点

    status_mask = np.zeros((rows, cols), dtype=np.uint8)

    # 记录哪些 blob 被使用了，避免重复使用

    used_blobs = set()

    

    # 构建 blob 的 KDTree 以加速搜索 (如果点少，暴力搜也可以，这里用简单的暴力搜配合距离排序)

    # 为了保证最近邻优先，我们对每个网格点计算所有 blob 的距离

    

    matched_count = 0

    interpolated_count = 0

    

    for r in range(rows):

        for c in range(cols):

            theoretical_pt = refined_grid_points[r, c]

            

            # 计算该理论点到所有 blob 的距离

            if len(all_blobs) > 0:

                dists = np.linalg.norm(all_blobs - theoretical_pt, axis=1)

                

                # 找到最近的 blob

                min_dist_idx = np.argmin(dists)

                min_dist = dists[min_dist_idx]

                

                # 判断是否在允许范围内，且该 blob 未被使用

                if min_dist < search_radius and min_dist_idx not in used_blobs:

                    # === 找到真实点 ===

                    final_grid[r, c] = all_blobs[min_dist_idx]

                    status_mask[r, c] = 1 # 标记为真实

                    used_blobs.add(min_dist_idx)

                    matched_count += 1

                else:

                    # === 没找到，使用理论点补全 ===

                    final_grid[r, c] = theoretical_pt

                    status_mask[r, c] = 0 # 标记为补全

                    interpolated_count += 1

            else:

                # 如果没有 blob，全部补全

                final_grid[r, c] = theoretical_pt

                status_mask[r, c] = 0

                interpolated_count += 1

                

    return final_grid, status_mask, matched_count, interpolated_count

def calculate_pose_and_euler(grid_points, image_shape, spacing_px, pattern_size=(15, 15)):
    """
    估算位姿、计算欧拉角、准备坐标轴绘制数据
    """
    rows, cols = pattern_size
    h, w = image_shape[:2]
    
    # 1. 估算相机内参 (如果未知)
    # 假设焦距 f 约为图像宽度的 1.0 - 1.2 倍，光心在图像中心
    focal_length = w * 1.0 
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1)) # 假设无畸变 (用于可视化足够)

    # 2. 定义世界坐标系中的对象点 (3D)
    # 我们定义 Ref 点 (0,0) 为原点，Z=0
    # X轴沿着列增加的方向，Y轴沿着行增加的方向 (符合 AprilTag 逻辑)
    obj_points = []
    img_points = []
    
    for r in range(rows):
        for c in range(cols):
            # 世界坐标: x=col*spacing, y=row*spacing, z=0
            obj_points.append([c * spacing_px, r * spacing_px, 0])
            img_points.append(grid_points[r, c])
            
    obj_points = np.array(obj_points, dtype=np.float64)
    img_points = np.array(img_points, dtype=np.float64)
    
    # 3. Solve PnP 获取旋转和平移
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
    
    if not success:
        return None
    
    # 4. 计算坐标轴的终点 (投影回 2D)
    # 轴长度设为 3 倍间距
    axis_length = spacing_px * 3
    axis_3d = np.float32([
        [axis_length, 0, 0],   # X轴终点
        [0, axis_length, 0],   # Y轴终点
        [0, 0, -axis_length]   # Z轴终点 (指向相机/上方)
    ]).reshape(-1, 3)
    
    axis_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
    origin_2d, _ = cv2.projectPoints(np.float32([[0, 0, 0]]), rvec, tvec, camera_matrix, dist_coeffs)
    
    origin = tuple(origin_2d[0].ravel().astype(int))
    
    # 5. 计算欧拉角
    rmat, _ = cv2.Rodrigues(rvec) # 旋转向量转矩阵
    
    # 计算 Euler Angles (假设 XYZ 顺序: Pitch-Yaw-Roll)
    # sy = sqrt(R00*R00 + R10*R10)
    sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(rmat[2, 1], rmat[2, 2])
        y = np.arctan2(-rmat[2, 0], sy)
        z = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        x = np.arctan2(-rmat[1, 2], rmat[1, 1])
        y = np.arctan2(-rmat[2, 0], sy)
        z = 0

    # 转换为角度
    pitch = x * 180 / np.pi
    yaw = y * 180 / np.pi
    roll = z * 180 / np.pi
    
    return {
        'origin': origin,
        'axis_points': axis_2d, # shape (3, 1, 2)
        'roll': roll,
        'pitch': pitch,
        'yaw': yaw,
        'tvec': tvec
    }

def detect_grid_with_apriltag(image, max_distance=100.0):
    """
    正确的检测流程 v4.1 (修复 KeyError: 'rms_error')
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print("="*80)
    print("AprilTag 引导网格检测（V4.1 - 完整修复版）")
    print("="*80)
    
    # 1. 检测 AprilTag
    apriltag_info = detect_apriltag(gray)
    if apriltag_info is None:
        print("❌ 未检测到 AprilTag")
        return None
    
    # 2. 检测所有圆点 blob
    blob_points, keypoints = detect_blobs(gray)
    if len(blob_points) == 0:
        print("❌ 未检测到圆点")
        return None
    
    # 3. 排除 AprilTag 区域的 blob
    filtered_blob_points, excluded_indices = filter_blobs_exclude_apriltag(
        blob_points, apriltag_info
    )
    
    # 创建过滤后的 keypoints
    valid_indices = [i for i in range(len(blob_points)) if i not in excluded_indices]
    filtered_keypoints = [keypoints[i] for i in valid_indices]
    
    # 4. 找四个角点
    corners_dict, corners_indices_dict = find_four_corner_blobs(
        filtered_blob_points, image.shape
    )
    
    if corners_dict is None:
        print("❌ 未找到角点")
        return None
    
    # 5. 从四个角点中选择最靠近 AprilTag 的
    corner_blob, corner_idx_filtered, corner_name = find_corner_near_apriltag(
        corners_dict, corners_indices_dict, apriltag_info
    )
    
    corner_idx_original = valid_indices[corner_idx_filtered]
    
    # 6. 从 AprilTag 获取方向
    unit_x, unit_y = get_grid_orientation_from_apriltag(apriltag_info)
    
    # 7. 估算圆点间距
    spacing_px = estimate_grid_spacing_robust(filtered_blob_points)
    if spacing_px is None:
        print("❌ 无法估算圆点间距")
        return None
    
    # 8. 根据角点位置计算网格原点
    grid_origin, corner_grid_pos = determine_grid_origin_from_corner(
        corner_name, corner_blob, unit_x, unit_y, spacing_px,
        pattern_size=(15, 15)
    )
    
    # 9. 构建初始线性网格
    coarse_grid_points, _ = build_grid_from_origin(
        grid_origin, unit_x, unit_y, spacing_px,
        pattern_size=(15, 15),
        image_shape=image.shape
    )
    
    # 10. 初始匹配（为了计算单应性矩阵）
    dummy_mask = np.ones((15, 15), dtype=bool)
    matched_corners, matched_indices, match_mask = match_blobs_to_grid(
        filtered_blob_points, coarse_grid_points, dummy_mask, max_distance=max_distance
    )
    
    if matched_corners is None or len(matched_corners) < 4:
        print("❌ 初始匹配点不足，无法计算单应性")
        return None

    # 11. 计算单应性矩阵并生成“完美理论网格”
    theoretical_grid_points, H = refine_grid_with_homography(
        coarse_grid_points, matched_indices, matched_corners
    )
    
    if H is None:
        return None
        
    # 12. 智能补全匹配 (Smart Fill)
    search_radius = spacing_px * 0.4
    print(f"  正在进行智能补全 (半径: {search_radius:.1f} px)...")
    
    final_grid_points, status_mask, n_found, n_interp = match_and_fill_missing_points(
        theoretical_grid_points, filtered_blob_points, 
        search_radius=search_radius, pattern_size=(15, 15)
    )
    
    # ==================== 【关键修复：计算 RMS 并添加到字典】 ====================
    # 13. 计算 RMS 误差（仅基于真实点）
    real_indices = np.where(status_mask == 1)
    if len(real_indices[0]) > 0:
        real_pts_detected = final_grid_points[real_indices]
        real_pts_theoretical = theoretical_grid_points[real_indices]
        
        error_vectors = real_pts_detected - real_pts_theoretical
        errors = np.linalg.norm(error_vectors, axis=1)
        rms_error = np.sqrt(np.mean(errors**2))
    else:
        rms_error = 0.0
        
    print(f"  ✅ 真实点: {n_found}, 补全点: {n_interp}")
    print(f"  📊 RMS 误差: {rms_error:.4f} px")
    # ========================================================================
    
    match_count = n_found
    valid_count = 15 * 15
    match_rate = match_count / valid_count * 100
    
    print(f"\n{'='*80}")
    print(f"✅ 处理完成: {match_count}/{valid_count} ({match_rate:.1f}%)")
    print(f"{'='*80}\n")
    
    return {
        'success': True,
        'final_grid_points': final_grid_points,
        'theoretical_grid_points': theoretical_grid_points,
        'status_mask': status_mask,
        'rms_error': rms_error,           # <--- 这里必须有！
        'matched_corners': matched_corners,
        'matched_indices': matched_indices,
        'grid_points': theoretical_grid_points,
        'valid_mask': np.ones((15,15), dtype=bool),
        'match_mask': status_mask.astype(bool),
        'blob_points': filtered_blob_points,
        'keypoints': filtered_keypoints,
        'all_blob_points': blob_points,
        'all_keypoints': keypoints,
        'excluded_indices': excluded_indices,
        'apriltag_info': apriltag_info,
        'corner_blob': corner_blob,
        'corner_idx': corner_idx_original,
        'corner_name': corner_name,
        'corner_grid_pos': corner_grid_pos,
        'four_corners': corners_dict,
        'match_count': match_count,
        'valid_count': valid_count,
        'match_rate': match_rate,
        'spacing_px': spacing_px,
        'homography_matrix': H
    }


def visualize_result(image, result):
    """
    可视化 V2: 
    - 绘制 XYZ 坐标轴 (X紫, Y红, Z蓝)
    - 绘制理论中心 vs 实际中心
    - 显示 Roll/Pitch/Yaw 和 Error
    """
    vis = image.copy()
    h, w = vis.shape[:2]
    
    grid = result['final_grid_points']
    mask = result['status_mask']
    rows, cols = grid.shape[:2]
    
    # 1. 基础网格绘制 (黄色点 & 蓝色十字)
    for r in range(rows):
        for c in range(cols):
            pt = tuple(grid[r, c].astype(int))
            if mask[r, c] == 1:
                # 真实点: 黄色实心
                cv2.circle(vis, pt, 4, (0, 255, 255), -1) 
            else:
                # 补全点: 蓝色十字
                cv2.drawMarker(vis, pt, (255, 0, 0), cv2.MARKER_CROSS, 12, 1)

    # 2. 计算中心点 (Centroids)
    # 理论中心: 整个 15x15 网格的几何中心 (包含补全点)
    theo_center = np.mean(grid, axis=(0, 1)).astype(int)
    
    # 实际中心: 仅计算实际检测到的真实点 (status_mask == 1)
    real_indices = np.where(mask == 1)
    if len(real_indices[0]) > 0:
        real_points = grid[real_indices]
        real_center = np.mean(real_points, axis=0).astype(int)
    else:
        real_center = theo_center

    # 绘制中心点
    # 理论中心: 白色空心圆 + 十字
    cv2.circle(vis, tuple(theo_center), 10, (255, 255, 255), 2)
    cv2.drawMarker(vis, tuple(theo_center), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(vis, "Theo Center", (theo_center[0]+15, theo_center[1]-15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 实际中心: 绿色实心圆
    cv2.circle(vis, tuple(real_center), 6, (0, 255, 0), -1)
    cv2.putText(vis, "Real Center", (real_center[0]+15, real_center[1]+15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 3. 绘制 3D 坐标轴 (Pose Estimation)
    # 原点为 Ref 点 (0,0)
    corner_r, corner_c = result['corner_grid_pos']
    
    # 为了保证 PnP 计算的稳定性，我们传入完整的 final_grid_points
    # 并使用估算的内参
    pose_data = calculate_pose_and_euler(
        grid, image.shape, result['spacing_px'], pattern_size=(rows, cols)
    )
    
    if pose_data:
        origin = pose_data['origin']
        axis_pts = pose_data['axis_points']
        
        # 提取轴的终点
        pt_x = tuple(axis_pts[0].ravel().astype(int))
        pt_y = tuple(axis_pts[1].ravel().astype(int))
        pt_z = tuple(axis_pts[2].ravel().astype(int))
        
        # 绘制轴线 (粗细=3)
        # X轴 (紫色: BGR -> 255, 0, 255)
        cv2.line(vis, origin, pt_x, (255, 0, 255), 3) 
        cv2.putText(vis, "X", pt_x, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        # Y轴 (红色: BGR -> 0, 0, 255)
        cv2.line(vis, origin, pt_y, (0, 0, 255), 3)
        cv2.putText(vis, "Y", pt_y, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Z轴 (蓝色: BGR -> 255, 0, 0)
        cv2.line(vis, origin, pt_z, (255, 0, 0), 3)
        cv2.putText(vis, "Z", pt_z, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # 在原点标注 Ref
        cv2.circle(vis, origin, 5, (0, 0, 0), -1) # 黑点核心
        cv2.putText(vis, "Ref(Origin)", (origin[0]-40, origin[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 4. 信息面板 (HUD)
    # 准备数据
    roll = pose_data['roll'] if pose_data else 0
    pitch = pose_data['pitch'] if pose_data else 0
    yaw = pose_data['yaw'] if pose_data else 0
    
    info_lines = [
        f"GRID DETECTOR V4.1",
        f"-----------------",
        f"Tag ID: {result['apriltag_info']['tag_id']}",
        f"RMS Error: {result['rms_error']:.3f} px",
        f"Max Error: {np.max(np.linalg.norm(grid - result['theoretical_grid_points'], axis=2)):.1f} px",
        f"-----------------",
        f"Roll  (X): {roll:.1f} deg",
        f"Pitch (Y): {pitch:.1f} deg",
        f"Yaw   (Z): {yaw:.1f} deg",
        f"-----------------",
        f"Found: {np.sum(mask)}",
        f"Interp: {np.sum(1-mask)}"
    ]
    
    # 绘制背景板
    line_h = 24
    pad = 15
    h_panel = len(info_lines) * line_h + pad * 2
    w_panel = 260
    
    # 确保不越界
    h_panel = min(h_panel, h)
    
    sub_img = vis[0:h_panel, 0:w_panel]
    black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 0.4, black_rect, 0.6, 1.0)
    vis[0:h_panel, 0:w_panel] = res
    
    # 绘制文字
    for i, line in enumerate(info_lines):
        y = pad + (i + 1) * line_h - 5
        # 根据内容变色
        color = (255, 255, 255)
        if "Error" in line:
            val = float(line.split(':')[1].split('p')[0])
            if val > 3.0: color = (0, 0, 255) # 红色警告
            elif val < 1.0: color = (0, 255, 0) # 绿色优秀
        elif "Roll" in line: color = (255, 0, 255) # 紫色
        elif "Pitch" in line: color = (100, 100, 255) # 浅红
        elif "Yaw" in line: color = (255, 100, 100) # 浅蓝
            
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    return vis




if __name__ == '__main__':

    image = cv2.imread('data/1764744101_27_picture.png')

    

    if image is None:

        print("无法读取图像")

        exit(1)

    

    # 检测

    result = detect_grid_with_apriltag(image, max_distance=100.0)

    

    if result and result['success']:

        # 可视化

        vis = visualize_result(image, result)

        

        cv2.imwrite('correct_apriltag_result.png', vis)

        cv2.imshow('Correct AprilTag Guided Detection', vis)

        

        print("按任意键关闭...")

        cv2.waitKey(0)

        cv2.destroyAllWindows()

    else:

        print("检测失败")

